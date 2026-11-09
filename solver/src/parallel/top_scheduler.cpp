// Top-part scheduler (see top_scheduler.hpp).

#include "top_scheduler.hpp"

#include "smf/solve_pool.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>

namespace smf {

namespace {
// An idle worker polls this long before it sleeps: long enough to bridge
// the gap between two small fronts, short enough that workers left idle by
// a big front (all BLAS threads on it) soon stop burning a core. As in
// solve_pool.cpp.
constexpr auto kSpin = std::chrono::microseconds(50);

// Spins until ready() or kSpin elapses; true when ready() held.
template <class Pred> bool spin_until(Pred ready) {
  const auto t0 = std::chrono::steady_clock::now();
  for (unsigned i = 1;; ++i) {
    if (ready()) return true;
    solve_pool_relax();
    if ((i & 255u) == 0 && std::chrono::steady_clock::now() - t0 > kSpin) return ready();
  }
}
} // namespace

TopScheduler::TopScheduler(std::vector<Int> parent, int threads, std::vector<double> priority,
                           std::vector<Int> external, int joined, std::vector<char> early)
    : parent_(std::move(parent)), pending_(parent_.size(), 0), share_(parent_.size()) {
  const std::size_t n = parent_.size();
  if (priority.size() == n) prio_ = std::move(priority);
  if (early.size() == n) early_ = std::move(early);
  for (std::size_t i = 0; i < n; ++i)
    if (parent_[i] >= 0) ++pending_[static_cast<std::size_t>(parent_[i])];
  if (external.size() == n)
    for (std::size_t i = 0; i < n; ++i) {
      pending_[i] += external[i];
      external_left_ += external[i];
    }
  for (std::size_t i = 0; i < n; ++i)
    if (pending_[i] == 0) make_ready(static_cast<Int>(i));
  completed_.reserve(n);
  threads_ = std::max(1, threads);
  joined_ = joined < 0 ? threads_ : std::min(joined, threads_);
  free_ = joined_;
  remaining_ = static_cast<int>(n);
  publish();
}

void TopScheduler::external_done(const Int *nodes, std::size_t count) {
  if (count == 0) return;
  std::lock_guard<std::mutex> lk(mtx_);
  for (std::size_t k = 0; k < count; ++k) {
    const std::size_t v = static_cast<std::size_t>(nodes[k]);
    --external_left_;
    if (--pending_[v] == 0) make_ready(static_cast<Int>(v));
  }
  if (external_left_ == 0) {
    for (const Int v : held_) push_ready(v);
    held_.clear();
  }
  grant_free();
  publish();
  wake_one_if_runnable();
}

void TopScheduler::join(int k) {
  std::lock_guard<std::mutex> lk(mtx_);
  k = std::clamp(k, 0, threads_ - joined_);
  joined_ += k;
  free_ += k;
  grant_free();
  publish();
  wake_one_if_runnable();
}

void TopScheduler::stop() {
  stopped_.store(true, std::memory_order_release);
  std::lock_guard<std::mutex> lk(mtx_);
  publish();
  cv_.notify_all();
}

namespace {
// Max-heap order on (priority, id).
struct ByPriority {
  const std::vector<double> *p;
  bool operator()(Int a, Int b) const {
    const double pa = (*p)[static_cast<std::size_t>(a)], pb = (*p)[static_cast<std::size_t>(b)];
    return pa < pb || (pa == pb && a < b);
  }
};
} // namespace

void TopScheduler::make_ready(Int v) {
  if (held(v))
    held_.push_back(v);
  else
    push_ready(v);
}

void TopScheduler::push_ready(Int v) {
  ready_.push_back(v);
  if (prio_.empty()) return;
  ready_prio_ += prio_[static_cast<std::size_t>(v)];
  std::push_heap(ready_.begin(), ready_.end(), ByPriority{&prio_});
}

Int TopScheduler::pop_ready() {
  if (!prio_.empty()) std::pop_heap(ready_.begin(), ready_.end(), ByPriority{&prio_});
  const Int v = ready_.back();
  ready_.pop_back();
  if (!prio_.empty()) {
    ready_prio_ -= prio_[static_cast<std::size_t>(v)];
    if (ready_.empty()) ready_prio_ = 0.0;  // no drift
  }
  return v;
}

int TopScheduler::start_share(double p) const {
  const int k = static_cast<int>(ready_.size());  // the others still listed
  const int cap = std::max(1, free_ - k);
  const double total = p + ready_prio_;
  int s = total > 0.0 ? static_cast<int>(std::floor(static_cast<double>(free_) * p / total))
                      : free_ / (k + 1);
  return std::clamp(s, 1, cap);
}

void TopScheduler::grant_free() {
  // Not while a subtree task may still ready a node, which then starts on them.
  if (prio_.empty() || free_ <= 0 || !ready_.empty() || running_.empty() || external_left_ > 0) return;
  Int best = running_.front();
  for (const Int v : running_)
    if (prio_[static_cast<std::size_t>(v)] > prio_[static_cast<std::size_t>(best)] ||
        (prio_[static_cast<std::size_t>(v)] == prio_[static_cast<std::size_t>(best)] && v > best))
      best = v;
  share_[static_cast<std::size_t>(best)].fetch_add(free_, std::memory_order_release);
  free_ = 0;
}

int TopScheduler::shares_in_use() {
  std::lock_guard<std::mutex> lk(mtx_);
  return joined_ - free_;
}

void TopScheduler::publish() {
  nready_.store(static_cast<int>(ready_.size()), std::memory_order_release);
  nfree_.store(free_, std::memory_order_release);
  noffers_.store(static_cast<int>(offers_.size()), std::memory_order_release);
  nremaining_.store(remaining_, std::memory_order_release);
}

bool TopScheduler::runnable_hint() const {
  return (nready_.load(std::memory_order_acquire) > 0 && nfree_.load(std::memory_order_acquire) > 0) ||
         noffers_.load(std::memory_order_acquire) > 0 ||
         nremaining_.load(std::memory_order_acquire) == 0 || stopped_.load(std::memory_order_acquire);
}

bool TopScheduler::runnable() const {
  return (!ready_.empty() && free_ > 0) || !offers_.empty() || remaining_ == 0 ||
         stopped_.load(std::memory_order_acquire);
}

// One sleeper per thing to take: a worker that takes something passes the
// wake-up on if more is left (acquire), so a burst of work wakes the
// sleepers one after another rather than all at once to fight for one node.
void TopScheduler::wake_one_if_runnable() {
  if (sleepers_ > 0 && ((!ready_.empty() && free_ > 0) || !offers_.empty())) cv_.notify_one();
}

TopScheduler::Task TopScheduler::acquire() {
  // Spin first: the next node or offer usually appears within microseconds,
  // well before a sleep / wake-up round trip would complete.
  spin_until([&] { return runnable_hint(); });
  std::unique_lock<std::mutex> lk(mtx_);
  if (!runnable()) {
    // Counted under the lock, which every producer holds when it checks
    // sleepers_, so a wake-up cannot be missed.
    ++sleepers_;
    cv_.wait(lk, [&] { return runnable(); });
    --sleepers_;
  }
  Task t;
  if (stopped_.load(std::memory_order_acquire) || remaining_ == 0) return t;
  if (!ready_.empty() && free_ > 0) {
    t.node = pop_ready();
    if (prio_.empty()) {
      t.threads = std::max(1, free_ / static_cast<int>(ready_.size() + 1));
    } else {
      t.threads = start_share(prio_[static_cast<std::size_t>(t.node)]);
      running_.push_back(t.node);
    }
    share_[static_cast<std::size_t>(t.node)].store(t.threads, std::memory_order_release);
    free_ -= t.threads;
    grant_free();
    ++queue_pops_;
  } else {
    // Tile helper: the newest offer (runnable() holds, so there is one).
    auto &o = offers_.back();
    t.job = o.first;
    if (--o.second == 0) offers_.pop_back();
    // Under our lock, so that a withdraw() after it is final.
    t.job->add_helper();
  }
  publish();
  wake_one_if_runnable();
  return t;
}

Int TopScheduler::finish(Int node, int tid, bool ok, int &threads) {
  // Visible to spinning and locking workers at once; no node is handed out
  // after this store (acquire and the continuation below check it under
  // the lock).
  if (!ok) stopped_.store(true, std::memory_order_release);
  std::lock_guard<std::mutex> lk(mtx_);
  if (prio_.empty()) {
    free_ += threads;
  } else {
    // Its current share: grants may have grown it since it started.
    free_ += share_[static_cast<std::size_t>(node)].load(std::memory_order_acquire);
    running_.erase(std::find(running_.begin(), running_.end(), node));
  }
  Int next = -1;
  if (!ok) {
    publish();
    cv_.notify_all();
    return next;
  }
  completed_.emplace_back(node, tid);
  --remaining_;
  if (remaining_ == 0) {
    publish();
    cv_.notify_all();
    return next;
  }
  // After another thread's stop: its notify_all has woken the sleepers.
  if (stopped_.load(std::memory_order_acquire)) return next;
  const Int p = parent_[static_cast<std::size_t>(node)];
  if (p >= 0 && --pending_[static_cast<std::size_t>(p)] == 0 && held(p)) {
    held_.push_back(p);
  } else if (p >= 0 && pending_[static_cast<std::size_t>(p)] == 0) {
    // Continuation: the parent's share is computed as if it were taken from
    // the list, among the nodes ready there.
    next = p;
    if (prio_.empty()) {
      threads = std::max(1, free_ / static_cast<int>(ready_.size() + 1));
    } else {
      threads = start_share(prio_[static_cast<std::size_t>(p)]);
      running_.push_back(p);
    }
    share_[static_cast<std::size_t>(p)].store(threads, std::memory_order_release);
    free_ -= threads;
  }
  grant_free();
  publish();
  // The threads this node released may let a sleeper take a listed node.
  wake_one_if_runnable();
  return next;
}

void TopScheduler::offer(TileRunner &job, int slots) {
  if (slots <= 0) return;
  std::lock_guard<std::mutex> lk(mtx_);
  offers_.emplace_back(&job, slots);
  publish();
  wake_one_if_runnable();
}

void TopScheduler::withdraw(TileRunner &job) {
  std::lock_guard<std::mutex> lk(mtx_);
  for (std::size_t i = 0; i < offers_.size(); ++i)
    if (offers_[i].first == &job) {
      offers_.erase(offers_.begin() + static_cast<std::ptrdiff_t>(i));
      break;
    }
  publish();
}

PassLanes::PassLanes(int lanes, bool all_busy)
    : busy_(static_cast<std::size_t>(std::max(1, lanes)), all_busy ? 1 : 0) {}

void PassLanes::release(int lane) {
  if (lane < 0) return;
  std::lock_guard<std::mutex> lk(mtx_);
  busy_[static_cast<std::size_t>(lane)] = 0;
}

} // namespace smf
