// Top-part scheduler (see top_scheduler.hpp).

#include "top_scheduler.hpp"

#include "smf/solve_pool.hpp"

#include <algorithm>
#include <chrono>
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

TopScheduler::TopScheduler(std::vector<Int> parent, int threads)
    : parent_(std::move(parent)), pending_(parent_.size(), 0) {
  const std::size_t n = parent_.size();
  for (std::size_t i = 0; i < n; ++i)
    if (parent_[i] >= 0) ++pending_[static_cast<std::size_t>(parent_[i])];
  for (std::size_t i = 0; i < n; ++i)
    if (pending_[i] == 0) ready_.push_back(static_cast<Int>(i));
  completed_.reserve(n);
  free_ = std::max(1, threads);
  remaining_ = static_cast<int>(n);
  publish();
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
    t.node = ready_.back();
    ready_.pop_back();
    t.threads = std::max(1, free_ / static_cast<int>(ready_.size() + 1));
    free_ -= t.threads;
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
  free_ += threads;
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
  if (p >= 0 && --pending_[static_cast<std::size_t>(p)] == 0) {
    // Continuation: the parent's share is computed as if it were taken from
    // the list, among the nodes ready there.
    next = p;
    threads = std::max(1, free_ / static_cast<int>(ready_.size() + 1));
    free_ -= threads;
  }
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

PassLanes::PassLanes(int lanes) : busy_(static_cast<std::size_t>(std::max(1, lanes)), 0) {}

void PassLanes::release(int lane) {
  if (lane < 0) return;
  std::lock_guard<std::mutex> lk(mtx_);
  busy_[static_cast<std::size_t>(lane)] = 0;
}

} // namespace smf
