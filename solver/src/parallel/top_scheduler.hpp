#pragma once
// Scheduler of the top part of the assembly tree (phase 2 of the
// subtree-parallel factorization): a DAG of fronts on a worker pool.
//
// The top part is mostly long chains of small fronts: each takes a few
// microseconds, so the cost of handing a front to another thread (a
// condition-variable wake-up is tens of microseconds on Windows) must not be
// paid per front. Hence:
// - continuation: the thread whose completion readies a parent runs that
//   parent itself, so a chain stays on one thread (and its contribution
//   blocks in that core's cache) and never touches the shared list, which
//   holds only the nodes ready at the start;
// - an idle worker spins briefly on lock-free mirrors of the state before
//   it sleeps, and a wake-up goes to one sleeper, only when there is one and
//   only when there is something it can take.
//
// No factorization knowledge here: nodes are local ids 0 .. n - 1 and
// run(node, blas_threads, tid) does the work. PassLanes (below) hands the
// passes' buffers to the chains.

#include "tile_tasks.hpp"

#include "smf/types.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <utility>
#include <vector>

namespace smf {

// Runs a DAG of top-part fronts on a pool. A node is ready once all its
// children have completed. Ready nodes share `threads` BLAS threads: a node
// taken while k others are ready gets max(1, free / (k + 1)), and the sum of
// the shares of running nodes never exceeds `threads` (the threads joined
// so far, with the overlap below).
//
// With priorities (Control::top_critical_shares, 8T engine Part 7; one value
// per node, its remaining path cost to the root):
// - the ready list is taken highest priority first (ties: the higher id);
// - a node taken with F free threads among the ready set R (itself
//   included; a continuation counts the listed nodes and itself) starts with
//   clamp(floor(F * p / sum_R p), 1, max(1, F - (|R| - 1))): every other
//   ready node can still get one;
// - threads freed while nothing is ready go to the running node with the
//   highest priority: its share (share(node), an atomic the front's BLAS
//   calls follow) grows, and finish() returns its current share.
//
// Overlap with the subtree phase (Control::top_overlap): a node may also
// wait for `external[i]` completions of work outside the DAG (its subtree
// tasks), reported by external_done(), and the budget may start with
// `joined` < threads threads, the others added by join() as the workers that
// hold them come to work(). While any external completion is outstanding,
// free threads are not granted to running nodes: a node readied by a
// subtree task can still start on them. A node with early[i] == 0 (a big
// front: on the few threads joined so far it would run far slower than
// once all have) is held, neither listed nor continued, until the last
// external completion.
class TopScheduler final : public TileHelpers {
public:
  // parent[i]: parent of local node i (-1: root); pending counts derive from it.
  // priority: empty (equal shares, LIFO list) or one value per node.
  // external: empty (none) or per node its external completions to wait for.
  // joined: threads in the budget at the start (-1: all `threads`).
  // early: empty (every node) or per node 1 when it may start while external
  // completions are outstanding.
  TopScheduler(std::vector<Int> parent, int threads, std::vector<double> priority = {},
               std::vector<Int> external = {}, int joined = -1, std::vector<char> early = {});
  TopScheduler(const TopScheduler &) = delete;
  TopScheduler &operator=(const TopScheduler &) = delete;

  // One external completion of each listed node (a node may be listed more
  // than once). A node with nothing else pending becomes ready.
  void external_done(const Int *nodes, std::size_t count);
  // `k` more threads in the budget (at most `threads` in all).
  void join(int k = 1);
  // Stop: no node is taken after this, and every work() returns.
  void stop();

  // Worker loop for thread `tid`. run(node, blas_threads, tid) -> bool
  // (false: stop everything). Returns when every node completed or after a stop.
  // chain_end(tid) follows the last node of each chain the thread ran (a
  // node taken from the shared list, then its continuations), stop included.
  template <class Run, class ChainEnd> void work(int tid, Run &&run, ChainEnd &&chain_end);
  template <class Run> void work(int tid, Run &&run) {
    work(tid, run, [](int) {});
  }

  // TileHelpers: a running front lends `slots` places to idle workers.
  void offer(TileRunner &job, int slots) override;
  void withdraw(TileRunner &job) override;

  // Completion order (topological), and how many nodes were taken from the
  // shared ready list rather than continued by the thread that readied them.
  // Read after every work() returned.
  const std::vector<std::pair<Int, int>> &completed() const { return completed_; }  // (node, tid)
  int queue_pops() const { return queue_pops_; }
  bool stopped() const { return stopped_.load(std::memory_order_acquire); }

  // The live BLAS share of a node while run(node, ...) runs (with
  // priorities it grows when threads come free).
  const std::atomic<int> &share(Int node) const { return share_[static_cast<std::size_t>(node)]; }
  // Threads held by running nodes (joined - free). For tests.
  int shares_in_use();

private:
  // What an idle worker gets: a node and its BLAS-thread share, a tile job
  // to help (a slot already taken), or neither (all done, or stopped).
  struct Task {
    Int node = -1;
    int threads = 0;
    TileRunner *job = nullptr;
  };

  Task acquire();
  // Records the end of `node` (ok: completed; false: stop) and returns the
  // parent it readied, which the caller runs next with `threads` updated to
  // its share, or -1.
  Int finish(Int node, int tid, bool ok, int &threads);

  // Under mtx_, with priorities: the start share of a node of priority p
  // taken with the listed nodes still ready, and the grant of free threads
  // to the highest-priority running node when nothing is listed.
  int start_share(double p) const;
  void grant_free();
  void push_ready(Int v);
  void make_ready(Int v);  // listed, or held (held())
  Int pop_ready();

  bool runnable_hint() const;  // lock-free, for spinning
  bool runnable() const;       // under mtx_
  void publish();              // under mtx_: refresh the lock-free mirrors
  void wake_one_if_runnable(); // under mtx_

  std::vector<Int> parent_;
  std::vector<Int> pending_;   // children not completed yet
  std::vector<Int> ready_;     // shared list (LIFO, or a max-heap on priority): initially ready nodes only
  std::vector<double> prio_;   // empty: no priorities
  double ready_prio_ = 0.0;    // sum of the listed nodes' priorities
  std::vector<std::atomic<int>> share_;  // per node: its BLAS share while it runs
  std::vector<Int> running_;   // with priorities: the nodes running now
  int threads_ = 1;
  std::vector<std::pair<TileRunner *, int>> offers_;  // (job, slots left), newest last
  std::vector<std::pair<Int, int>> completed_;
  int free_ = 0;               // BLAS threads not held by running nodes
  int joined_ = 0;             // threads in the budget (free_ + the running shares)
  long long external_left_ = 0;  // external completions not reported yet
  std::vector<char> early_;    // empty: every node may start early
  std::vector<Int> held_;      // ready nodes waiting for the last external completion
  bool held(Int v) const {
    return external_left_ > 0 && !early_.empty() && early_[static_cast<std::size_t>(v)] == 0;
  }
  int remaining_ = 0;          // nodes not completed
  int sleepers_ = 0;           // workers blocked on cv_
  int queue_pops_ = 0;
  std::mutex mtx_;
  std::condition_variable cv_;
  // Mirrors of the state above, written under mtx_ and polled by spinning
  // workers without it (a hint only: what a worker takes is decided under
  // mtx_). stopped_ is set before the lock is taken, so that no thread
  // takes a node once a run() has returned false.
  std::atomic<int> nready_{0};
  std::atomic<int> nfree_{0};
  std::atomic<int> noffers_{0};
  std::atomic<int> nremaining_{0};
  std::atomic<bool> stopped_{false};
};

// The factorization's per-pass buffers in the top phase, handed to chains
// of fronts as lanes instead of being tied to threads. A pass's buffers
// grow to the largest fronts it has held and are kept across
// factorizations; growing one (a reallocation, then the first touch of
// every fresh page) costs tens of milliseconds for the root fronts of
// TSOPF_FS_b300_c3, more than those fronts' own work saves from threading.
// Which thread starts or continues a chain is a race, so passes tied to
// threads grow a fresh pass whenever the race goes another way; a chain
// instead takes, and moves to, a free lane that already has the room its
// next front needs. room(l) is lane l's room (the touched space past its
// fill); pick reads it for the lanes it does not hold under its lock, so a
// lane's holder is the only thread that changes its room.
class PassLanes {
public:
  // all_busy: every lane starts held (by the subtree phase, which releases
  // each when its thread is done with it).
  explicit PassLanes(int lanes, bool all_busy = false);
  PassLanes(const PassLanes &) = delete;
  PassLanes &operator=(const PassLanes &) = delete;

  // The lane on which a chain runs its next front, which needs `need` of
  // room. held: the chain's lane (-1: a chain start, which takes `home`
  // when that is free and has the room, else the free lane with the most
  // room). A held lane keeps the chain while it has the room; otherwise
  // the chain moves to the free lane with the most room when that one has
  // it (and releases `held`). Ties go to the lowest lane. A chain start
  // needs a free lane: at most `lanes` chains at once.
  template <class Room> int pick(int held, int home, std::size_t need, Room &&room);
  // A chain ended: its lane is free.
  void release(int lane);

private:
  std::mutex mtx_;
  std::vector<char> busy_;
};

template <class Run, class ChainEnd>
void TopScheduler::work(int tid, Run &&run, ChainEnd &&chain_end) {
  for (;;) {
    const Task t = acquire();
    if (t.job != nullptr) {
      t.job->help();
      continue;
    }
    if (t.node < 0) return;
    int threads = t.threads;
    for (Int v = t.node; v >= 0;) {
      const bool ok = static_cast<bool>(run(v, threads, tid));
      v = finish(v, tid, ok, threads);
    }
    chain_end(tid);
  }
}

template <class Room> int PassLanes::pick(int held, int home, std::size_t need, Room &&room) {
  // The common case, lock-free: the held lane has the room.
  if (held >= 0 && room(held) >= need) return held;
  const int n = static_cast<int>(busy_.size());
  std::lock_guard<std::mutex> lk(mtx_);
  if (held < 0 && home >= 0 && home < n && busy_[static_cast<std::size_t>(home)] == 0 &&
      room(home) >= need) {
    busy_[static_cast<std::size_t>(home)] = 1;
    return home;
  }
  int best = -1;
  std::size_t best_room = 0;
  for (int l = 0; l < n; ++l) {
    if (busy_[static_cast<std::size_t>(l)] != 0) continue;
    const std::size_t r = room(l);
    if (best < 0 || r > best_room) {
      best = l;
      best_room = r;
    }
  }
  if (held >= 0) {
    if (best < 0 || best_room < need) return held;  // grows the held lane
    busy_[static_cast<std::size_t>(held)] = 0;
  }
  // A chain start always finds a free lane (at most `lanes` chains run at
  // once); should that ever break, stop rather than index busy_[-1].
  if (best < 0) std::abort();
  busy_[static_cast<std::size_t>(best)] = 1;
  return best;
}

} // namespace smf
