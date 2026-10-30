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
// run(node, blas_threads, tid) does the work.

#include "tile_tasks.hpp"

#include "smf/types.hpp"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <utility>
#include <vector>

namespace smf {

// Runs a DAG of top-part fronts on a pool. A node is ready once all its
// children have completed. Ready nodes share `threads` BLAS threads: a node
// taken while k others are ready gets max(1, free / (k + 1)), and the sum of
// the shares of running nodes never exceeds `threads`.
class TopScheduler final : public TileHelpers {
public:
  // parent[i]: parent of local node i (-1: root); pending counts derive from it.
  TopScheduler(std::vector<Int> parent, int threads);
  TopScheduler(const TopScheduler &) = delete;
  TopScheduler &operator=(const TopScheduler &) = delete;

  // Worker loop for thread `tid`. run(node, blas_threads, tid) -> bool
  // (false: stop everything). Returns when every node completed or after a stop.
  template <class Run> void work(int tid, Run &&run);

  // TileHelpers: a running front lends `slots` places to idle workers.
  void offer(TileRunner &job, int slots) override;
  void withdraw(TileRunner &job) override;

  // Completion order (topological), and how many nodes were taken from the
  // shared ready list rather than continued by the thread that readied them.
  // Read after every work() returned.
  const std::vector<std::pair<Int, int>> &completed() const { return completed_; }  // (node, tid)
  int queue_pops() const { return queue_pops_; }
  bool stopped() const { return stopped_.load(std::memory_order_acquire); }

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

  bool runnable_hint() const;  // lock-free, for spinning
  bool runnable() const;       // under mtx_
  void publish();              // under mtx_: refresh the lock-free mirrors
  void wake_one_if_runnable(); // under mtx_

  std::vector<Int> parent_;
  std::vector<Int> pending_;   // children not completed yet
  std::vector<Int> ready_;     // shared list (LIFO): initially ready nodes only
  std::vector<std::pair<TileRunner *, int>> offers_;  // (job, slots left), newest last
  std::vector<std::pair<Int, int>> completed_;
  int free_ = 0;               // BLAS threads not held by running nodes
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

template <class Run> void TopScheduler::work(int tid, Run &&run) {
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
  }
}

} // namespace smf
