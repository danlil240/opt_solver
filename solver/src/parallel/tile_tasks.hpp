#pragma once
// 2D tile tasks for the fronts at the top of the assembly tree.
//
// A large front is factored as a DAG of block tasks on the APTP block-column
// structure (aptp_internal.hpp), with block width nb = AptpParams::nb over
// the eligible columns and row blocks of nb rows (eligible rows on the same
// boundaries, then the non-eligible rows):
//
//   Factor(k)     back up block column k, factor its diagonal tile, apply the
//                 tile permutation, form L_t and D^{-1};
//   ApplyN(i, k)  L of row block i of block column k, with its a posteriori
//                 check (first failed pivot of those rows);
//   Finish(k)     nelim = min over ApplyN(., k); ApplyT, restore of the failed
//                 columns, (L D)^T of the eligible columns right of them, the
//                 update of the block column's own failed columns and of the
//                 earlier failed columns (UpdateNT/TN);
//   Update(i,j,k) tile (i, j), j > k, -= L(i, k) (L D)(j, k)^T.
//
// Dependencies: ApplyN(., k) and Finish(k) follow Factor(k); Finish(k)
// follows every ApplyN(., k); Update(i, j, k) follows Finish(k) and
// Update(i, j, k - 1); Factor(k + 1) follows Finish(k) and the updates
// Update(., k + 1, k) of its own block column. Updates of later block
// columns therefore overlap Factor / ApplyN / Finish of the next one
// (lookahead). Updates into a tile are applied in block-column order k, so
// the result does not depend on the schedule nor on the thread count (it
// differs from the serial kernel's only by the rounding of the split GEMMs).
// After the DAG, the failed columns are moved to the back (serial) and the
// non-eligible trailing triangle is updated as a second DAG: FormW(j) per
// block column of it, then its tiles.
//
// Every task calls BLAS single-threaded. The calling thread runs tasks, and
// with nthreads > 1 so do nthreads - 1 helpers: idle threads of the caller's
// pool recruited through TileHelpers, or (helpers == nullptr) threads of its
// own.

#include "smf/ldlt_aptp.hpp"
#include "smf/solve_pool.hpp"
#include "smf/types.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <type_traits>
#include <vector>

namespace smf {

class TileRunner;

/// Cooperative job an idle TopScheduler worker can help execute.
class HelperJob {
public:
    virtual void add_helper() = 0;
    virtual void help() = 0;

protected:
    ~HelperJob() = default;
};

/// Task graph run by a TileRunner. execute() runs a task; release() calls
/// runner.release(t) for each successor t of a finished task (under the
/// runner's lock); urgent() tasks (the critical path) are taken first.
class TileDag {
public:
    virtual void execute(int task) = 0;
    virtual void release(int task, TileRunner &runner) = 0;
    virtual bool urgent(int task) const = 0;

protected:
    ~TileDag() = default;
};

/// A pool that lends idle threads to a tile or strip job: offer() makes `slots`
/// places available (an idle thread takes one with HelperJob::add_helper()
/// under the pool's lock, then calls HelperJob::help()); withdraw() removes
/// whatever is left, after which no thread joins.
class TileHelpers {
public:
    virtual void offer(TileRunner &job, int slots) = 0;
    virtual void withdraw(TileRunner &job) = 0;
    virtual void offer_helper(HelperJob & /*job*/, int /*slots*/) {}
    virtual void withdraw_helper(HelperJob & /*job*/) {}

protected:
    ~TileHelpers() = default;
};

/// Thread-local active TopScheduler helpers pointer (non-null while a
/// SolvePool worker is executing a top front inside TopScheduler::work).
TileHelpers *current_top_helpers() noexcept;

class TopHelpersScope {
public:
    explicit TopHelpersScope(TileHelpers *h) noexcept;
    ~TopHelpersScope() noexcept;
    TopHelpersScope(const TopHelpersScope &) = delete;
    TopHelpersScope &operator=(const TopHelpersScope &) = delete;

private:
    TileHelpers *prev_;
};

/// Lightweight lock-free stack-allocated parallel-for job executed cooperatively
/// on the calling thread plus up to (nthreads - 1) idle SolvePool workers via
/// TopScheduler::offer_helper(). Never spawns OS/OpenMP threads and never
/// deadlocks even if all other pool workers are busy on sibling fronts.
class StripJob final : public HelperJob {
public:
    StripJob(int ntasks, void (*fn)(int, const void *), const void *ctx) noexcept
        : ntasks_(ntasks), fn_(fn), ctx_(ctx) {}

    void add_helper() override { helpers_.fetch_add(1, std::memory_order_acq_rel); }

    void help() override {
        drain();
        helpers_.fetch_sub(1, std::memory_order_release);
    }

    void drain() noexcept {
        for (;;) {
            const int t = next_.fetch_add(1, std::memory_order_relaxed);
            if (t >= ntasks_) break;
            fn_(t, ctx_);
        }
    }

    void wait_helpers() const noexcept {
        while (helpers_.load(std::memory_order_acquire) > 0) solve_pool_relax();
    }

private:
    std::atomic<int> next_{0};
    std::atomic<int> helpers_{0};
    int ntasks_;
    void (*fn_)(int, const void *);
    const void *ctx_;
};

template <class Fn>
inline void parallel_for_helpers(int ntasks, int nthreads, TileHelpers *helpers, const Fn &fn) {
    if (ntasks <= 0) return;
    if (ntasks == 1 || nthreads <= 1 || helpers == nullptr) {
        for (int t = 0; t < ntasks; ++t) fn(t);
        return;
    }
    auto thunk = [](int t, const void *p) { (*static_cast<const Fn *>(p))(t); };
    StripJob job(ntasks, thunk, &fn);
    const int slots = std::min(nthreads, ntasks) - 1;
    if (slots > 0) helpers->offer_helper(job, slots);
    job.drain();
    if (slots > 0) {
        helpers->withdraw_helper(job);
        job.wait_helpers();
    }
}

/// Reused scratch of the tile engine (high-water mark sizes).
struct TileWorkspace {
    std::vector<int> ints;
    int64_t growths = 0; ///< capacity growths of `ints` (allocation count)
};

/// Executes one TileDag at a time on the calling thread plus helpers.
class TileRunner final : public HelperJob {
public:
    /// Runs `dag` to completion. pending[t]: number of unfinished
    /// predecessors of each task (the dag's own numbering); roots: the tasks
    /// without any; total: the number of tasks. ready_hi / ready_lo: room
    /// for `total` task ids each.
    void run(TileDag &dag, int *pending, const int *roots, int nroots, int total, int *ready_hi,
             int *ready_lo, int nthreads, TileHelpers *helpers);
    /// Called by the dag's release(): one predecessor of `task` finished.
    void release(int task);
    /// A recruited thread: take a slot (under the pool's lock), then help().
    void add_helper() override;
    void help() override;

private:
    void work();

    std::mutex mtx_;
    std::condition_variable cv_;
    TileDag *dag_ = nullptr;
    int *pending_ = nullptr;
    int *hi_ = nullptr; // LIFO
    int *lo_ = nullptr; // FIFO
    int nhi_ = 0, lo_head_ = 0, lo_tail_ = 0;
    int done_ = 0, total_ = 0;
    int pushed_ = 0;   // tasks made ready by the current release()
    int helpers_ = 0;  // recruited threads not yet out of help()
};

/// ldlt_aptp_front (same contract and output layout) as 2D tile tasks on
/// `nthreads` threads. `work` as for ldlt_aptp_front (aptp_work_size).
/// Bitwise identical for every nthreads >= 1.
AptpResult ldlt_aptp_front_tiled(double *F, int f, int num_elig, Int *rows, const AptpParams &prm,
                                 int8_t *tags, double *work, TileWorkspace &tws, int nthreads,
                                 TileHelpers *helpers);

} // namespace smf
