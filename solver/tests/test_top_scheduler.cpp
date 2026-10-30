// TopScheduler: the top part of the assembly tree as a DAG on a worker pool.
// A node runs once all its children completed; a thread that readies a
// parent continues with it; ready nodes share the BLAS-thread budget.

#include "parallel/top_scheduler.hpp"
#include "smf/solve_pool.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <random>
#include <vector>

using namespace smf;

namespace {

constexpr int T = 8;

// Children of each node, from the parent array.
std::vector<std::vector<Int>> children_of(const std::vector<Int> &parent) {
    std::vector<std::vector<Int>> ch(parent.size());
    for (std::size_t i = 0; i < parent.size(); ++i)
        if (parent[i] >= 0) ch[static_cast<std::size_t>(parent[i])].push_back(static_cast<Int>(i));
    return ch;
}

// Every node exactly once in completed(), each after all of its children.
void expect_topological(const TopScheduler &sched, const std::vector<Int> &parent) {
    const auto &done = sched.completed();
    ASSERT_EQ(done.size(), parent.size());
    std::vector<Int> pos(parent.size(), -1);
    for (std::size_t k = 0; k < done.size(); ++k) {
        const Int v = done[k].first;
        ASSERT_GE(v, 0);
        ASSERT_LT(static_cast<std::size_t>(v), parent.size());
        ASSERT_EQ(pos[static_cast<std::size_t>(v)], -1) << "node " << v << " completed twice";
        pos[static_cast<std::size_t>(v)] = static_cast<Int>(k);
        ASSERT_GE(done[k].second, 0);
        ASSERT_LT(done[k].second, T);
    }
    for (std::size_t i = 0; i < parent.size(); ++i)
        if (parent[i] >= 0)
            EXPECT_LT(pos[i], pos[static_cast<std::size_t>(parent[i])]) << "node " << i;
}

} // namespace

TEST(TopScheduler, ChainRunsInOrderWithOnePop) {
    const Int n = 1000;
    std::vector<Int> parent(n);
    for (Int i = 0; i < n; ++i) parent[static_cast<std::size_t>(i)] = i + 1 < n ? i + 1 : -1;
    TopScheduler sched(parent, T);
    std::vector<std::atomic<int>> runs(n);
    std::atomic<int> out_of_order{0};
    std::atomic<Int> last{-1};
    SolvePool pool(T);
    pool.run([&](int tid) {
        sched.work(tid, [&](Int v, int, int) {
            runs[static_cast<std::size_t>(v)].fetch_add(1);
            if (last.exchange(v) != v - 1) out_of_order.fetch_add(1);
            return true;
        });
    });
    for (Int i = 0; i < n; ++i) EXPECT_EQ(runs[static_cast<std::size_t>(i)].load(), 1) << i;
    EXPECT_EQ(out_of_order.load(), 0);
    EXPECT_EQ(sched.queue_pops(), 1);
    EXPECT_FALSE(sched.stopped());
    const auto &done = sched.completed();
    ASSERT_EQ(done.size(), static_cast<std::size_t>(n));
    for (Int i = 0; i < n; ++i) EXPECT_EQ(done[static_cast<std::size_t>(i)].first, i);
}

TEST(TopScheduler, BinaryTreeRespectsDependenciesAndBudget) {
    // Heap numbering h = 1 .. 2047 (children 2h, 2h + 1), node id 2047 - h:
    // the 1024 leaves are 0 .. 1023, the root is 2046.
    const Int leaves = 1024, n = 2 * leaves - 1;
    std::vector<Int> parent(static_cast<std::size_t>(n));
    for (Int h = 1; h <= n; ++h) parent[static_cast<std::size_t>(n - h)] = h == 1 ? -1 : n - h / 2;
    const auto ch = children_of(parent);
    TopScheduler sched(parent, T);
    std::vector<std::atomic<int>> finished(static_cast<std::size_t>(n));
    std::atomic<int> in_use{0}, peak{0}, early{0}, bad_share{0};
    SolvePool pool(T);
    pool.run([&](int tid) {
        sched.work(tid, [&](Int v, int blas_threads, int) {
            if (blas_threads < 1 || blas_threads > T) bad_share.fetch_add(1);
            const int now = in_use.fetch_add(blas_threads) + blas_threads;
            int p = peak.load();
            while (now > p && !peak.compare_exchange_weak(p, now)) {
            }
            for (const Int c : ch[static_cast<std::size_t>(v)])
                if (finished[static_cast<std::size_t>(c)].load() == 0) early.fetch_add(1);
            finished[static_cast<std::size_t>(v)].store(1);
            in_use.fetch_sub(blas_threads);
            return true;
        });
    });
    EXPECT_LE(peak.load(), T);
    EXPECT_EQ(early.load(), 0);
    EXPECT_EQ(bad_share.load(), 0);
    EXPECT_LE(sched.queue_pops(), leaves);
    expect_topological(sched, parent);
}

TEST(TopScheduler, LoneRootGetsAllThreads) {
    const Int leaves = 64;
    std::vector<Int> parent(static_cast<std::size_t>(leaves) + 1, leaves);
    parent[static_cast<std::size_t>(leaves)] = -1;
    TopScheduler sched(parent, T);
    std::atomic<int> root_threads{0};
    SolvePool pool(T);
    pool.run([&](int tid) {
        sched.work(tid, [&](Int v, int blas_threads, int) {
            if (v == leaves) root_threads.store(blas_threads);
            return true;
        });
    });
    EXPECT_EQ(root_threads.load(), T);
    expect_topological(sched, parent);
}

TEST(TopScheduler, StopEndsAllWorkers) {
    // 500 independent leaves (0 .. 499) plus a 2000-node chain (500 ..
    // 2499) whose 700th node fails. The chain's head has the highest id of
    // the initially ready nodes, so it is taken first (the shared list is
    // LIFO); the leaves are slow enough (20 us each) that most are still
    // listed when the chain fails.
    const Int extra = 500, chain = 2000, n = extra + chain, fail_at = extra + 700;
    std::vector<Int> parent(static_cast<std::size_t>(n), -1);
    for (Int i = extra; i + 1 < n; ++i) parent[static_cast<std::size_t>(i)] = i + 1;
    TopScheduler sched(parent, T);
    std::vector<std::atomic<int>> runs(static_cast<std::size_t>(n));
    std::atomic<bool> stop_seen{false};
    // Nodes whose run() began after the failing one returned false. Such a
    // node can only have been taken before the scheduler saw the stop, so
    // each other thread has at most one (a node is never handed out once
    // the stop is known; the scheduler records it as soon as run() returns).
    std::atomic<int> late{0};
    SolvePool pool(T);
    pool.run([&](int tid) {
        sched.work(tid, [&](Int v, int, int) {
            if (stop_seen.load()) late.fetch_add(1);
            runs[static_cast<std::size_t>(v)].fetch_add(1);
            if (v == fail_at) {
                stop_seen.store(true);
                return false;
            }
            if (v < extra) {
                const auto t0 = std::chrono::steady_clock::now();
                while (std::chrono::steady_clock::now() - t0 < std::chrono::microseconds(20)) {
                }
            }
            return true;
        });
    });
    // Every work() returned (we got here) and the stop is recorded.
    EXPECT_TRUE(sched.stopped());
    EXPECT_EQ(runs[static_cast<std::size_t>(fail_at)].load(), 1);
    for (Int i = 0; i < n; ++i) EXPECT_LE(runs[static_cast<std::size_t>(i)].load(), 1) << i;
    // The chain stops at the failed node: its parent never becomes ready.
    for (Int i = fail_at + 1; i < n; ++i) EXPECT_EQ(runs[static_cast<std::size_t>(i)].load(), 0) << i;
    EXPECT_LE(late.load(), T - 1);
    // completed() holds the nodes that succeeded, never the failed one.
    for (const auto &[v, tid] : sched.completed()) {
        EXPECT_NE(v, fail_at);
        EXPECT_EQ(runs[static_cast<std::size_t>(v)].load(), 1);
    }
}

TEST(TopScheduler, RandomTreesStress) {
    std::mt19937 rng(1);
    SolvePool pool(T);
    for (int trial = 0; trial < 300; ++trial) {
        const Int n = std::uniform_int_distribution<Int>(1, 3000)(rng);
        std::vector<Int> parent(static_cast<std::size_t>(n), -1);
        for (Int i = 0; i + 1 < n; ++i)
            parent[static_cast<std::size_t>(i)] = std::uniform_int_distribution<Int>(i + 1, n - 1)(rng);
        TopScheduler sched(parent, T);
        std::vector<std::atomic<int>> runs(static_cast<std::size_t>(n));
        pool.run([&](int tid) {
            sched.work(tid, [&](Int v, int, int) {
                runs[static_cast<std::size_t>(v)].fetch_add(1);
                return true;
            });
        });
        for (Int i = 0; i < n; ++i)
            ASSERT_EQ(runs[static_cast<std::size_t>(i)].load(), 1) << "trial " << trial << " node " << i;
        ASSERT_FALSE(sched.stopped());
        expect_topological(sched, parent);
    }
}
