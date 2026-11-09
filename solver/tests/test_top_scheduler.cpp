// TopScheduler: the top part of the assembly tree as a DAG on a worker pool.
// A node runs once all its children completed; a thread that readies a
// parent continues with it; ready nodes share the BLAS-thread budget.
// PassLanes and ChunkArena: the factorization buffers the phases hand out
// independently of which thread runs what.

#include "parallel/chunk_arena.hpp"
#include "parallel/top_scheduler.hpp"
#include "smf/solve_pool.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <random>
#include <thread>
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
    // Nodes whose run() began after the scheduler recorded the stop. Such a
    // node can only have been taken before, so each other thread has at
    // most one (a node is never handed out once the stop is known). Counting
    // from the failing run's return instead would also count the nodes the
    // others take while the failing thread is preempted before recording it.
    std::atomic<int> late{0};
    SolvePool pool(T);
    pool.run([&](int tid) {
        sched.work(tid, [&](Int v, int, int) {
            if (sched.stopped()) late.fetch_add(1);
            runs[static_cast<std::size_t>(v)].fetch_add(1);
            if (v == fail_at) return false;
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

TEST(TopScheduler, ChainEndFollowsEachChain) {
    // chain_end(tid) once per chain (a node taken from the list, then its
    // continuations), after the chain's last run() on that thread: as many
    // as queue_pops(), with a stop too.
    std::mt19937 rng(7);
    SolvePool pool(T);
    for (int trial = 0; trial < 100; ++trial) {
        const Int n = std::uniform_int_distribution<Int>(1, 2000)(rng);
        std::vector<Int> parent(static_cast<std::size_t>(n), -1);
        for (Int i = 0; i + 1 < n; ++i)
            parent[static_cast<std::size_t>(i)] = std::uniform_int_distribution<Int>(i + 1, n - 1)(rng);
        const Int fail_at = (trial % 4 == 3) ? n / 2 : -1;
        TopScheduler sched(parent, T);
        std::vector<int> in_chain(T, 0);  // per thread, touched by that thread only
        std::atomic<int> ends{0}, bad{0};
        pool.run([&](int tid) {
            sched.work(
                tid,
                [&](Int v, int, int t) {
                    in_chain[static_cast<std::size_t>(t)] = 1;
                    return v != fail_at;
                },
                [&](int t) {
                    if (in_chain[static_cast<std::size_t>(t)] != 1) bad.fetch_add(1);
                    in_chain[static_cast<std::size_t>(t)] = 0;
                    ends.fetch_add(1);
                });
        });
        ASSERT_EQ(bad.load(), 0) << "trial " << trial;
        ASSERT_EQ(ends.load(), sched.queue_pops()) << "trial " << trial;
        for (int t = 0; t < T; ++t) ASSERT_EQ(in_chain[static_cast<std::size_t>(t)], 0) << "trial " << trial;
    }
}

// Control::top_critical_shares (8T engine Part 7): with priorities (the
// remaining path cost to the root), ready nodes are taken highest first, a
// node starts with a share in proportion to its priority among the ready
// ones, and threads freed while nothing is ready go to the running node with
// the highest priority.
TEST(TopScheduler, PriorityPicksHighestReadyFirst) {
    const std::vector<Int> parent = {4, 4, 4, 4, -1};
    TopScheduler sched(parent, 1, {1.0, 5.0, 3.0, 2.0, 9.0});
    SolvePool pool(1);
    pool.run([&](int tid) { sched.work(tid, [&](Int, int, int) { return true; }); });
    const auto &done = sched.completed();
    ASSERT_EQ(done.size(), 5u);
    const Int want[5] = {1, 2, 3, 0, 4};
    for (int k = 0; k < 5; ++k) EXPECT_EQ(done[static_cast<std::size_t>(k)].first, want[k]) << k;
}

TEST(TopScheduler, StartShareFollowsPriority) {
    const std::vector<Int> parent = {3, 3, 3, -1};
    TopScheduler sched(parent, T, {6.0, 1.0, 1.0, 8.0});
    std::atomic<int> first_node{-1}, first_share{0};
    SolvePool pool(1);
    pool.run([&](int tid) {
        sched.work(tid, [&](Int v, int blas_threads, int) {
            int none = -1;
            if (first_node.compare_exchange_strong(none, static_cast<int>(v))) first_share.store(blas_threads);
            return true;
        });
    });
    EXPECT_EQ(first_node.load(), 0);
    EXPECT_EQ(first_share.load(), 6);  // floor(8 * 6 / 8), at most 8 - 2 so the other two get one each
}

TEST(TopScheduler, FreedThreadsGrowTheRunningFront) {
    const std::vector<Int> parent = {2, 2, -1};
    TopScheduler sched(parent, T, {10.0, 1.0, 11.0});
    std::atomic<int> start0{0}, grown0{0};
    SolvePool pool(2);
    pool.run([&](int tid) {
        sched.work(tid, [&](Int v, int blas_threads, int) {
            if (v != 0) return true;
            start0.store(blas_threads);
            const auto t0 = std::chrono::steady_clock::now();
            while (sched.share(0).load() != T && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(2)) {
            }
            grown0.store(sched.share(0).load());
            return true;
        });
    });
    EXPECT_EQ(start0.load(), 7);  // floor(8 * 10 / 11), at most 8 - 1
    EXPECT_EQ(grown0.load(), T);  // node 1's thread came back to it
    expect_topological(sched, parent);
}

TEST(TopScheduler, GrantRacingFinishKeepsBudget) {
    std::mt19937 rng(11);
    SolvePool pool(T);
    for (int trial = 0; trial < 200; ++trial) {
        const Int n = std::uniform_int_distribution<Int>(1, 2000)(rng);
        std::vector<Int> parent(static_cast<std::size_t>(n), -1);
        for (Int i = 0; i + 1 < n; ++i)
            parent[static_cast<std::size_t>(i)] = std::uniform_int_distribution<Int>(i + 1, n - 1)(rng);
        std::vector<double> prio(static_cast<std::size_t>(n));
        for (double &p : prio) p = std::uniform_real_distribution<double>(0.0, 100.0)(rng);
        TopScheduler sched(parent, T, prio);
        std::atomic<int> over{0}, bad_share{0};
        pool.run([&](int tid) {
            sched.work(tid, [&](Int v, int blas_threads, int) {
                if (blas_threads < 1 || blas_threads > T) bad_share.fetch_add(1);
                if (sched.shares_in_use() > T) over.fetch_add(1);
                if (sched.share(v).load() < 1) bad_share.fetch_add(1);
                return true;
            });
        });
        ASSERT_EQ(over.load(), 0) << "trial " << trial;
        ASSERT_EQ(bad_share.load(), 0) << "trial " << trial;
        ASSERT_EQ(sched.shares_in_use(), 0) << "trial " << trial;
        expect_topological(sched, parent);
    }
}

// Overlap with the subtree phase (Control::top_overlap): a node also waits
// for its external completions (subtree tasks), and the BLAS budget starts
// empty and grows as workers join.
TEST(TopScheduler, ExternalCompletionsHoldTheirNode) {
    // Leaves 0 and 1 under root 2; leaf 0 also waits for one external
    // completion. One thread joins at once; the other reports the external
    // completion after a while, then joins.
    const std::vector<Int> parent = {2, 2, -1};
    TopScheduler sched(parent, 2, {}, {1, 0, 0}, 0);
    EXPECT_EQ(sched.shares_in_use(), 0);
    std::atomic<bool> reported{false};
    std::atomic<int> ran0_before{0}, ran1_before{0};
    SolvePool pool(2);
    pool.run([&](int tid) {
        if (tid == 1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            const Int v = 0;
            reported.store(true);
            sched.external_done(&v, 1);
        }
        sched.join();
        sched.work(tid, [&](Int v, int blas_threads, int) {
            EXPECT_GE(blas_threads, 1);
            if (!reported.load()) (v == 0 ? ran0_before : ran1_before).fetch_add(1);
            return true;
        });
    });
    EXPECT_EQ(ran0_before.load(), 0);  // held until reported
    EXPECT_EQ(ran1_before.load(), 1);  // ran on the one joined thread meanwhile
    EXPECT_EQ(sched.shares_in_use(), 0);
    expect_topological(sched, parent);
}

TEST(TopScheduler, JoinedThreadsBoundTheShares) {
    // Random trees whose leaves wait for external completions, reported by
    // the workers in random order before each joins: a node never runs
    // before its completions, and the shares never exceed the joined threads.
    std::mt19937 rng(5);
    SolvePool pool(T);
    for (int trial = 0; trial < 100; ++trial) {
        const Int n = std::uniform_int_distribution<Int>(1, 500)(rng);
        std::vector<Int> parent(static_cast<std::size_t>(n), -1);
        for (Int i = 0; i + 1 < n; ++i)
            parent[static_cast<std::size_t>(i)] = std::uniform_int_distribution<Int>(i + 1, n - 1)(rng);
        std::vector<double> prio;
        if (trial % 2 == 1) {
            prio.resize(static_cast<std::size_t>(n));
            for (double &p : prio) p = std::uniform_real_distribution<double>(0.0, 100.0)(rng);
        }
        std::vector<Int> external(static_cast<std::size_t>(n), 0);
        std::vector<Int> events;  // one entry per external completion
        for (Int i = 0; i < n; ++i) {
            const Int k = std::uniform_int_distribution<Int>(0, 2)(rng);
            external[static_cast<std::size_t>(i)] = k;
            for (Int j = 0; j < k; ++j) events.push_back(i);
        }
        std::shuffle(events.begin(), events.end(), rng);
        std::vector<std::atomic<int>> reported(static_cast<std::size_t>(n));
        TopScheduler sched(parent, T, prio, external, 0);
        std::atomic<std::size_t> next{0};
        std::atomic<int> joined{0}, over{0}, early{0};
        pool.run([&](int tid) {
            for (std::size_t e; (e = next.fetch_add(1)) < events.size();) {
                const Int v = events[e];
                reported[static_cast<std::size_t>(v)].fetch_add(1);
                sched.external_done(&v, 1);
            }
            joined.fetch_add(1);
            sched.join();
            sched.work(tid, [&](Int v, int blas_threads, int) {
                if (reported[static_cast<std::size_t>(v)].load() != external[static_cast<std::size_t>(v)])
                    early.fetch_add(1);
                // In this order: the joined count read after the shares
                // (joins only add to both).
                const int in_use = sched.shares_in_use();
                const int j = joined.load();
                if (in_use > j || blas_threads > j) over.fetch_add(1);
                return true;
            });
        });
        ASSERT_EQ(early.load(), 0) << "trial " << trial;
        ASSERT_EQ(over.load(), 0) << "trial " << trial;
        ASSERT_EQ(sched.shares_in_use(), 0) << "trial " << trial;
        expect_topological(sched, parent);
    }
}

TEST(TopScheduler, HeldNodesWaitForTheLastExternalCompletion) {
    // Chain 0 -> 1 -> 2 (root); node 0 also waits for one external
    // completion, node 2 for another, and node 1 may not start while any
    // external completion is outstanding (a big front: started on the few
    // threads joined so far, it would run far slower than once all have).
    // Node 0 runs once reported; node 1, readied by it, waits (it is not
    // continued) until node 2's completion is reported too.
    const std::vector<Int> parent = {1, 2, -1};
    TopScheduler sched(parent, 2, {}, {1, 0, 1}, 0, {1, 0, 1});
    std::atomic<int> phase{0};  // 1: node 0's completion reported, 2: node 2's
    std::atomic<int> ran1_at{-1};
    SolvePool pool(2);
    pool.run([&](int tid) {
        if (tid == 1) {
            const Int v0 = 0, v2 = 2;
            phase.store(1);
            sched.external_done(&v0, 1);
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            phase.store(2);
            sched.external_done(&v2, 1);
        }
        sched.join();
        sched.work(tid, [&](Int v, int, int) {
            if (v == 1) ran1_at.store(phase.load());
            return true;
        });
    });
    EXPECT_EQ(ran1_at.load(), 2);
    expect_topological(sched, parent);
}

TEST(TopScheduler, StopEndsWorkersWaitingOnExternalCompletions) {
    // Node 0 waits for a completion that never comes (its subtree task
    // failed): stop() ends the joined workers.
    const std::vector<Int> parent = {1, -1};
    TopScheduler sched(parent, T, {}, {1, 0}, 0);
    std::atomic<int> runs{0};
    SolvePool pool(T);
    pool.run([&](int tid) {
        if (tid == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            sched.stop();
        }
        sched.join();
        sched.work(tid, [&](Int, int, int) {
            runs.fetch_add(1);
            return true;
        });
    });
    EXPECT_TRUE(sched.stopped());
    EXPECT_EQ(runs.load(), 0);
    EXPECT_TRUE(sched.completed().empty());
}

namespace {

// Rooms of 8 lanes, as PassLanes::pick reads them.
struct Rooms {
    std::vector<std::size_t> r = std::vector<std::size_t>(T, 10);
    std::size_t operator()(int l) const { return r[static_cast<std::size_t>(l)]; }
};

} // namespace

TEST(PassLanes, ChainStartTakesHomeWhenItHasTheRoom) {
    PassLanes lanes(T);
    Rooms rooms;
    rooms.r[5] = 100;
    EXPECT_EQ(lanes.pick(-1, 2, 10, rooms), 2);
    // Home taken: the next start from home 2 gets the free lane with the
    // most room.
    EXPECT_EQ(lanes.pick(-1, 2, 10, rooms), 5);
}

TEST(PassLanes, GrownLaneServesTheChainWhicheverThreadStartsIt) {
    // The regression: lane 5 grew for the root fronts in an earlier
    // factorization; this time thread 2 starts that chain. It must run on
    // lane 5, not grow lane 2 (a reallocation and fresh pages, tens of ms
    // on TSOPF_FS_b300_c3).
    for (int home = 0; home < T; ++home) {
        PassLanes lanes(T);
        Rooms rooms;
        rooms.r[5] = 100;
        EXPECT_EQ(lanes.pick(-1, home, 50, rooms), 5) << "home " << home;
    }
}

TEST(PassLanes, ChainMovesOnlyToAFreeLaneWithTheRoom) {
    PassLanes lanes(T);
    Rooms rooms;
    rooms.r[3] = 40;
    rooms.r[6] = 100;
    int a = lanes.pick(-1, 1, 5, rooms);
    ASSERT_EQ(a, 1);
    // Room enough: kept, whatever the others have.
    EXPECT_EQ(lanes.pick(a, 1, 10, rooms), 1);
    // Another chain holds lane 6.
    const int b = lanes.pick(-1, 6, 5, rooms);
    ASSERT_EQ(b, 6);
    // Lane 1 lacks the room for 30; lane 3 is the free one with the most.
    a = lanes.pick(a, 1, 30, rooms);
    EXPECT_EQ(a, 3);
    // Lane 1 is free again: a start from home 1 takes it.
    EXPECT_EQ(lanes.pick(-1, 1, 5, rooms), 1);
    // No free lane has room for 60: the chain stays on lane 3 (it grows).
    EXPECT_EQ(lanes.pick(a, 1, 60, rooms), 3);
    // Lane 6 released: now it has the room.
    lanes.release(b);
    EXPECT_EQ(lanes.pick(a, 1, 60, rooms), 6);
}

TEST(PassLanes, NeverTwoHoldersOfALane) {
    // Threads run chains of fronts on lanes; a lane is used (the front's
    // window below) by one thread at a time.
    PassLanes lanes(T);
    std::vector<std::atomic<int>> inside(T);
    std::atomic<int> clash{0};
    SolvePool pool(T);
    pool.run([&](int tid) {
        std::mt19937 rng(static_cast<unsigned>(tid) + 1);
        // Rooms are fixed here (read for free lanes only).
        const auto room = [](int l) { return static_cast<std::size_t>(l) * 10; };
        for (int chain = 0; chain < 2000; ++chain) {
            int lane = -1;
            for (int front = 0; front < 4; ++front) {
                const std::size_t need = std::uniform_int_distribution<std::size_t>(0, 80)(rng);
                lane = lanes.pick(lane, tid, need, room);
                if (inside[static_cast<std::size_t>(lane)].fetch_add(1) != 0) clash.fetch_add(1);
                for (int k = 0; k < 50; ++k) solve_pool_relax();
                inside[static_cast<std::size_t>(lane)].fetch_sub(1);
            }
            lanes.release(lane);
        }
    });
    EXPECT_EQ(clash.load(), 0);
}

TEST(PassLanes, BusyLanesWaitForTheirRelease) {
    // Lanes held by the subtree phase at the start: a chain takes only a
    // released one.
    PassLanes lanes(T, /*all_busy=*/true);
    Rooms rooms;
    lanes.release(5);
    EXPECT_EQ(lanes.pick(-1, 0, 1, rooms), 5);
    lanes.release(2);
    EXPECT_EQ(lanes.pick(-1, 0, 1, rooms), 2);
}

TEST(ChunkArena, NeedsTheTotalWhateverTheOrder) {
    // The subtree roots' blocks of a factorization, taken in a different
    // order every time: the arena needs about their total, whatever the
    // order (a chunk ends unused by less than the largest block), unlike a
    // pass's stack, which needs what its thread happened to take.
    ChunkArena<double> arena(1000);
    std::vector<std::size_t> sizes;
    std::mt19937 rng(3);
    std::size_t total = 0;
    for (int b = 0; b < 300; ++b) {
        sizes.push_back(std::uniform_int_distribution<std::size_t>(1, 120)(rng));
        total += sizes.back();
    }
    const std::size_t bound = total / (1000 - 119) + 1;
    std::size_t first_chunks = 0;
    for (int f = 0; f < 20; ++f) {
        arena.reset();
        std::shuffle(sizes.begin(), sizes.end(), rng);
        std::vector<std::pair<double *, std::size_t>> taken;
        for (const std::size_t n : sizes) {
            double *p = arena.take(n);
            for (std::size_t k = 0; k < n; ++k) p[k] = static_cast<double>(taken.size());
            taken.emplace_back(p, n);
        }
        // Blocks do not overlap: each still holds its own index.
        for (std::size_t i = 0; i < taken.size(); ++i)
            for (std::size_t k = 0; k < taken[i].second; ++k)
                ASSERT_EQ(taken[i].first[k], static_cast<double>(i)) << "factorization " << f;
        if (f == 0) first_chunks = arena.chunks();
        EXPECT_LE(arena.chunks(), bound) << "factorization " << f;
    }
    EXPECT_GE(first_chunks, total / 1000);
}

TEST(ChunkArena, OversizedBlockGetsItsOwnChunkAndThreadsNeverShareABlock) {
    ChunkArena<int> arena(64);
    int *big = arena.take(1000);
    for (int k = 0; k < 1000; ++k) big[k] = k;
    EXPECT_EQ(arena.chunks(), 1u);
    int *small = arena.take(10);
    EXPECT_EQ(arena.chunks(), 2u);
    small[0] = -1;
    EXPECT_EQ(big[999], 999);
    // Threads taking blocks at once get disjoint ones.
    arena.reset();
    SolvePool pool(T);
    std::vector<std::vector<int *>> got(T);
    pool.run([&](int tid) {
        for (int b = 0; b < 500; ++b) {
            int *p = arena.take(7);
            for (int k = 0; k < 7; ++k) p[k] = tid;
            got[static_cast<std::size_t>(tid)].push_back(p);
        }
    });
    for (int t = 0; t < T; ++t)
        for (int *p : got[static_cast<std::size_t>(t)])
            for (int k = 0; k < 7; ++k) ASSERT_EQ(p[k], t);
}

TEST(ChunkArena, OversizedBlocksAnywhereTakeNoNewChunkAfterTheFirstPass) {
    // Regular blocks that fill chunks exactly (no order-dependent waste) and
    // blocks larger than a chunk, at different positions after every reset:
    // the arena's chunks after the first pass serve every later order. An
    // oversized block in the bump chain would skip the chunks left and make
    // the arena grow again whenever it came earlier.
    ChunkArena<double> arena(1000);
    std::vector<std::size_t> sizes(60, 100);
    for (const std::size_t big : {2500, 4000, 1500}) sizes.push_back(big);
    std::mt19937 rng(11);
    std::size_t after_first = 0;
    for (int f = 0; f < 30; ++f) {
        arena.reset();
        std::shuffle(sizes.begin(), sizes.end(), rng);
        std::vector<std::pair<double *, std::size_t>> taken;
        for (const std::size_t n : sizes) {
            double *p = arena.take(n);
            for (std::size_t k = 0; k < n; ++k) p[k] = static_cast<double>(taken.size());
            taken.emplace_back(p, n);
        }
        for (std::size_t i = 0; i < taken.size(); ++i)
            for (std::size_t k = 0; k < taken[i].second; ++k)
                ASSERT_EQ(taken[i].first[k], static_cast<double>(i)) << "factorization " << f;
        if (f == 0) after_first = arena.chunks();
        EXPECT_EQ(arena.chunks(), after_first) << "factorization " << f;
        EXPECT_EQ(arena.growths(), after_first) << "factorization " << f;
    }
    // 6000 regular elements in chunks of 1000, and one chunk per large block.
    EXPECT_EQ(after_first, 6u + 3u);
}

// One oversized block per factorization (a tree's root block, whose size
// follows the delays), larger every time: its own-size chunk is reallocated,
// never joined by another, and each reallocation counts as a growth.
TEST(ChunkArena, GrowingOversizedBlockKeepsOneOwnChunk) {
    ChunkArena<double> arena(1000);
    for (int f = 0; f < 50; ++f) {
        arena.reset();
        for (int b = 0; b < 5; ++b) arena.take(100);
        const std::size_t n = 1500 + 37 * static_cast<std::size_t>(f);
        double *p = arena.take(n);
        for (std::size_t k = 0; k < n; ++k) p[k] = static_cast<double>(k);
        ASSERT_EQ(p[n - 1], static_cast<double>(n - 1));
        EXPECT_EQ(arena.chunks(), 2u) << "factorization " << f;  // one bump, one own-size
        EXPECT_EQ(arena.growths(), 2u + static_cast<std::size_t>(f)) << "factorization " << f;
    }
}

// Three oversized blocks per factorization whose sizes drift upward in a
// random walk, at shuffled positions: the arena holds three own-size chunks
// (the most live at once), however many new highs the sizes set.
TEST(ChunkArena, DriftingOversizedSizesStayBounded) {
    ChunkArena<double> arena(1000);
    std::mt19937 rng(5);
    std::vector<std::size_t> big{1200, 2000, 3000};
    std::size_t highs = 0;
    for (int f = 0; f < 50; ++f) {
        arena.reset();
        for (std::size_t &b : big) {
            const std::size_t up = std::uniform_int_distribution<std::size_t>(0, 300)(rng);
            const std::size_t down = std::uniform_int_distribution<std::size_t>(0, 200)(rng);
            const std::size_t prev_max = *std::max_element(big.begin(), big.end());
            b = std::max<std::size_t>(1001, b + up - std::min(down, b));
            if (b > prev_max) ++highs;
        }
        std::vector<std::size_t> sizes(60, 100);
        sizes.insert(sizes.end(), big.begin(), big.end());
        std::shuffle(sizes.begin(), sizes.end(), rng);
        std::vector<std::pair<double *, std::size_t>> taken;
        for (const std::size_t n : sizes) {
            double *p = arena.take(n);
            for (std::size_t k = 0; k < n; ++k) p[k] = static_cast<double>(taken.size());
            taken.emplace_back(p, n);
        }
        for (std::size_t i = 0; i < taken.size(); ++i)
            for (std::size_t k = 0; k < taken[i].second; ++k)
                ASSERT_EQ(taken[i].first[k], static_cast<double>(i)) << "factorization " << f;
        EXPECT_EQ(arena.chunks(), 6u + 3u) << "factorization " << f;
    }
    EXPECT_GT(highs, 5u) << "premise: the sizes set new highs";
}
