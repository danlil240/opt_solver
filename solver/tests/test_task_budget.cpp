#include "../benchmarks/common/validation.hpp"
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/info.hpp"
#include "smf/solver.hpp"
#include "smf/threading.hpp"
#include "smf/workspace.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

using namespace smf;

namespace {

// Build a tree with 16 tiny leaf supernodes (width 1, front 2 -> ~4 flops each)
// feeding into a root supernode 16.
void make_tiny_leaves_tree(std::vector<Supernode> &snodes,
                           std::vector<FrontalInfo> &fronts) {
    snodes.resize(17);
    fronts.resize(17);
    for (int i = 0; i < 16; ++i) {
        snodes[i].col_start = i;
        snodes[i].col_end = i + 1;
        snodes[i].parent = 16;
        snodes[16].children.push_back(i);
        fronts[i].row_indices = {i, 16};
    }
    snodes[16].col_start = 16;
    snodes[16].col_end = 17;
    snodes[16].parent = -1;
    fronts[16].row_indices = {16};
}

} // namespace

TEST(TaskBudget, CoarsensTinySubtrees) {
    std::vector<Supernode> snodes;
    std::vector<FrontalInfo> fronts;
    make_tiny_leaves_tree(snodes, fronts);

    ExecutionPolicy policy;
    policy.threads = 4;
    policy.min_task_flops = 10000; // Each leaf has < 10 flops -> must coarsen!
    policy.memory_budget_bytes = 0;

    std::atomic<int> visited{0};
    TaskScheduleStats st = run_costed_postorder(
        snodes, fronts, policy, [&](Int) { visited.fetch_add(1, std::memory_order_relaxed); });

    EXPECT_EQ(visited.load(), 17);
    EXPECT_GT(st.coarsened_subtrees, 0);
    EXPECT_LE(st.tasks_spawned, 2); // Coarsened rather than spawning 17 separate tasks!
}

TEST(TaskBudget, RespectsLiveMemory) {
    // 8 independent child subtrees, each producing a 50x50 contribution block (20,000 bytes each).
    // If all 8 ran simultaneously before parent consumed them, live contribution memory would be 160,000 bytes.
    // Set memory_budget_bytes = 50,000 bytes -> scheduler must throttle so peak_live_contrib_bytes <= 50,000!
    std::vector<Supernode> snodes(9);
    std::vector<FrontalInfo> fronts(9);
    for (int i = 0; i < 8; ++i) {
        snodes[i].col_start = i * 10;
        snodes[i].col_end = (i + 1) * 10;
        snodes[i].parent = 8;
        snodes[8].children.push_back(i);
        // p = 10, f = 60 -> q = 50 extension rows -> 50*50*8 = 20,000 bytes contribution
        fronts[i].row_indices.resize(60, 0);
    }
    snodes[8].col_start = 80;
    snodes[8].col_end = 130;
    snodes[8].parent = -1;
    fronts[8].row_indices.resize(50, 0);

    ExecutionPolicy policy;
    policy.threads = 4;
    policy.min_task_flops = 100;
    policy.memory_budget_bytes = 50000;

    std::atomic<int> visited{0};
    TaskScheduleStats st = run_costed_postorder(
        snodes, fronts, policy, [&](Int) { visited.fetch_add(1, std::memory_order_relaxed); });

    EXPECT_EQ(visited.load(), 9);
    EXPECT_LE(st.peak_live_contrib_bytes, 50000u);
    EXPECT_GT(st.memory_throttled_waits, 0);
}

TEST(TaskBudget, LargeRootMakesProgress) {
    // Even if a single node's contribution or root front exceeds the nominal memory_budget_bytes,
    // the scheduler must still admit it when live memory is 0 so a large root always makes progress!
    std::vector<Supernode> snodes(2);
    std::vector<FrontalInfo> fronts(2);
    snodes[0].col_start = 0;
    snodes[0].col_end = 20;
    snodes[0].parent = 1;
    fronts[0].row_indices.resize(100, 0); // q = 80 -> 80*80*8 = 51,200 bytes

    snodes[1].col_start = 20;
    snodes[1].col_end = 100;
    snodes[1].parent = -1;
    snodes[1].children = {0};
    fronts[1].row_indices.resize(80, 0);

    ExecutionPolicy policy;
    policy.threads = 2;
    policy.min_task_flops = 10;
    policy.memory_budget_bytes = 1024; // smaller than single child contribution

    std::vector<Int> order;
    TaskScheduleStats st = run_costed_postorder(
        snodes, fronts, policy, [&](Int s) { order.push_back(s); });

    ASSERT_EQ(order.size(), 2u);
    EXPECT_EQ(order[0], 0);
    EXPECT_EQ(order[1], 1);
    EXPECT_GE(st.peak_live_contrib_bytes, 51200u);
}

TEST(TaskBudget, ConcurrentInstancesAndDeepTree) {
    // 1. Deep chain of 2500 nodes must not stack-overflow
    {
        const int depth = 2500;
        std::vector<Supernode> snodes(depth);
        std::vector<FrontalInfo> fronts(depth);
        for (int i = 0; i < depth; ++i) {
            snodes[i].col_start = i;
            snodes[i].col_end = i + 1;
            snodes[i].parent = (i + 1 < depth) ? (i + 1) : -1;
            if (i > 0) snodes[i].children = {i - 1};
            fronts[i].row_indices = {i};
        }
        ExecutionPolicy policy{2, 0, 1000, false};
        int count = 0;
        run_costed_postorder(snodes, fronts, policy, [&](Int) { ++count; });
        EXPECT_EQ(count, depth);
    }

    // 2. Concurrent independent Solver instances with multi-threading enabled must not race
    CscLower A;
    A.n = 6;
    A.col_ptr = {0, 2, 4, 6, 8, 10, 11};
    A.row_idx = {0, 5, 1, 5, 2, 5, 3, 5, 4, 5, 5};
    A.values = {4.0, 1.0, -3.0, 0.5, 5.0, -1.0, -2.0, 1.5, 6.0, 0.2, 2.0};

    constexpr int kInstances = 4;
    std::vector<int> ok(kInstances, 1);
    std::vector<std::thread> workers;
    for (int w = 0; w < kInstances; ++w) {
        workers.emplace_back([&, w]() {
            Solver solver;
            Control ctrl;
            ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
            ctrl.nemin = 1;
            ctrl.num_threads = 2;
            ctrl.tree_parallel = true;
            ctrl.memory_budget_bytes = 65536;
            Info info;
            auto ak = solver.analyse(A, ctrl, info);
            if (!ak) { ok[w] = 0; return; }
            NumericWorkspace nws;
            SolveWorkspace sws;
            FactorKeep fk;
            for (int it = 0; it < 10; ++it) {
                if (solver.factor(*ak, A, ctrl, info, fk, nws) != FactorStatus::Success) {
                    ok[w] = 0;
                    return;
                }
                std::vector<double> b = {1.0, -2.0, 3.0, -1.0, 2.0, 4.0};
                std::vector<double> x = b;
                if (solver.solve(fk, ctrl, info, x.data(), 6, 1, sws) != 0) {
                    ok[w] = 0;
                    return;
                }
                if (!smf::bench::is_qualified_validation(smf::bench::validate(A, b, x, 1))) {
                    ok[w] = 0;
                    return;
                }
            }
        });
    }
    for (auto &th : workers) th.join();
    for (int w = 0; w < kInstances; ++w) {
        EXPECT_EQ(ok[w], 1) << "Concurrent instance " << w << " failed";
    }
}
