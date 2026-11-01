// Thread-count model of the subtree-parallel factorization: the tree
// partition (largest-first, on arrays) and plan_threads' cost per candidate
// thread count, on synthetic trees and one analysed matrix.

#include "parallel/thread_plan.hpp"
#include "smf/analysis.hpp"
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/info.hpp"
#include "smf/solver.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace smf;

namespace {

// A tree built top-down (a node's parent is added before it), with its
// postorder from an iterative DFS (chains are 10k deep).
struct Synth {
    std::vector<Int> parent;
    std::vector<double> work;
    std::vector<std::uint8_t> split;
    std::vector<Int> post;

    Int add(Int par, double w, bool tiles = false) {
        parent.push_back(par);
        work.push_back(w);
        split.push_back(tiles ? 1 : 0);
        return static_cast<Int>(parent.size() - 1);
    }

    TreeView view() {
        const std::size_t n = parent.size();
        std::vector<std::vector<Int>> ch(n);
        std::vector<Int> roots;
        for (std::size_t i = 0; i < n; ++i) {
            if (parent[i] < 0) roots.push_back(static_cast<Int>(i));
            else ch[static_cast<std::size_t>(parent[i])].push_back(static_cast<Int>(i));
        }
        post.clear();
        std::vector<std::pair<Int, std::size_t>> stack;
        for (const Int r : roots) {
            stack.emplace_back(r, 0);
            while (!stack.empty()) {
                auto &[s, k] = stack.back();
                const auto &c = ch[static_cast<std::size_t>(s)];
                if (k < c.size()) {
                    const Int next = c[k++];
                    stack.emplace_back(next, 0);
                } else {
                    post.push_back(s);
                    stack.pop_back();
                }
            }
        }
        return TreeView{post, parent, work, split};
    }
};

// Complete binary tree of the given depth below `par`.
void binary(Synth &t, Int par, int depth, double leaf, double internal, bool split = false) {
    const Int s = t.add(par, depth == 0 ? leaf : internal, split);
    if (depth == 0) return;
    binary(t, s, depth - 1, leaf, internal);
    binary(t, s, depth - 1, leaf, internal);
}

// 5-point Laplacian on an m x m grid (SPD), lower CSC.
CscLower laplacian_2d(int m) {
    CscLower A;
    A.n = m * m;
    A.col_ptr.push_back(0);
    for (int j = 0; j < A.n; ++j) {
        A.row_idx.push_back(j);
        A.values.push_back(4.0);
        if ((j % m) + 1 < m) {
            A.row_idx.push_back(j + 1);
            A.values.push_back(-1.0);
        }
        if (j + m < A.n) {
            A.row_idx.push_back(j + m);
            A.values.push_back(-1.0);
        }
        A.col_ptr.push_back(static_cast<Int>(A.row_idx.size()));
    }
    return A;
}

} // namespace

// A chain never splits into two subtrees: no candidate beyond 1.
TEST(ThreadPlan, ChainIsSerial) {
    Synth chain;
    Int s = chain.add(-1, 1e6);
    for (int i = 1; i < 1000; ++i) s = chain.add(s, 1e6);
    const TreeView t = chain.view();
    const ThreadPlan p = plan_threads(t, 16, {});
    EXPECT_EQ(p.threads, 1);
    EXPECT_DOUBLE_EQ(p.predicted_speedup, 1.0);
}

// 4096 heavy leaves under light internal nodes: 64 subtrees, 63 top nodes
// at 16 threads, cost ~ total / 16.
TEST(ThreadPlan, BalancedTreeUsesAllThreads) {
    Synth bal;
    binary(bal, -1, 12, 1e7, 1e5);
    const TreeView t = bal.view();
    const auto p = plan_threads(t, 16, {});
    EXPECT_EQ(p.threads, 16);
    EXPECT_GT(p.predicted_speedup, 8.0);
}

// A root with 98% of the work: unsplit, the top phase alone takes 0.98 of
// the serial time, so no thread count gains more than 2% (the tie rule
// keeps the smallest); split into tiles it scales (to 16 threads with the
// split cap lifted to 16; SplitCapStopsAtCap has the default cap of 8).
TEST(ThreadPlan, SerialRootCapsThreads) {
    for (const bool root_split : {false, true}) {
        Synth rooty;
        const Int root = rooty.add(-1, 5e10, root_split);
        binary(rooty, root, 9, 1e6, 1e3);
        binary(rooty, root, 9, 1e6, 1e3);
        const TreeView t = rooty.view();
        ThreadCosts costs;
        costs.split_cap = 16;
        const ThreadPlan p = plan_threads(t, 16, costs);
        if (root_split) {
            EXPECT_EQ(p.threads, 16);
        } else {
            EXPECT_LE(p.threads, 2);
            EXPECT_LT(p.predicted_speedup, 1.03);
        }
    }
}

// The two phases: an unsplit root with 60% of the work caps the speedup at
// 1/0.6, but the 1024 balanced leaves below it still scale (phase 1 is
// sub_total / T, not the heaviest subtree), so all 16 threads pay off.
TEST(ThreadPlan, SubtreePhaseScalesUnderSerialRoot) {
    Synth rooty;
    const Int root = rooty.add(-1, 1.5e9);
    binary(rooty, root, 9, 1e6, 1e3);
    binary(rooty, root, 9, 1e6, 1e3);
    const TreeView t = rooty.view();
    const ThreadPlan p = plan_threads(t, 16, {});
    EXPECT_EQ(p.threads, 16);
    EXPECT_GT(p.predicted_speedup, 1.5);
    EXPECT_LT(p.predicted_speedup, 1.0 / 0.59);
}

// robot_arm-like: a 10000-node spine of light fronts, each with one leaf.
// Every partition puts most of the spine in the top part.
TEST(ThreadPlan, ManyTinyTopFrontsAreSerial) {
    Synth cat;
    Int s = -1;
    for (int i = 0; i < 10000; ++i) {
        s = cat.add(s, 2e3);
        cat.add(s, 1e3);
    }
    const TreeView t = cat.view();
    EXPECT_EQ(plan_threads(t, 16, {}).threads, 1);
}

// max_threads joins the powers of two as a candidate, and caps the choice.
TEST(ThreadPlan, CandidatesStopAtMaxThreads) {
    Synth bal;
    binary(bal, -1, 12, 1e7, 1e5);
    const TreeView t = bal.view();
    EXPECT_EQ(plan_threads(t, 12, {}).threads, 12);
    EXPECT_EQ(plan_threads(t, 3, {}).threads, 3);
    EXPECT_EQ(plan_threads(t, 1, {}).threads, 1);
}

// The partition of an analysed tree: disjoint whole subtrees and a top part
// that cover every node once, each subtree within total / (4 T) unless a
// single front.
TEST(ThreadPlan, PartitionCoversTree) {
    const CscLower A = laplacian_2d(120);
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
    Solver solver;
    Info info;
    auto ak = solver.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);
    const AnalysisKeep &keep = *ak;
    const std::size_t ns = keep.supernodes.size();

    std::vector<Int> parent;
    std::vector<double> work;
    std::vector<std::uint8_t> split;
    const TreeView t = tree_view(keep, /*tiles_on=*/true, parent, work, split);
    ASSERT_EQ(t.postorder.size(), ns);
    ASSERT_EQ(t.parent.size(), ns);
    double total = 0.0;
    for (std::size_t s = 0; s < ns; ++s) {
        const double f = static_cast<double>(keep.front_size(static_cast<Int>(s)));
        EXPECT_EQ(t.parent[s], keep.supernodes[s].parent);
        EXPECT_DOUBLE_EQ(t.work[s], f * f * static_cast<double>(keep.supernodes[s].width()));
        EXPECT_EQ(t.split[s], keep.front_size(static_cast<Int>(s)) >= 512 ? 1 : 0);
        total += t.work[s];
    }
    ASSERT_GT(total, kTreeParallelMinWork) << "premise: heavy enough to partition";

    std::vector<std::uint8_t> top;
    const auto subtrees = partition_subtrees(t, 4, kTreeParallelMinWork, top);
    ASSERT_GE(subtrees.size(), 2u);
    ASSERT_EQ(top.size(), ns);
    std::vector<int> covered(ns, 0);
    for (const auto &[lo, hi] : subtrees) {
        ASSERT_LE(lo, hi);
        ASSERT_LT(hi, ns);
        double w = 0.0;
        for (std::size_t k = lo; k <= hi; ++k) {
            const std::size_t s = static_cast<std::size_t>(t.postorder[k]);
            ++covered[s];
            w += t.work[s];
        }
        // A whole subtree: its root (the last in postorder) hangs off the
        // top part or is a root of the forest.
        const Int r = t.postorder[hi];
        const Int pr = t.parent[static_cast<std::size_t>(r)];
        EXPECT_TRUE(pr < 0 || top[static_cast<std::size_t>(pr)] == 1);
        if (lo != hi) EXPECT_LE(w, total / 16.0);
    }
    for (std::size_t s = 0; s < ns; ++s)
        EXPECT_EQ(covered[s] + top[s], 1) << "supernode " << s;
}

namespace {

// Checks a partition of t: [lo, hi] postorder ranges that are disjoint,
// each closed under children (a node's parent is in the same range, in the
// top part, or none), covering with the top part every node exactly once.
// Returns each range's work.
std::vector<double> check_ranges(const TreeView &t,
                                 const std::vector<std::pair<std::size_t, std::size_t>> &ranges,
                                 const std::vector<std::uint8_t> &top) {
    const std::size_t ns = t.parent.size();
    std::vector<std::size_t> pos(ns);
    for (std::size_t k = 0; k < t.postorder.size(); ++k)
        pos[static_cast<std::size_t>(t.postorder[k])] = k;
    std::vector<int> covered(ns, 0);
    std::vector<double> works;
    for (const auto &[lo, hi] : ranges) {
        EXPECT_LE(lo, hi);
        EXPECT_LT(hi, ns);
        double w = 0.0;
        for (std::size_t k = lo; k <= hi && k < ns; ++k) {
            const std::size_t s = static_cast<std::size_t>(t.postorder[k]);
            ++covered[s];
            w += t.work[s];
            const Int p = t.parent[s];
            if (p < 0) continue;
            const std::size_t pk = pos[static_cast<std::size_t>(p)];
            EXPECT_TRUE((pk >= lo && pk <= hi) || top[static_cast<std::size_t>(p)] == 1)
                << "node " << s << " escapes its range";
        }
        works.push_back(w);
    }
    for (std::size_t s = 0; s < ns; ++s)
        EXPECT_EQ(covered[s] + top[s], 1) << "supernode " << s;
    return works;
}

} // namespace

// A comb: a 20-front chain of heavy fronts (the top part), each with 500
// tiny leaves, the bottom one also with 4000 tiny leaves and one heavier
// leaf. partition_subtrees keeps one range per leaf (the delay budget's
// regions); partition_tasks merges each run of adjacent leaves into
// grain-sized tasks, never across a chain front, and leaves the heavy leaf
// alone.
TEST(ThreadPlan, TasksMergeTinySubtrees) {
    Synth comb;
    std::vector<Int> chain;
    Int c = -1;
    // The bottom front holds most of the work, so that every chain front
    // is heavier than total / (4 T) at every T.
    for (int i = 0; i < 20; ++i) chain.push_back(c = comb.add(c, i == 19 ? 1e8 : 2e6));
    std::size_t leaves = 0;
    for (const Int ci : chain) {
        for (int j = 0; j < 500; ++j, ++leaves) comb.add(ci, 10.0 + (j % 7));
    }
    for (int j = 0; j < 4000; ++j, ++leaves) comb.add(chain.back(), 5.0);
    const Int heavy = comb.add(chain.back(), 2e5);
    ++leaves;
    const TreeView t = comb.view();
    double total = 0.0;
    for (const double w : comb.work) total += w;
    ASSERT_GT(total, kTreeParallelMinWork) << "premise: heavy enough to partition";

    for (const int T : {2, 4, 8, 16}) {
        std::vector<std::uint8_t> btop, ttop;
        const auto sub = partition_subtrees(t, T, kTreeParallelMinWork, btop);
        const auto tasks = partition_tasks(t, T, kTreeParallelMinWork, ttop);
        EXPECT_EQ(btop, ttop) << "T = " << T;
        for (const Int ci : chain) EXPECT_EQ(ttop[static_cast<std::size_t>(ci)], 1);

        // The budget partition: one single-leaf range per leaf, heaviest
        // first (ties: the larger node first).
        ASSERT_EQ(sub.size(), leaves) << "T = " << T;
        const std::vector<double> sw = check_ranges(t, sub, btop);
        for (std::size_t i = 0; i < sub.size(); ++i) {
            EXPECT_EQ(sub[i].first, sub[i].second);
            if (i > 0) {
                const Int a = t.postorder[sub[i - 1].first], b = t.postorder[sub[i].first];
                EXPECT_TRUE(sw[i - 1] > sw[i] || (sw[i - 1] == sw[i] && a > b)) << i;
            }
        }

        // The tasks: a valid partition with the same top part, far fewer
        // ranges, each within the grain unless a single subtree (the heavy
        // leaf), heaviest first.
        const std::vector<double> tw = check_ranges(t, tasks, ttop);
        EXPECT_LT(tasks.size(), leaves / 20) << "T = " << T;
        EXPECT_GE(tasks.size(), chain.size()) << "runs never cross a chain front";
        double sub_total = 0.0;
        for (const double w : sw) sub_total += w;
        const double grain = task_grain(sub_total, T);
        EXPECT_LE(tasks.size(), static_cast<std::size_t>(2.0 * kTaskGrainsPerThread * T) +
                                    chain.size() + 1);
        bool heavy_alone = false;
        for (std::size_t i = 0; i < tasks.size(); ++i) {
            if (tasks[i].first == tasks[i].second &&
                t.postorder[tasks[i].first] == heavy)
                heavy_alone = true;
            else
                EXPECT_LE(tw[i], grain * (1.0 + 1e-12)) << "task " << i;
            if (i > 0) EXPECT_GE(tw[i - 1], tw[i]);
        }
        EXPECT_TRUE(heavy_alone) << "T = " << T;

        // The plan's terms describe the tasks.
        const PartitionCount pc = partition_count(t, T, kTreeParallelMinWork);
        EXPECT_EQ(pc.subtrees, tasks.size());
        EXPECT_EQ(pc.top, chain.size());
        for (const auto &k : factor_plan_terms(t, T)) {
            if (k.threads != T) continue;
            EXPECT_DOUBLE_EQ(k.max_subtree, *std::max_element(tw.begin(), tw.end()));
            EXPECT_NEAR(k.sub_total, sub_total, 1e-9 * sub_total);
        }
    }
}

// plan_threads counts each candidate's partition without the heap; the
// counts must be partition_tasks' sizes, and its top part
// partition_subtrees' (the tasks merge whole subtrees of that split).
TEST(ThreadPlan, PartitionCountMatchesPartition) {
    Synth bal;
    binary(bal, -1, 12, 1e7, 1e5);
    Synth cat;
    Int s = -1;
    for (int i = 0; i < 10000; ++i) {
        s = cat.add(s, 2e3);
        cat.add(s, 1e3);
    }
    Synth rooty;
    const Int root = rooty.add(-1, 1.5e9);
    binary(rooty, root, 9, 1e6, 1e3);
    binary(rooty, root, 7, 3e6, 0.0);
    rooty.add(-1, 4e8);  // a second root: a forest

    const CscLower A = laplacian_2d(120);
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
    Solver solver;
    Info info;
    auto ak = solver.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);
    std::vector<Int> parent;
    std::vector<double> work;
    std::vector<std::uint8_t> split;

    const TreeView views[] = {bal.view(), cat.view(), rooty.view(),
                              tree_view(*ak, false, parent, work, split)};
    for (std::size_t v = 0; v < std::size(views); ++v) {
        for (const int T : {1, 2, 3, 4, 8, 12, 16, 64}) {
            std::vector<std::uint8_t> top, ttop;
            const auto sub = partition_subtrees(views[v], T, kTreeParallelMinWork, top);
            const auto tasks = partition_tasks(views[v], T, kTreeParallelMinWork, ttop);
            const PartitionCount pc = partition_count(views[v], T, kTreeParallelMinWork);
            std::size_t ntop = 0;
            for (const auto x : top) ntop += x;
            EXPECT_EQ(pc.subtrees, tasks.size()) << "tree " << v << ", T = " << T;
            EXPECT_LE(tasks.size(), sub.size()) << "tree " << v << ", T = " << T;
            EXPECT_EQ(pc.top, ntop) << "tree " << v << ", T = " << T;
            EXPECT_EQ(ttop, top) << "tree " << v << ", T = " << T;
            if (!sub.empty()) check_ranges(views[v], tasks, ttop);
        }
    }
}

// One SubtreeSums refilled for tree after tree (largest first, so later
// fills reuse longer buffers), as the engine keeps it per workspace: the
// partitions and the plan terms are those of a fresh one, bitwise.
TEST(ThreadPlan, ReusedSubtreeSumsMatchFresh) {
    Synth bal;
    binary(bal, -1, 12, 1e7, 1e5);
    Synth cat;
    Int s = -1;
    for (int i = 0; i < 10000; ++i) {
        s = cat.add(s, 2e3);
        cat.add(s, 1e3);
    }
    Synth rooty;
    const Int root = rooty.add(-1, 1.5e9, true);
    binary(rooty, root, 9, 1e6, 1e3);
    binary(rooty, root, 7, 3e6, 0.0);
    rooty.add(-1, 4e8);
    Synth light;
    binary(light, -1, 3, 1e5, 1e3);

    const TreeView views[] = {cat.view(), bal.view(), rooty.view(), light.view(), bal.view()};
    SubtreeSums kept;
    for (std::size_t v = 0; v < std::size(views); ++v) {
        subtree_sums(views[v], kept);
        for (const int T : {2, 4, 8, 16}) {
            std::vector<std::uint8_t> top, ktop;
            EXPECT_EQ(partition_tasks(views[v], kept, T, kTreeParallelMinWork, ktop),
                      partition_tasks(views[v], T, kTreeParallelMinWork, top))
                << "tree " << v << ", T = " << T;
            EXPECT_EQ(ktop, top) << "tree " << v << ", T = " << T;
            EXPECT_EQ(partition_subtrees(views[v], kept, T, kTreeParallelMinWork, ktop),
                      partition_subtrees(views[v], T, kTreeParallelMinWork, top))
                << "tree " << v << ", T = " << T;
            EXPECT_EQ(ktop, top) << "tree " << v << ", T = " << T;
        }
        const auto fresh = factor_plan_terms(views[v], 16);
        const auto reused = factor_plan_terms(views[v], kept, 16);
        ASSERT_EQ(reused.size(), fresh.size()) << "tree " << v;
        for (std::size_t i = 0; i < fresh.size(); ++i) {
            EXPECT_EQ(reused[i].ok, fresh[i].ok);
            EXPECT_EQ(reused[i].sub_total, fresh[i].sub_total);
            EXPECT_EQ(reused[i].max_subtree, fresh[i].max_subtree);
            EXPECT_EQ(reused[i].top_total, fresh[i].top_total);
            EXPECT_EQ(reused[i].cp_top, fresh[i].cp_top);
            EXPECT_EQ(reused[i].top_fronts, fresh[i].top_fronts);
            EXPECT_EQ(reused[i].split_top_fronts, fresh[i].split_top_fronts);
            EXPECT_EQ(reused[i].split_top_work, fresh[i].split_top_work);
            EXPECT_EQ(reused[i].total, fresh[i].total);
        }
        EXPECT_EQ(plan_threads(views[v], kept, 16, {}).threads,
                  plan_threads(views[v], 16, {}).threads)
            << "tree " << v;
    }
}

// The solve's model: two light subtrees under a 5000-step chain of top
// steps. Every parallel candidate pays the chain serially plus a hand-off
// per top step (5000 x 2e3 against 2.7e5 of work), so the solve is serial.
TEST(ThreadPlan, SolveChainTopIsSerial) {
    const std::vector<double> sub{1e4, 1e4};
    const std::vector<double> top(5000, 50.0);
    std::vector<Int> chain_parent(5000);
    for (std::size_t i = 0; i < chain_parent.size(); ++i) chain_parent[i] = static_cast<Int>(i) + 1;
    chain_parent.back() = -1;
    EXPECT_EQ(plan_solve_threads(sub, top, chain_parent, 4000, 16, {}), 1);
}

// 64 heavy subtrees under 3 light top steps: the subtree phase scales.
TEST(ThreadPlan, SolveBalancedSubtreesParallel) {
    const std::vector<double> sub64(64, 1e6);
    const std::vector<double> top3{1e3, 1e3, 1e3};
    const std::vector<Int> parent3{2, 2, -1};
    EXPECT_GE(plan_solve_threads(sub64, top3, parent3, 300, 16, {}), 8);
    // max_threads caps it, and fewer than 2 subtrees or threads is serial.
    EXPECT_EQ(plan_solve_threads(sub64, top3, parent3, 300, 4, {}), 4);
    EXPECT_EQ(plan_solve_threads(sub64, top3, parent3, 300, 1, {}), 1);
    EXPECT_EQ(plan_solve_threads(std::span<const double>(sub64).first(1), top3, parent3, 300, 16, {}), 1);
}

namespace {

// The 2% tie rule over (T, cost) pairs in ascending T, cost < 0 meaning no
// candidate, written out independently of least_cost_threads.
int pick_least(const std::vector<std::pair<int, double>> &costs) {
    double least = costs.front().second;
    for (const auto &k : costs)
        if (k.second >= 0.0) least = std::min(least, k.second);
    for (const auto &k : costs)
        if (k.second >= 0.0 && k.second <= least * 1.02) return k.first;
    return 1;
}

} // namespace

// The plan trace's terms are the model's: factor_plan_terms gives one entry
// per candidate (1, 2, 4, ..., max), the cost written out from its fields
// is factor_plan_cost and picks plan_threads' count, and each candidate's
// top-part count is partition_count's.
TEST(ThreadPlan, FactorTermsReproducePlan) {
    Synth bal;
    binary(bal, -1, 12, 1e7, 1e5);
    Synth cat;
    Int s = -1;
    for (int i = 0; i < 10000; ++i) {
        s = cat.add(s, 2e3);
        cat.add(s, 1e3);
    }
    Synth rooty;
    const Int root = rooty.add(-1, 1.5e9);
    binary(rooty, root, 9, 1e6, 1e3);
    binary(rooty, root, 9, 1e6, 1e3);
    Synth tiled;  // a split root over one split and one plain subtree root
    const Int troot = tiled.add(-1, 5e10, true);
    binary(tiled, troot, 9, 1e6, 1e3, true);
    binary(tiled, troot, 9, 1e6, 1e3);
    Synth light;  // below kTreeParallelMinWork: 1 alone
    binary(light, -1, 3, 1e5, 1e3);

    const TreeView views[] = {bal.view(), cat.view(), rooty.view(), tiled.view(), light.view()};
    const ThreadCosts costs[] = {ThreadCosts{},
                                 ThreadCosts{2e4, 1e3},
                                 ThreadCosts{5e5, 2e4},
                                 ThreadCosts{1e5, 5e3, 0.0, 16},
                                 ThreadCosts{2e5, 5e3, 1e6, 4},
                                 ThreadCosts{1e5, 5e3, 3e5, 0}};
    for (std::size_t v = 0; v < std::size(views); ++v) {
        for (const int max_threads : {1, 3, 8, 12, 16}) {
            const auto terms = factor_plan_terms(views[v], max_threads);
            ASSERT_FALSE(terms.empty());
            EXPECT_EQ(terms[0].threads, 1);
            EXPECT_TRUE(terms[0].ok);
            const bool candidates = max_threads >= 2 && terms[0].total >= kTreeParallelMinWork;
            if (!candidates) {
                EXPECT_EQ(terms.size(), 1u) << "tree " << v;
            } else {
                std::vector<int> want;
                for (int T = 2; T < max_threads; T *= 2) want.push_back(T);
                want.push_back(max_threads);
                ASSERT_EQ(terms.size(), want.size() + 1) << "tree " << v;
                for (std::size_t i = 0; i < want.size(); ++i) {
                    const auto &k = terms[i + 1];
                    EXPECT_EQ(k.threads, want[i]);
                    EXPECT_DOUBLE_EQ(k.total, terms[0].total);
                    const PartitionCount pc =
                        partition_count(views[v], k.threads, kTreeParallelMinWork);
                    EXPECT_EQ(k.top_fronts, pc.top);
                    EXPECT_EQ(k.ok, pc.subtrees >= 2);
                    EXPECT_LE(k.split_top_fronts, k.top_fronts);
                    EXPECT_LE(k.split_top_work, k.top_total);
                }
            }
            for (const ThreadCosts &c : costs) {
                std::vector<std::pair<int, double>> by_hand;
                for (const auto &k : terms) {
                    double cost = k.total;
                    if (k.threads > 1) {
                        const double T = k.threads;
                        const double cap = std::max(1, c.split_cap);
                        const double top =
                            std::max(k.cp_top, (k.top_total - k.split_top_work) / T +
                                                   k.split_top_work / std::min(T, cap));
                        cost = !k.ok ? -1.0
                                     : std::max(k.max_subtree, k.sub_total / T) + top +
                                           T * c.per_thread +
                                           static_cast<double>(k.top_fronts) * c.per_top_front +
                                           static_cast<double>(k.split_top_fronts) * T *
                                               c.per_split_front_thread;
                    }
                    EXPECT_DOUBLE_EQ(factor_plan_cost(k, c), cost);
                    by_hand.emplace_back(k.threads, cost);
                }
                const ThreadPlan p = plan_threads(views[v], max_threads, c);
                EXPECT_EQ(p.threads, pick_least(by_hand)) << "tree " << v << ", max " << max_threads;
            }
        }
    }

    // The split fronts of the top part: the tiled root at every candidate,
    // its work / T on the top path.
    const auto terms = factor_plan_terms(tiled.view(), 16);
    ASSERT_EQ(terms.size(), 5u);
    for (std::size_t i = 1; i < terms.size(); ++i) {
        EXPECT_GE(terms[i].split_top_fronts, 1u);
        EXPECT_GE(terms[i].split_top_work, 5e10);
        EXPECT_LT(terms[i].cp_top, 5e10 / terms[i].threads + 1e9);
    }
}

// factor_plan_cost, the terms written out by hand: the split work speeds up
// to split_cap threads, each split front costs per_split_front_thread per
// thread, cp_top is taken as traced.
TEST(ThreadPlan, FactorCostSplitTerms) {
    FactorPlanTerms k;
    k.threads = 16;
    k.ok = true;
    k.sub_total = 1.6e9;
    k.max_subtree = 1e7;
    k.top_total = 8e9;
    k.cp_top = 1e8;
    k.top_fronts = 40;
    k.split_top_fronts = 3;
    k.split_top_work = 6.4e9;
    k.total = 1e10;

    // Subtrees 1.6e9 / 16 = 1e8; top: (8e9 - 6.4e9) / 16 + 6.4e9 / 8 = 9e8;
    // 16 * 7e5 + 40 * 5e3 + 3 * 16 * 3e5 = 2.58e7.
    ThreadCosts c{7e5, 5e3, 3e5, 8};
    EXPECT_DOUBLE_EQ(factor_plan_cost(k, c), 1025800000.0);
    // Uncapped (cap = T): top = 1e8 + 4e8.
    c.split_cap = 16;
    EXPECT_DOUBLE_EQ(factor_plan_cost(k, c), 625800000.0);
    c.split_cap = 64;  // beyond T: min(T, cap) = T
    EXPECT_DOUBLE_EQ(factor_plan_cost(k, c), 625800000.0);
    // Below 1 counts as 1: the split work runs on one thread.
    for (const int cap : {0, 1, -3}) {
        c.split_cap = cap;
        EXPECT_DOUBLE_EQ(factor_plan_cost(k, c), 6625800000.0) << "cap " << cap;
    }
    // cp_top floors the top phase.
    c.split_cap = 8;
    k.cp_top = 2e9;
    EXPECT_DOUBLE_EQ(factor_plan_cost(k, c), 1e8 + 2e9 + 2.58e7);
    k.cp_top = 1e8;

    // T = 4 is under the cap: (8e9 - 6.4e9) / 4 + 6.4e9 / 4 = 2e9, subtrees
    // 4e8, 4 * 7e5 + 40 * 5e3 + 3 * 4 * 3e5 = 6.6e6.
    k.threads = 4;
    EXPECT_DOUBLE_EQ(factor_plan_cost(k, c), 2406600000.0);
    // No split fronts: the per-split term and the cap vanish.
    k.split_top_fronts = 0;
    k.split_top_work = 0.0;
    EXPECT_DOUBLE_EQ(factor_plan_cost(k, c), 4e8 + 2e9 + 4 * 7e5 + 40 * 5e3);
    // T = 1 is the total; !ok is -1.
    k.threads = 1;
    EXPECT_DOUBLE_EQ(factor_plan_cost(k, c), 1e10);
    k.threads = 16;
    k.ok = false;
    EXPECT_DOUBLE_EQ(factor_plan_cost(k, c), -1.0);
}

// A tree whose work is a tile-split root: capped at 8 threads, 16 buys
// nothing and costs 8 more threads' terms, so the plan stops at 8; uncapped
// it takes 16.
TEST(ThreadPlan, SplitCapStopsAtCap) {
    Synth rooty;
    const Int root = rooty.add(-1, 5e10, true);
    binary(rooty, root, 9, 1e6, 1e3);
    binary(rooty, root, 9, 1e6, 1e3);
    const TreeView t = rooty.view();
    ThreadCosts c;  // the defaults: cap 8
    EXPECT_EQ(c.split_cap, 8);
    const ThreadPlan capped = plan_threads(t, 16, c);
    EXPECT_EQ(capped.threads, 8);
    c.split_cap = 16;
    EXPECT_EQ(plan_threads(t, 16, c).threads, 16);
    // The cap never limits a tree with no split work.
    Synth bal;
    binary(bal, -1, 12, 1e7, 1e5);
    c.split_cap = 8;
    EXPECT_EQ(plan_threads(bal.view(), 16, c).threads, 16);
}

// plan_is_serial's per-thread bound, total <= 4 per_thread, is the model's
// and sharp: 1024 equal leaves under zero-work internal nodes at exactly
// the bound tie at T = 2 (1 kept), and 10% above it 2 threads win. Any tree
// shape at or below it, any max_threads and cost set, picks 1.
TEST(ThreadPlan, SerialBoundFromPerThreadCost) {
    ThreadCosts c;
    c.per_thread = 1e8;  // bound 4e8, above kTreeParallelMinWork
    const double bound = 4.0 * c.per_thread;
    ASSERT_GT(bound, kTreeParallelMinWork);
    Synth at;
    binary(at, -1, 10, bound / 1024.0, 0.0);
    Synth above;
    binary(above, -1, 10, 1.1 * bound / 1024.0, 0.0);
    EXPECT_TRUE(plan_is_serial(bound, c));
    EXPECT_FALSE(plan_is_serial(1.1 * bound, c));
    for (const int max_threads : {2, 3, 4, 8, 12, 16})
        EXPECT_EQ(plan_threads(at.view(), max_threads, c).threads, 1) << "max " << max_threads;
    EXPECT_GT(plan_threads(above.view(), 16, c).threads, 1);

    // Other shapes scaled to the bound or below.
    Synth rooty;  // a heavy split root over balanced subtrees
    const Int root = rooty.add(-1, 0.25 * bound, true);
    binary(rooty, root, 8, 0.375 * bound / 256.0, 0.0);
    binary(rooty, root, 8, 0.375 * bound / 256.0, 0.0, true);
    Synth wide;  // 4096 leaves under one root
    const Int wroot = wide.add(-1, 0.0);
    for (int i = 0; i < 4096; ++i) wide.add(wroot, bound / 4096.0);
    Synth half;
    binary(half, -1, 12, 0.5 * bound / 4096.0, 1.0);
    const ThreadCosts sets[] = {c, ThreadCosts{1e8, 0.0, 0.0, 1}, ThreadCosts{1e8, 1e3, 1e5, 16}};
    for (Synth *s : {&at, &rooty, &wide, &half}) {
        const TreeView t = s->view();
        double total = 0.0;
        for (const double w : t.work) total += w;
        ASSERT_LE(total, bound);
        for (const ThreadCosts &k : sets) {
            EXPECT_TRUE(plan_is_serial(total, k));
            for (const int max_threads : {2, 3, 4, 8, 12, 16})
                EXPECT_EQ(plan_threads(t, max_threads, k).threads, 1) << "max " << max_threads;
        }
    }

    // A negative cost voids the per-thread bound; kTreeParallelMinWork
    // stands whatever the costs.
    ThreadCosts neg = c;
    neg.per_top_front = -1.0;
    EXPECT_FALSE(plan_is_serial(bound, neg));
    EXPECT_TRUE(plan_is_serial(0.5 * kTreeParallelMinWork, neg));
    EXPECT_TRUE(plan_is_serial(0.0, ThreadCosts{}));
    // The defaults: the per-thread bound (2.8e6) lies below the minimum work.
    EXPECT_TRUE(plan_is_serial(kTreeParallelMinWork - 1.0, ThreadCosts{}));
    EXPECT_FALSE(plan_is_serial(kTreeParallelMinWork, ThreadCosts{}));
}

// The factorization skips the plan on a tree plan_is_serial rules serial:
// no plan is cached (NumericWorkspace::thread_plan stays empty) and no
// tree-parallel buffers or partition are made, at 16 threads with
// auto_threads on, with a light tree (below kTreeParallelMinWork) and with
// a heavier one under a per-thread cost that rules it out. The same heavier
// tree under the default costs builds its plan.
TEST(ThreadPlan, SerialBoundSkipsPlan) {
    const auto tree_work = [](const AnalysisKeep &keep) {
        double w = 0.0;
        for (std::size_t s = 0; s < keep.supernodes.size(); ++s) {
            const double f = static_cast<double>(keep.front_size(static_cast<Int>(s)));
            w += f * f * static_cast<double>(keep.supernodes[s].width());
        }
        return w;
    };
    struct Case {
        int m;
        double per_thread;
        bool serial;
    };
    for (const Case cs : {Case{30, 7e5, true}, Case{120, 1e12, true}, Case{120, 7e5, false}}) {
        const CscLower A = laplacian_2d(cs.m);
        Control ctrl;
        ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
        ctrl.num_threads = 16;
        ctrl.thread_cost_per_thread = cs.per_thread;
        Solver solver;
        Info info;
        auto ak = solver.analyse(A, ctrl, info);
        ASSERT_NE(ak.get(), nullptr);
        const double total = tree_work(*ak);
        ThreadCosts costs;
        costs.per_thread = cs.per_thread;
        ASSERT_EQ(plan_is_serial(total, costs), cs.serial) << "premise, m=" << cs.m;
        FactorKeep fk;
        NumericWorkspace ws;
        ASSERT_EQ(solver.factor(*ak, A.values, ctrl, info, fk, ws), FactorStatus::Success);
        if (cs.serial) {
            EXPECT_EQ(info.threads_used, 1) << "m=" << cs.m;
            EXPECT_EQ(ws.thread_plan.analysis, 0u) << "m=" << cs.m << ": a plan was built";
            EXPECT_TRUE(ws.tree_par == nullptr) << "m=" << cs.m;
        } else {
            EXPECT_EQ(ws.thread_plan.analysis, ak->serial) << "m=" << cs.m;
        }
    }
}

// The trace text: one plan-factor line per candidate and the chosen count.
TEST(ThreadPlan, FactorTraceFormat) {
    Synth bal;
    binary(bal, -1, 12, 1e7, 1e5);
    const auto terms = factor_plan_terms(bal.view(), 4);
    const std::string text = factor_plan_trace(terms, 4);
    const std::string first = text.substr(0, text.find('\n'));
    EXPECT_EQ(first.rfind("plan-factor T=1 ok=1 sub_total=", 0), 0u) << first;
    EXPECT_NE(text.find("\nplan-factor T=2 ok=1 sub_total="), std::string::npos) << text;
    EXPECT_NE(text.find(" split_top_fronts=0 split_top_work=0 total="), std::string::npos) << text;
    EXPECT_NE(text.find("\nplan-factor T=4 ok=1 "), std::string::npos) << text;
    EXPECT_NE(text.find("\nplan-factor chosen=4\n"), std::string::npos) << text;
}

// The solve's terms: solve_plan_cost is the documented formula and picks
// plan_solve_threads' count; the trace prints them per candidate.
TEST(ThreadPlan, SolveTermsReproducePlan) {
    const std::vector<double> sub64(64, 1e6);
    const std::vector<double> top3{1e3, 2e3, 4e3};
    const std::vector<Int> parent3{2, 2, -1};
    const std::vector<double> sub2{1e4, 1e4};
    const std::vector<double> top5000(5000, 50.0);
    std::vector<Int> chain_parent(5000);
    for (std::size_t i = 0; i < chain_parent.size(); ++i) chain_parent[i] = static_cast<Int>(i) + 1;
    chain_parent.back() = -1;

    struct Case {
        std::span<const double> sub, top;
        std::span<const Int> parent;
        std::size_t rows;
    };
    const Case cases[] = {{sub64, top3, parent3, 300},
                          {sub64, top3, parent3, 300000},
                          {sub2, top5000, chain_parent, 4000}};
    const SolveCosts costs[] = {SolveCosts{}, SolveCosts{5e3, 5e2, 0.5}, SolveCosts{1e5, 5e3, 4.0}};
    for (const Case &k : cases) {
        const SolvePlanTerms st = solve_plan_terms(k.sub, k.top, k.parent, k.rows);
        EXPECT_EQ(st.n_top, k.top.size());
        EXPECT_EQ(st.top_rows, k.rows);
        if (k.top.size() == 3) {
            EXPECT_DOUBLE_EQ(st.sub_total, 6.4e7);
            EXPECT_DOUBLE_EQ(st.max_subtree, 1e6);
            EXPECT_DOUBLE_EQ(st.top_total, 7e3);
            EXPECT_DOUBLE_EQ(st.top_cp, 6e3);  // 2e3 + 4e3
        }
        for (const SolveCosts &c : costs) {
            for (const int max_threads : {2, 3, 8, 16}) {
                std::vector<std::pair<int, double>> by_hand{{1, st.sub_total + st.top_total}};
                for (const int T : candidate_threads(max_threads)) {
                    if (T == 1) continue;
                    const double cost = std::max(st.max_subtree, st.sub_total / T) + st.top_cp +
                                        T * c.per_thread +
                                        static_cast<double>(st.n_top) * c.per_top_step +
                                        T * static_cast<double>(st.top_rows) * c.per_top_row;
                    EXPECT_DOUBLE_EQ(solve_plan_cost(st, T, c), cost);
                    by_hand.emplace_back(T, cost);
                }
                EXPECT_DOUBLE_EQ(solve_plan_cost(st, 1, c), st.sub_total + st.top_total);
                EXPECT_EQ(plan_solve_threads(k.sub, k.top, k.parent, k.rows, max_threads, c),
                          pick_least(by_hand));
            }
        }
    }
    EXPECT_EQ(candidate_threads(1), std::vector<int>{1});
    EXPECT_EQ(candidate_threads(12), (std::vector<int>{1, 2, 4, 8, 12}));
    EXPECT_EQ(candidate_threads(16), (std::vector<int>{1, 2, 4, 8, 16}));

    const SolvePlanTerms st = solve_plan_terms(sub64, top3, parent3, 300);
    const std::string text = solve_plan_trace(st, 4, 2);
    EXPECT_EQ(text.rfind("plan-solve T=1 sub_total=6.4e+07 max_subtree=1e+06 top_total=7000 "
                         "top_cp=6000 n_top=3 top_rows=300\n",
                         0),
              0u)
        << text;
    EXPECT_NE(text.find("\nplan-solve T=4 sub_total="), std::string::npos) << text;
    EXPECT_NE(text.find("\nplan-solve chosen=2\n"), std::string::npos) << text;
}
