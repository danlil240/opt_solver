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

#include <cstdint>
#include <iterator>
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

// A root with 60% of the work: unsplit it bounds the speedup at 1/0.6
// (2 threads are as good as more, within 2%); split into tiles it scales.
TEST(ThreadPlan, SerialRootCapsThreads) {
    for (const bool root_split : {false, true}) {
        Synth rooty;
        const Int root = rooty.add(-1, 1.5e9, root_split);
        binary(rooty, root, 9, 1e6, 1e3);
        binary(rooty, root, 9, 1e6, 1e3);
        const TreeView t = rooty.view();
        const ThreadPlan p = plan_threads(t, 16, {});
        if (root_split) {
            EXPECT_EQ(p.threads, 16);
        } else {
            EXPECT_LE(p.threads, 2);
            EXPECT_LT(p.predicted_speedup, 1.0 / 0.59);
        }
    }
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

// plan_threads counts each candidate's partition without the heap; the
// counts must be partition_subtrees' sizes.
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
            std::vector<std::uint8_t> top;
            const auto sub = partition_subtrees(views[v], T, kTreeParallelMinWork, top);
            const PartitionCount pc = partition_count(views[v], T, kTreeParallelMinWork);
            std::size_t ntop = 0;
            for (const auto x : top) ntop += x;
            EXPECT_EQ(pc.subtrees, sub.size()) << "tree " << v << ", T = " << T;
            EXPECT_EQ(pc.top, ntop) << "tree " << v << ", T = " << T;
        }
    }
}
