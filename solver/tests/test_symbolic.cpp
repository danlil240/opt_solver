#include "smf/solver.hpp"
#include "smf/analysis.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/control.hpp"
#include "smf/info.hpp"
#include "smf/types.hpp"
#include "smf/check_matrix.hpp"
#include "smf/etree.hpp"
#include "smf/matrix_market.hpp"
#include "smf/ordering.hpp"
#include "smf/supernode.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <cmath>
#include <filesystem>
#include <random>
#include <string>
#include <utility>

namespace smf {

// Build a 50x50 SPD banded lower CSC (bandwidth 3)
static CscLower make_spd50() {
    const Int n = 50;
    const Int bw = 3;
    CscLower A;
    A.n = n;
    A.col_ptr.resize(n + 1, 0);
    for (Int j = 0; j < n; ++j) {
        Int cnt = 1; // diagonal
        for (Int k = 1; k <= bw && j + k < n; ++k) ++cnt;
        A.col_ptr[j + 1] = A.col_ptr[j] + cnt;
    }
    A.row_idx.reserve(A.col_ptr[n]);
    A.values.reserve(A.col_ptr[n]);
    for (Int j = 0; j < n; ++j) {
        A.row_idx.push_back(j);          // diagonal
        A.values.push_back(10.0 + bw);   // dominant diagonal
        for (Int k = 1; k <= bw && j + k < n; ++k) {
            A.row_idx.push_back(j + k);
            A.values.push_back(-1.0);
        }
    }
    return A;
}

// End-to-end: analyse a 50x50 SPD matrix, check all predictions populated
TEST(Symbolic, Analyse50x50) {
    Solver solver;
    CscLower A = make_spd50();
    Control ctrl{};
    Info info{};

    auto keep = solver.analyse(A, ctrl, info);

    ASSERT_NE(keep.get(), nullptr) << "analyse returned nullptr";
    EXPECT_EQ(info.status, ErrorCode::Success);
    EXPECT_EQ(keep->n, 50);

    // Permutation vectors should be valid
    ASSERT_EQ((Int)keep->perm.size(), 50);
    ASSERT_EQ((Int)keep->iperm.size(), 50);
    // perm and iperm are inverses
    for (Int i = 0; i < 50; ++i) {
        EXPECT_EQ(keep->iperm[keep->perm[i]], i) << "i=" << i;
    }

    // Predictions should be positive
    EXPECT_GT(info.predicted_factor_entries, 0);
    EXPECT_GT(info.predicted_flops, 0.0);
    EXPECT_GT(info.max_front_size, 0);
    EXPECT_GT(info.max_supernode_size, 0);

    // Timing should be non-negative
    EXPECT_GE(info.analyse_seconds, 0.0);
}

// n=0 edge case
TEST(Symbolic, AnalyseEmpty) {
    Solver solver;
    CscLower A;
    A.n = 0;
    A.col_ptr = {0};
    Control ctrl{};
    Info info{};
    auto keep = solver.analyse(A, ctrl, info);
    ASSERT_NE(keep.get(), nullptr);
    EXPECT_EQ(keep->n, 0);
    EXPECT_EQ(info.status, ErrorCode::Success);
}

// Diagonal 5x5 matrix (no off-diagonal entries)
TEST(Symbolic, AnalyseDiagonal5) {
    Solver solver;
    CscLower A;
    A.n = 5;
    A.col_ptr = {0, 1, 2, 3, 4, 5};
    A.row_idx = {0, 1, 2, 3, 4};
    A.values  = {1, 1, 1, 1, 1};
    Control ctrl{};
    Info info{};
    auto keep = solver.analyse(A, ctrl, info);
    ASSERT_NE(keep.get(), nullptr);
    EXPECT_EQ(keep->n, 5);
    EXPECT_EQ(info.status, ErrorCode::Success);
    EXPECT_GT(info.predicted_factor_entries, 0);
}

} // namespace smf

namespace smf {
namespace {

// OPF-like saddle-point KKT: n_x primal variables (5% with tiny curvature),
// m constraints with zero diagonal and 2-4 random Jacobian entries each.
CscLower opf_like_kkt(int n_x, int m, unsigned seed) {
    std::uint64_t state = seed;
    auto next = [&]() { state = state * 6364136223846793005ULL + 1442695040888963407ULL; return state >> 11; };
    auto unit = [&]() { return static_cast<double>(next()) / 9007199254740992.0; };
    const int n = n_x + m;
    std::vector<std::vector<std::pair<Int, double>>> cols(static_cast<std::size_t>(n));
    for (int j = 0; j < n_x; ++j)
        cols[static_cast<std::size_t>(j)].push_back({j, unit() < 0.05 ? 1e-5 : 1.0 + unit()});
    for (int r = 0; r < m; ++r) {
        const int row = n_x + r;
        const int k = 2 + static_cast<int>(next() % 3);
        std::vector<int> used;
        for (int t = 0; t < k; ++t) {
            const int j = static_cast<int>(next() % static_cast<std::uint64_t>(n_x));
            if (std::find(used.begin(), used.end(), j) != used.end()) continue;
            used.push_back(j);
            cols[static_cast<std::size_t>(j)].push_back({row, 0.5 + unit()});
        }
        cols[static_cast<std::size_t>(row)].push_back({row, 0.0});
    }
    CscLower A;
    A.n = n;
    A.col_ptr.assign(static_cast<std::size_t>(n) + 1, 0);
    for (int j = 0; j < n; ++j) {
        auto &c = cols[static_cast<std::size_t>(j)];
        std::sort(c.begin(), c.end());
        A.col_ptr[static_cast<std::size_t>(j) + 1] = A.col_ptr[static_cast<std::size_t>(j)] + static_cast<Int>(c.size());
        for (const auto &e : c) {
            A.row_idx.push_back(e.first);
            A.values.push_back(e.second);
        }
    }
    return A;
}

} // namespace

// ScalingMethod::Auto applies the analysis' matching scaling. That scaling
// only helps together with the matching ordering that keeps 2x2 pairs
// adjacent; with AMD kept it degrades accuracy (pglib case9241 failed the
// replay accuracy gate on 17 of 73 steps through it). An analysis that kept
// AMD must therefore carry no matching scale.
TEST(Symbolic, AutoMatchingScaleOnlyWithMatchingOrdering) {
    const CscLower A = opf_like_kkt(6000, 5000, 7);
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    ctrl.ordering = OrderingMethod::AutoSerial;
    Info info;
    Solver solver;
    auto ak = solver.analyse(A, ctrl, info);
    ASSERT_NE(ak, nullptr);
    ASSERT_EQ(info.ordering_used, static_cast<int>(OrderingMethod::AMD))
        << "fixture must make the auto policy keep AMD";
    EXPECT_TRUE(ak->matching_scale.empty());
}

// Per-step analyse timing: every phase is non-negative, ordering is measured,
// and the phases account for (nearly) all of analyse_seconds.
TEST(Symbolic, AnalysePhasesSumToTotal) {
    const Int g = 60, n = g * g;
    CscLower A;
    A.n = n;
    A.col_ptr.assign(1, 0);
    for (Int j = 0; j < n; ++j) {
        const Int x = j % g, y = j / g;
        A.row_idx.push_back(j);
        A.values.push_back(4.0);
        if (x + 1 < g) { A.row_idx.push_back(j + 1); A.values.push_back(-1.0); }
        if (y + 1 < g) { A.row_idx.push_back(j + g); A.values.push_back(-1.0); }
        A.col_ptr.push_back(static_cast<Int>(A.row_idx.size()));
    }
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
    ctrl.ordering = OrderingMethod::AMD;
    Info info;
    Solver solver;
    auto ak = solver.analyse(A, ctrl, info);
    ASSERT_NE(ak, nullptr);
    const AnalysePhaseSeconds &p = info.analyse_phases;
    EXPECT_GE(p.check, 0.0);
    EXPECT_GT(p.ordering, 0.0);
    EXPECT_GE(p.matching, 0.0);
    EXPECT_GE(p.etree_counts, 0.0);
    EXPECT_GE(p.amalgamate, 0.0);
    EXPECT_GE(p.assembly_tree, 0.0);
    EXPECT_GE(p.keep_build, 0.0);
    const double sum = p.check + p.ordering + p.matching + p.etree_counts +
                       p.amalgamate + p.assembly_tree + p.keep_build;
    ASSERT_GT(info.analyse_seconds, 0.0);
    EXPECT_LE(sum, info.analyse_seconds * 1.0000001);
    EXPECT_GE(sum, 0.8 * info.analyse_seconds);
}

} // namespace smf

// ---------------------------------------------------------------------------
// Small-n analyse fast path: it must reproduce the reference symbolic
// pipeline exactly (same AMD permutation, etree, column counts, supernode
// partition), and Control::check_input = false must not change the result on
// clean input.
namespace smf {
namespace {

bool load_optim20(const char *name, CscLower &A) {
    const std::filesystem::path mtx = std::filesystem::path(__FILE__).parent_path() / ".." / ".." /
                                      "benchmarks" / "fixtures" / "optim20" / (std::string(name) + ".mtx");
    std::error_code ec;
    if (!std::filesystem::exists(mtx, ec)) return false;
    auto mm = read_matrix_market(mtx.string());
    if (!mm.ok) return false;
    A = std::move(mm.matrix);
    return true;
}

// Random sparse SPD pattern: diagonal plus up to 3 random rows below it per
// column.
CscLower random_spd(Int n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<Int> pick(0, n - 1);
    CscLower A;
    A.n = n;
    A.col_ptr.assign(1, 0);
    for (Int j = 0; j < n; ++j) {
        std::vector<Int> rows{j};
        for (int t = 0; t < 3; ++t) {
            const Int i = pick(rng);
            if (i > j) rows.push_back(i);
        }
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        for (Int i : rows) {
            A.row_idx.push_back(i);
            A.values.push_back(i == j ? 10.0 : -1.0);
        }
        A.col_ptr.push_back(static_cast<Int>(A.row_idx.size()));
    }
    return A;
}

// Lower triangle of P C P^T by rows (iperm[old] = new).
void row_form(const CscLower &C, const std::vector<Int> &iperm, std::vector<Int> &rp,
              std::vector<Int> &rc) {
    const Int n = C.n;
    rp.assign(static_cast<std::size_t>(n) + 1, 0);
    for (Int j = 0; j < n; ++j)
        for (Int k = C.col_ptr[j]; k < C.col_ptr[j + 1]; ++k)
            ++rp[static_cast<std::size_t>(std::max(iperm[C.row_idx[k]], iperm[j])) + 1];
    for (Int r = 0; r < n; ++r) rp[r + 1] += rp[r];
    rc.assign(static_cast<std::size_t>(rp[n]), 0);
    std::vector<Int> head(rp.begin(), rp.end() - 1);
    for (Int j = 0; j < n; ++j)
        for (Int k = C.col_ptr[j]; k < C.col_ptr[j + 1]; ++k) {
            const Int a = iperm[C.row_idx[k]], b = iperm[j];
            rc[static_cast<std::size_t>(head[std::max(a, b)]++)] = std::min(a, b);
        }
}

struct RefSymbolic {
    std::vector<Int> perm, parent;
    std::vector<Supernode> snodes;
};

// The reference pipeline (analyse before the fast path) for the auto/AMD
// orderings: identity for banded matrices, else AMD on the lower CSC; etree
// and counts from the row form; postorder relabel; relaxed amalgamation.
RefSymbolic reference_symbolic(const CscLower &C, Int nemin) {
    const Int n = C.n;
    Int bw = 0;
    for (Int j = 0; j < n; ++j)
        if (C.col_ptr[j + 1] > C.col_ptr[j]) bw = std::max(bw, C.row_idx[C.col_ptr[j + 1] - 1] - j);
    const LongInt nnz = C.col_ptr[n];
    const bool dense_band = static_cast<LongInt>(bw + 1) * n <= 2 * nnz;
    std::vector<Int> perm(static_cast<std::size_t>(n));
    std::vector<Int> work;
    if (bw <= 64 && (dense_band || n <= 2048))
        std::iota(perm.begin(), perm.end(), 0);
    else
        EXPECT_TRUE(amd_order_lower_csc(n, C.col_ptr, C.row_idx, perm, work));
    std::vector<Int> iperm(perm.size()), rp, rc;
    for (Int k = 0; k < n; ++k) iperm[perm[k]] = k;
    row_form(C, iperm, rp, rc);
    const EliminationTree et = build_elimination_tree_from_rows(n, rp.data(), rc.data());
    const std::vector<Int> cc = filled_column_counts_from_rows(n, rp.data(), rc.data(), et);
    std::vector<Int> inv_po(perm.size()), parent1(perm.size()), cc1(perm.size());
    for (Int k = 0; k < n; ++k) inv_po[et.postorder[k]] = k;
    for (Int k = 0; k < n; ++k) {
        const Int old = et.postorder[k];
        parent1[k] = et.parent[old] >= 0 ? inv_po[et.parent[old]] : -1;
        cc1[k] = cc[old];
    }
    RefSymbolic ref;
    std::vector<Int> order;
    amalgamate_relaxed(n, parent1.data(), cc1.data(), nemin, order, ref.snodes);
    std::vector<Int> inv_order(perm.size());
    for (Int k = 0; k < n; ++k) inv_order[order[k]] = k;
    ref.perm.resize(perm.size());
    ref.parent.resize(perm.size());
    for (Int k = 0; k < n; ++k) {
        const Int k1 = order[k];
        ref.parent[k] = parent1[k1] >= 0 ? inv_order[parent1[k1]] : -1;
        ref.perm[k] = perm[et.postorder[k1]];
    }
    return ref;
}

void expect_fast_path_matches(const CscLower &A, MatrixType type) {
    Info cinfo;
    const CscLower C = clean_lower_csc(A, cinfo).clean;
    const Int n = C.n;
    const auto sn = static_cast<std::size_t>(n);

    // Adjacency built in one fill pass, then AMD on it: same permutation as
    // AMD on the lower CSC.
    std::vector<Int> xadj(sn + 1);
    std::vector<Int> adj(lower_csc_adjacency_size(n, C.col_ptr.data(), C.row_idx.data()));
    lower_csc_adjacency(n, C.col_ptr.data(), C.row_idx.data(), xadj.data(), adj.data());
    ASSERT_EQ(static_cast<std::size_t>(xadj[sn]), adj.size());
    std::vector<Int> p_ref(sn), p_new(sn), amd_work;
    ASSERT_TRUE(amd_order_lower_csc(n, C.col_ptr, C.row_idx, p_ref, amd_work));
    std::vector<Int> work(amd_adjacency_work_size(n, adj.size()));
    ASSERT_TRUE(amd_order_adjacency(n, xadj.data(), adj.data(), p_new.data(), work.data()));
    EXPECT_EQ(p_new, p_ref);

    // Etree, postorder and column counts from the adjacency, for that
    // permutation: the same as from the row form.
    std::vector<Int> iperm(sn), rp, rc;
    for (Int k = 0; k < n; ++k) iperm[p_ref[k]] = k;
    row_form(C, iperm, rp, rc);
    const EliminationTree et = build_elimination_tree_from_rows(n, rp.data(), rc.data());
    const std::vector<Int> cc = filled_column_counts_from_rows(n, rp.data(), rc.data(), et);
    std::vector<Int> parent(sn), post(sn), cnt(sn), w(4 * sn + 1);
    etree_from_adjacency(n, xadj.data(), adj.data(), p_ref.data(), iperm.data(), parent.data(),
                         w.data());
    const Int depth = etree_postorder(n, parent.data(), post.data(), w.data());
    filled_column_counts_from_adjacency(n, xadj.data(), adj.data(), p_ref.data(), iperm.data(),
                                        parent.data(), post.data(), cnt.data(), w.data());
    EXPECT_EQ(parent, et.parent);
    EXPECT_EQ(post, et.postorder);
    EXPECT_EQ(depth, et.max_depth);
    EXPECT_EQ(cnt, cc);

    // End to end: analyse's permutation, etree and supernode partition.
    Control ctrl;
    ctrl.matrix_type = type;
    Info info;
    Solver solver;
    auto ak = solver.analyse(A, ctrl, info);
    ASSERT_NE(ak, nullptr);
    const RefSymbolic ref = reference_symbolic(C, static_cast<Int>(ctrl.nemin));
    EXPECT_EQ(ak->perm, ref.perm);
    EXPECT_EQ(ak->etree.parent, ref.parent);
    ASSERT_EQ(ak->supernodes.size(), ref.snodes.size());
    for (std::size_t s = 0; s < ref.snodes.size(); ++s) {
        EXPECT_EQ(ak->supernodes[s].col_start, ref.snodes[s].col_start) << "s=" << s;
        EXPECT_EQ(ak->supernodes[s].col_end, ref.snodes[s].col_end) << "s=" << s;
        EXPECT_EQ(ak->supernodes[s].parent, ref.snodes[s].parent) << "s=" << s;
    }
}

const char *const kA1Cases[] = {"spd_barrier_800",    "kkt_network_flow", "kkt_auglag",
                                "indef_helmholtz_50", "spd_mrf_2500",     "spd_poisson2d_60"};

MatrixType a1_type(const std::string &name) {
    return name.rfind("spd_", 0) == 0 ? MatrixType::RealSymmetricPositiveDefinite
                                      : MatrixType::RealSymmetricIndefinite;
}

} // namespace

TEST(Symbolic, FastPathMatchesReference) {
    for (const char *name : kA1Cases) {
        CscLower A;
        ASSERT_TRUE(load_optim20(name, A)) << "fixture " << name << " missing";
        SCOPED_TRACE(name);
        expect_fast_path_matches(A, a1_type(name));
    }
    SCOPED_TRACE("random_spd seed 7");
    expect_fast_path_matches(random_spd(3000, 7), MatrixType::RealSymmetricPositiveDefinite);
}

} // namespace smf
