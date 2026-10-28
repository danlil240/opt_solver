#include "../benchmarks/common/validation.hpp"
#include "smf/analysis.hpp"
#include "smf/assembly_tree.hpp"
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/etree.hpp"
#include "smf/info.hpp"
#include "smf/matrix_market.hpp"
#include "smf/solver.hpp"
#include "smf/supernode.hpp"
#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <span>
#include <vector>

namespace smf {

// Helper: 5x5 dense lower CSC (all entries)
static CscLower make_dense5() {
    CscLower A;
    A.n = 5;
    A.col_ptr = {0, 5, 9, 12, 14, 15};
    A.row_idx = {0,1,2,3,4, 1,2,3,4, 2,3,4, 3,4, 4};
    A.values.resize(15, 1.0);
    return A;
}

// Helper: single supernode covering all 5 columns
static std::vector<Supernode> one_supernode5() {
    Supernode sn;
    sn.col_start = 0;
    sn.col_end = 5;
    sn.parent = -1;
    return {sn};
}

// Children CSR + front rows for a hand-made supernode list.
struct FlatTree {
    std::vector<Int> child_ptr, child_idx, row_ptr, row_idx;
    Int front_size(Int s) const { return row_ptr[s + 1] - row_ptr[s]; }
};

static FlatTree build_flat(const CscLower &A, const std::vector<Supernode> &snodes,
                           Info &info) {
    FlatTree t;
    build_children_csr(snodes, t.child_ptr, t.child_idx);
    build_assembly_tree(A, snodes, t.child_ptr, t.child_idx, t.row_ptr, t.row_idx, info);
    return t;
}

// Dense 5x5: one supernode, front_size = 5, predicted_factor_entries = 5
TEST(AssemblyTree, Dense5x5) {
    CscLower A = make_dense5();
    auto snodes = one_supernode5();
    Info info{};
    const FlatTree t = build_flat(A, snodes, info);

    ASSERT_EQ(t.row_ptr.size(), 2u);
    EXPECT_EQ(t.front_size(0), 5);
    EXPECT_EQ(info.predicted_factor_entries, 5);
    // flops = width * front_size^2 = 5 * 25 = 125
    EXPECT_DOUBLE_EQ(info.predicted_flops, 125.0);
    EXPECT_EQ(info.max_front_size, 5);
    EXPECT_EQ(info.max_supernode_size, 5);
}

// 10x10 banded (tridiagonal): each column has 2 entries (diag + 1 subdiag),
// except last which has 1. With n singleton supernodes, each front has size 2
// (except last = size 1).
TEST(AssemblyTree, Banded10x10Singletons) {
    const Int n = 10;
    CscLower A;
    A.n = n;
    A.col_ptr.resize(n + 1, 0);
    for (Int j = 0; j < n; ++j)
        A.col_ptr[j + 1] = A.col_ptr[j] + (j < n - 1 ? 2 : 1);
    A.row_idx.clear();
    for (Int j = 0; j < n; ++j) {
        A.row_idx.push_back(j);
        if (j < n - 1) A.row_idx.push_back(j + 1);
    }
    A.values.resize(A.row_idx.size(), 1.0);

    // Make n singleton supernodes in a chain
    std::vector<Supernode> snodes(n);
    for (Int j = 0; j < n; ++j) {
        snodes[j].col_start = j;
        snodes[j].col_end = j + 1;
        snodes[j].parent = (j < n - 1) ? j + 1 : -1;
    }

    Info info{};
    const FlatTree t = build_flat(A, snodes, info);

    ASSERT_EQ(t.row_ptr.size(), static_cast<std::size_t>(n) + 1);
    for (Int j = 0; j < n - 1; ++j)
        EXPECT_EQ(t.front_size(j), 2) << "j=" << j;
    EXPECT_EQ(t.front_size(n - 1), 1);
    // Chain: every node but the first has exactly its predecessor as child.
    EXPECT_EQ(t.child_ptr[1] - t.child_ptr[0], 0);
    for (Int j = 1; j < n; ++j) {
        ASSERT_EQ(t.child_ptr[j + 1] - t.child_ptr[j], 1);
        EXPECT_EQ(t.child_idx[t.child_ptr[j]], j - 1);
    }

    EXPECT_GT(info.predicted_factor_entries, 0);
    EXPECT_GT(info.predicted_flops, 0.0);
    EXPECT_EQ(info.max_front_size, 2);
    EXPECT_EQ(info.max_supernode_size, 1);
}

// Empty matrix
TEST(AssemblyTree, Empty) {
    CscLower A;
    A.n = 0;
    A.col_ptr = {0};
    std::vector<Supernode> snodes;
    Info info{};
    const FlatTree t = build_flat(A, snodes, info);
    EXPECT_TRUE(t.row_idx.empty());
    EXPECT_EQ(t.row_ptr.size(), 1u);
    EXPECT_EQ(info.predicted_factor_entries, 0);
    EXPECT_DOUBLE_EQ(info.predicted_flops, 0.0);
}

// Predictions are non-negative and max values are positive for non-trivial input
TEST(AssemblyTree, PredictionsNonNegative) {
    CscLower A = make_dense5();
    auto snodes = one_supernode5();
    Info info{};
    build_flat(A, snodes, info);
    EXPECT_GE(info.predicted_factor_entries, 0);
    EXPECT_GE(info.predicted_flops, 0.0);
    EXPECT_GE(info.max_front_size, 0);
    EXPECT_GE(info.max_supernode_size, 0);
    EXPECT_GE(info.max_tree_depth, 0);
}

// ---------------------------------------------------------------------------
// Flat CSR storage vs the old nested (vector-of-vectors) builder
// ---------------------------------------------------------------------------

namespace {

namespace fs = std::filesystem;

// Test-local copy of the pre-CSR builder: children pushed in increasing
// index from the parent links, then reversed (descending postorder), and
// each front's rows = its pivot columns, its columns' pattern and its
// children's extension rows, sorted.
struct NestedTree {
    std::vector<std::vector<Int>> children;
    std::vector<std::vector<Int>> rows;
};

NestedTree nested_builder(const CscLower &A, const std::vector<Supernode> &sn) {
    const Int nsn = static_cast<Int>(sn.size());
    NestedTree t;
    t.children.resize(sn.size());
    t.rows.resize(sn.size());
    for (Int s = 0; s < nsn; ++s)
        if (sn[s].parent >= 0) t.children[sn[s].parent].push_back(s);
    for (auto &ch : t.children) std::reverse(ch.begin(), ch.end());
    std::vector<Int> mark(static_cast<std::size_t>(A.n), -1);
    for (Int s = 0; s < nsn; ++s) {
        std::vector<Int> dirty;
        auto add = [&](Int r) {
            if (mark[r] != s) { mark[r] = s; dirty.push_back(r); }
        };
        for (Int j = sn[s].col_start; j < sn[s].col_end; ++j) add(j);
        for (Int j = sn[s].col_start; j < sn[s].col_end; ++j)
            for (Int p = A.col_ptr[j]; p < A.col_ptr[j + 1]; ++p) add(A.row_idx[p]);
        for (Int c : t.children[s])
            for (Int r : t.rows[c])
                if (r >= sn[c].col_end) add(r);
        std::sort(dirty.begin(), dirty.end());
        t.rows[s] = std::move(dirty);
    }
    return t;
}

// The permuted pattern the analysis built its fronts from.
CscLower permuted_pattern(const AnalysisKeep &ak) {
    if (ak.perm_col_ptr.empty()) return ak.cleaned;
    CscLower P;
    P.n = ak.n;
    P.col_ptr = ak.perm_col_ptr;
    P.row_idx = ak.perm_row_idx;
    return P;
}

void expect_flat_matches_nested(const AnalysisKeep &ak) {
    const NestedTree ref = nested_builder(permuted_pattern(ak), ak.supernodes);
    const Int nsn = static_cast<Int>(ak.supernodes.size());
    ASSERT_GT(nsn, 1);
    ASSERT_EQ(ak.num_supernodes(), nsn);
    // analyse sizes front_row_idx exactly from the column counts (one allocation).
    EXPECT_EQ(ak.front_row_idx.capacity(), ak.front_row_idx.size());
    for (Int s = 0; s < nsn; ++s) {
        const std::span<const Int> ch = ak.children(s);
        ASSERT_EQ(ch.size(), ref.children[s].size()) << "s=" << s;
        for (std::size_t k = 0; k < ch.size(); ++k)
            EXPECT_EQ(ch[k], ref.children[s][k]) << "s=" << s << " k=" << k;
        const std::span<const Int> rows = ak.front_rows(s);
        ASSERT_EQ(rows.size(), ref.rows[s].size()) << "s=" << s;
        EXPECT_EQ(ak.front_size(s), static_cast<Int>(ref.rows[s].size()));
        for (std::size_t k = 0; k < rows.size(); ++k)
            EXPECT_EQ(rows[k], ref.rows[s][k]) << "s=" << s << " k=" << k;
        // Optional per-child scatter maps: empty spans when not stored.
        for (std::size_t k = 0; k < ch.size(); ++k) {
            const std::span<const Int> cpr = ak.child_parent_rows(s, static_cast<Int>(k));
            if (!ak.has_child_parent_rows()) EXPECT_TRUE(cpr.empty());
        }
    }
}

CscLower make_poisson3d(int N) {
    const int n = N * N * N;
    auto id = [N](int i, int j, int k) { return (k * N + j) * N + i; };
    CscLower A;
    A.n = n;
    A.col_ptr.assign(1, 0);
    for (int k = 0; k < N; ++k)
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                A.row_idx.push_back(id(i, j, k));
                A.values.push_back(6.0);
                if (i + 1 < N) { A.row_idx.push_back(id(i + 1, j, k)); A.values.push_back(-1.0); }
                if (j + 1 < N) { A.row_idx.push_back(id(i, j + 1, k)); A.values.push_back(-1.0); }
                if (k + 1 < N) { A.row_idx.push_back(id(i, j, k + 1)); A.values.push_back(-1.0); }
                A.col_ptr.push_back(static_cast<Int>(A.row_idx.size()));
            }
    return A;
}

bool load_fixture(const char *name, CscLower &A) {
    const fs::path mtx = fs::path(__FILE__).parent_path() / ".." / ".." / "benchmarks" /
                         "fixtures" / "optim20" / (std::string(name) + ".mtx");
    std::error_code ec;
    if (!fs::exists(mtx, ec)) return false;
    auto mm = read_matrix_market(mtx.string());
    if (!mm.ok) return false;
    A = std::move(mm.matrix);
    return true;
}

// Analyse, factor and solve A x = A*1; the backward error must pass the gate.
void factor_solve_within_gate(const CscLower &A, MatrixType type) {
    Control ctrl;
    ctrl.matrix_type = type;
    ctrl.num_threads = 1;
    Info info;
    Solver solver;
    auto ak = solver.analyse(A, ctrl, info);
    ASSERT_NE(ak, nullptr);
    ASSERT_EQ(ak->num_supernodes(), static_cast<Int>(ak->supernodes.size()));
    ASSERT_EQ(ak->child_ptr.size(), ak->supernodes.size() + 1);
    ASSERT_EQ(ak->front_row_ptr.size(), ak->supernodes.size() + 1);
    FactorKeep fk;
    ASSERT_EQ(solver.factor(*ak, ctrl, info, fk), FactorStatus::Success);
    const std::size_t n = static_cast<std::size_t>(A.n);
    std::vector<double> b(n, 0.0);
    for (Int j = 0; j < A.n; ++j)
        for (Int p = A.col_ptr[j]; p < A.col_ptr[j + 1]; ++p) {
            const Int i = A.row_idx[p];
            b[i] += A.values[p];
            if (i != j) b[j] += A.values[p];
        }
    std::vector<double> x = b;
    ASSERT_EQ(solver.solve(fk, ctrl, info, x.data(), A.n), 0);
    const bench::Validation v = bench::validate(A, b, x);
    EXPECT_TRUE(bench::is_qualified_validation(v))
        << "eta_inf=" << v.eta_inf << " omega=" << v.omega;
}

} // namespace

TEST(AssemblyTree, FlatStorageMatchesNested) {
    {
        CscLower A;
        ASSERT_TRUE(load_fixture("kkt_lqr_h20", A)) << "fixture kkt_lqr_h20 missing";
        Control ctrl;
        ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
        Info info;
        Solver solver;
        auto ak = solver.analyse(A, ctrl, info);
        ASSERT_NE(ak, nullptr);
        SCOPED_TRACE("kkt_lqr_h20");
        expect_flat_matches_nested(*ak);
    }
    {
        const CscLower A = make_poisson3d(8);
        Control ctrl;
        ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
        Info info;
        Solver solver;
        auto ak = solver.analyse(A, ctrl, info);
        ASSERT_NE(ak, nullptr);
        SCOPED_TRACE("poisson3d_8");
        expect_flat_matches_nested(*ak);
    }
}

// The child_parent_rows(s, k) accessor reads cpr_ptr/cpr_idx by child slot
// child_ptr[s] + k; without stored maps every span is empty.
TEST(AssemblyTree, ChildParentRowsAccessor) {
    AnalysisKeep ak;
    ak.child_ptr = {0, 0, 0, 2};
    ak.child_idx = {1, 0};
    EXPECT_FALSE(ak.has_child_parent_rows());
    EXPECT_TRUE(ak.child_parent_rows(2, 0).empty());
    ak.cpr_ptr = {0, 2, 3};
    ak.cpr_idx = {1, 2, 2};
    ASSERT_TRUE(ak.has_child_parent_rows());
    const auto a = ak.child_parent_rows(2, 0);
    const auto b = ak.child_parent_rows(2, 1);
    ASSERT_EQ(a.size(), 2u);
    EXPECT_EQ(a[0], 1);
    EXPECT_EQ(a[1], 2);
    ASSERT_EQ(b.size(), 1u);
    EXPECT_EQ(b[0], 2);
}

TEST(AssemblyTree, EdgeCases) {
    {
        SCOPED_TRACE("n = 1");
        CscLower A;
        A.n = 1;
        A.col_ptr = {0, 1};
        A.row_idx = {0};
        A.values = {4.0};
        factor_solve_within_gate(A, MatrixType::RealSymmetricPositiveDefinite);
        factor_solve_within_gate(A, MatrixType::RealSymmetricIndefinite);
    }
    {
        SCOPED_TRACE("diagonal n = 5 (5 roots)");
        CscLower A;
        A.n = 5;
        A.col_ptr = {0, 1, 2, 3, 4, 5};
        A.row_idx = {0, 1, 2, 3, 4};
        A.values = {1.0, 2.0, -3.0, 4.0, 5.0};
        Control ctrl;
        ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
        Info info;
        Solver solver;
        auto ak = solver.analyse(A, ctrl, info);
        ASSERT_NE(ak, nullptr);
        Int roots = 0;
        for (Int s = 0; s < ak->num_supernodes(); ++s) {
            roots += ak->supernodes[s].parent < 0 ? 1 : 0;
            EXPECT_TRUE(ak->children(s).empty());
            EXPECT_EQ(ak->front_size(s), ak->supernodes[s].width());
        }
        EXPECT_EQ(roots, ak->num_supernodes());
        factor_solve_within_gate(A, MatrixType::RealSymmetricIndefinite);
    }
    {
        SCOPED_TRACE("empty column");
        // [[1 1 0 0]; [1 0 0 0]; [0 0 1 1]; [0 0 1 2]]: column 1 stores nothing.
        CscLower A;
        A.n = 4;
        A.col_ptr = {0, 2, 2, 4, 5};
        A.row_idx = {0, 1, 2, 3, 3};
        A.values = {1.0, 1.0, 1.0, 1.0, 2.0};
        factor_solve_within_gate(A, MatrixType::RealSymmetricIndefinite);
    }
}

} // namespace smf
