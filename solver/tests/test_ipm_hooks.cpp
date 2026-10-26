// IPM hooks of the indefinite factorization.
//
// Control::max_negative_pivots: once more negative pivots than the limit
// have been accepted, the inertia is certainly wrong for an interior-point
// method, so the factorization stops early with FactorStatus::InertiaExceeded.
// When the limit is not exceeded the factor must be bit-identical to an
// unarmed run.

#include "../benchmarks/common/validation.hpp"
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/error.hpp"
#include "smf/info.hpp"
#include "smf/pivot_order.hpp"
#include "smf/solver.hpp"
#include "smf/types.hpp"
#include "smf/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

using namespace smf;

namespace {

// KKT [[H, J^T], [J, 0]] with n = 40 primal and m = 15 constraint rows.
// H = diag(-1, -1, -1, 2, ...): the three negative coordinates lie in
// null(J) because J has no entry in columns 0..2, and J has full row rank
// (row i alone touches column 3 + i). Inertia: (n - m - 3 + m, 3 + m, 0)
// = (37, 18, 0).
constexpr int kN = 40;
constexpr int kM = 15;

CscLower small_kkt() {
    std::vector<std::vector<std::pair<int, double>>> cols(kN + kM);
    for (int j = 0; j < kN; ++j)
        cols[static_cast<std::size_t>(j)].push_back({j, j < 3 ? -1.0 : 2.0 + 0.01 * j});
    for (int i = 0; i < kM; ++i) {
        cols[static_cast<std::size_t>(3 + i)].push_back({kN + i, 1.0 + 0.1 * i});
        cols[static_cast<std::size_t>(18 + i)].push_back({kN + i, 0.5});
        cols[static_cast<std::size_t>(33 + i % 7)].push_back({kN + i, 0.25});
    }
    CscLower A;
    A.n = kN + kM;
    A.col_ptr.push_back(0);
    for (auto &c : cols) {
        std::sort(c.begin(), c.end());
        for (const auto &[i, x] : c) {
            A.row_idx.push_back(i);
            A.values.push_back(x);
        }
        A.col_ptr.push_back(static_cast<Int>(A.row_idx.size()));
    }
    return A;
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

// Saddle point [L B^T; B 0] (copied from test_tree_parallel.cpp): L a 2D
// Laplacian on m x m, B picks sums of neighbouring pairs.
CscLower grid_kkt(int m, int stride) {
    const int nv = m * m;
    const int nc = nv / stride;
    std::vector<std::vector<std::pair<int, double>>> cols(static_cast<std::size_t>(nv + nc));
    const CscLower L = laplacian_2d(m);
    for (int j = 0; j < nv; ++j)
        for (Int p = L.col_ptr[static_cast<std::size_t>(j)]; p < L.col_ptr[static_cast<std::size_t>(j) + 1]; ++p)
            cols[static_cast<std::size_t>(j)].push_back({L.row_idx[static_cast<std::size_t>(p)], L.values[static_cast<std::size_t>(p)]});
    std::mt19937 rng(3);
    std::uniform_real_distribution<double> unif(0.5, 2.0);
    for (int r = 0; r < nc; ++r) {
        const int v = r * stride;
        cols[static_cast<std::size_t>(v)].push_back({nv + r, unif(rng)});
        if (v + 1 < nv) cols[static_cast<std::size_t>(v + 1)].push_back({nv + r, -unif(rng)});
    }
    CscLower A;
    A.n = nv + nc;
    A.col_ptr.push_back(0);
    for (auto &c : cols) {
        std::sort(c.begin(), c.end());
        for (const auto &[i, x] : c) {
            A.row_idx.push_back(i);
            A.values.push_back(x);
        }
        A.col_ptr.push_back(static_cast<Int>(A.row_idx.size()));
    }
    return A;
}

std::vector<double> rhs(Int n) {
    std::vector<double> b(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = std::sin(0.013 * static_cast<double>(i)) + 0.5;
    return b;
}

bool bit_equal(const std::vector<double> &a, const std::vector<double> &b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}

double max_rel_diff(const std::vector<double> &a, const std::vector<double> &b) {
    double d = 0.0, s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        d = std::max(d, std::abs(a[i] - b[i]));
        s = std::max(s, std::abs(a[i]));
    }
    return d / s;
}

// One analysis, factorizations on a shared NumericWorkspace / FactorKeep.
struct Harness {
    Solver solver;
    Control ctrl;
    CscLower A;
    std::unique_ptr<AnalysisKeep> ak;
    NumericWorkspace nws;
    FactorKeep fk;
    Info info;

    Harness(CscLower a, Control c) : ctrl(c), A(std::move(a)) {
        ak = solver.analyse(A, ctrl, info);
    }
    FactorStatus factor(int limit) {
        Control c = ctrl;
        c.max_negative_pivots = limit;
        info = Info{};
        return solver.factor(*ak, A, c, info, fk, nws);
    }
    // Solve with the current factor; returns the solve's return code.
    int solve(std::vector<double> &x) {
        x = rhs(A.n);
        SolveWorkspace sws;
        return solver.solve(fk, ctrl, info, x.data(), static_cast<int>(A.n), 1, sws);
    }
    double eta(const std::vector<double> &x) const {
        return smf::bench::validate(A, rhs(A.n), x, 1).eta_inf;
    }
};

Control indef_ctrl() {
    Control c;
    c.matrix_type = MatrixType::RealSymmetricIndefinite;
    return c;
}

Control grid_ctrl(int threads) {
    Control c;
    c.matrix_type = MatrixType::RealSymmetricIndefinite;
    c.ordering = OrderingMethod::AMD;
    c.scaling = ScalingMethod::None;
    c.num_threads = threads;
    return c;
}

} // namespace

TEST(IpmHooks, EnumValues) {
    EXPECT_EQ(static_cast<int>(FactorStatus::InertiaExceeded), 4);
    EXPECT_EQ(static_cast<int>(ErrorCode::InertiaExceeded), -9);
    EXPECT_EQ(Control{}.max_negative_pivots, -1);
    EXPECT_EQ(Info{}.aborted_flops_skipped, 0.0);
}

TEST(IpmHooks, FixtureHasExpectedInertia) {
    Harness h(small_kkt(), indef_ctrl());
    ASSERT_NE(h.ak.get(), nullptr);
    ASSERT_EQ(h.factor(-1), FactorStatus::Success);
    EXPECT_EQ(h.info.num_positive, 37);
    EXPECT_EQ(h.info.num_negative, 18);
    EXPECT_EQ(h.info.num_zero, 0);
    EXPECT_EQ(h.info.aborted_flops_skipped, 0.0);
}

TEST(IpmHooks, AbortsWhenNegativesExceedLimit) {
    Harness h(small_kkt(), indef_ctrl());
    ASSERT_NE(h.ak.get(), nullptr);
    EXPECT_EQ(h.factor(15), FactorStatus::InertiaExceeded);
    EXPECT_EQ(h.info.factor_status, FactorStatus::InertiaExceeded);
    EXPECT_EQ(h.info.status, ErrorCode::InertiaExceeded);
    EXPECT_EQ(h.fk.status, FactorStatus::InertiaExceeded);
    EXPECT_GT(h.info.num_negative, 15);
    EXPECT_GE(h.info.aborted_flops_skipped, 0.0);

    // The aborted factor is refused by the solve, even when singular
    // factors are allowed.
    std::vector<double> x;
    EXPECT_NE(h.solve(x), 0);
    EXPECT_EQ(h.info.status, ErrorCode::InertiaExceeded);
    h.ctrl.continue_on_singular = true;
    EXPECT_NE(h.solve(x), 0);
}

TEST(IpmHooks, AbortSkipsLaterFronts) {
    // Limit 0 stops at the first front with a negative pivot; the large
    // grid KKT has fronts after it, whose predicted flops are reported.
    Harness h(grid_kkt(40, 7), grid_ctrl(1));
    ASSERT_NE(h.ak.get(), nullptr);
    EXPECT_EQ(h.factor(0), FactorStatus::InertiaExceeded);
    EXPECT_GT(h.info.num_negative, 0);
    EXPECT_GT(h.info.aborted_flops_skipped, 0.0);
    EXPECT_LT(h.info.num_negative, 40 * 40 / 7);
}

TEST(IpmHooks, NoAbortAtExactLimit) {
    Harness h(small_kkt(), indef_ctrl());
    ASSERT_NE(h.ak.get(), nullptr);
    ASSERT_EQ(h.factor(18), FactorStatus::Success);
    EXPECT_EQ(h.info.num_positive, 37);
    EXPECT_EQ(h.info.num_negative, 18);
    EXPECT_EQ(h.info.num_zero, 0);
    EXPECT_EQ(h.info.aborted_flops_skipped, 0.0);
}

TEST(IpmHooks, ArmedButUntriggeredIsBitIdentical) {
    Harness off(small_kkt(), indef_ctrl());
    Harness on(small_kkt(), indef_ctrl());
    ASSERT_EQ(off.factor(-1), FactorStatus::Success);
    ASSERT_EQ(on.factor(1000), FactorStatus::Success);
    EXPECT_TRUE(bit_equal(off.fk.factor_values, on.fk.factor_values));
    std::vector<double> x_off, x_on;
    ASSERT_EQ(off.solve(x_off), 0);
    ASSERT_EQ(on.solve(x_on), 0);
    EXPECT_TRUE(bit_equal(x_off, x_on));
    EXPECT_LT(off.eta(x_off), 1e-12);
}

TEST(IpmHooks, IgnoredForSpd) {
    Control c;
    c.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
    Harness h(laplacian_2d(10), c);
    ASSERT_NE(h.ak.get(), nullptr);
    EXPECT_EQ(h.factor(0), FactorStatus::Success);
}

TEST(IpmHooks, FactorAgainAfterAbort) {
    // An abort must leave the workspace and FactorKeep fit for the next
    // factorization on them.
    Harness fresh(small_kkt(), indef_ctrl());
    ASSERT_EQ(fresh.factor(-1), FactorStatus::Success);
    Harness h(small_kkt(), indef_ctrl());
    ASSERT_EQ(h.factor(15), FactorStatus::InertiaExceeded);
    ASSERT_EQ(h.factor(-1), FactorStatus::Success);
    EXPECT_EQ(h.info.num_negative, 18);
    EXPECT_EQ(h.info.aborted_flops_skipped, 0.0);
    EXPECT_TRUE(bit_equal(fresh.fk.factor_values, h.fk.factor_values));
    std::vector<double> x;
    ASSERT_EQ(h.solve(x), 0);
    EXPECT_LT(h.eta(x), 1e-12);
}

TEST(IpmHooks, TreeParallelAbort) {
    // Unarmed tree-parallel baseline: proves the parallel path and gives
    // the true number of negatives.
    Harness base(grid_kkt(110, 7), grid_ctrl(4));
    ASSERT_NE(base.ak.get(), nullptr);
    ASSERT_EQ(base.factor(-1), FactorStatus::Success);
    ASSERT_GE(base.info.parallel_subtrees, 2) << "parallel path not taken";
    const int neg = base.info.num_negative;
    ASSERT_GT(neg, 0);
    std::vector<double> x_base;
    ASSERT_EQ(base.solve(x_base), 0);

    Harness h(grid_kkt(110, 7), grid_ctrl(4));
    ASSERT_NE(h.ak.get(), nullptr);

    // Repeated, so that the abort races the threads in several ways.
    for (int rep = 0; rep < 3; ++rep) {
        // One too few allowed: the last negative trips it.
        EXPECT_EQ(h.factor(neg - 1), FactorStatus::InertiaExceeded);
        EXPECT_GT(h.info.num_negative, neg - 1);
        EXPECT_GE(h.info.aborted_flops_skipped, 0.0);
        std::vector<double> x;
        EXPECT_NE(h.solve(x), 0);

        // Far too few: stops inside the subtrees and skips most of the work.
        EXPECT_EQ(h.factor(10), FactorStatus::InertiaExceeded);
        EXPECT_GT(h.info.num_negative, 10);
        EXPECT_LT(h.info.num_negative, neg);
        EXPECT_GT(h.info.aborted_flops_skipped, 0.0);
    }

    // Exactly enough, on the workspace of the aborted runs: the same
    // factorization as the unarmed run. The tree-parallel engine is not
    // bitwise reproducible from run to run even unarmed (rounding differs
    // at the 1e-14 level between two unarmed runs), so the comparison is
    // exact on the integer outcomes and to rounding on the solution; the
    // bitwise check is SerialGridArmedIsBitIdentical.
    ASSERT_EQ(h.factor(neg), FactorStatus::Success);
    EXPECT_EQ(h.info.num_negative, neg);
    EXPECT_EQ(h.info.num_positive, base.info.num_positive);
    EXPECT_EQ(h.info.num_zero, base.info.num_zero);
    EXPECT_EQ(h.info.delayed_pivots, base.info.delayed_pivots);
    EXPECT_EQ(h.info.parallel_subtrees, base.info.parallel_subtrees);
    EXPECT_EQ(h.info.aborted_flops_skipped, 0.0);
    EXPECT_EQ(h.fk.factor_values.size(), base.fk.factor_values.size());
    std::vector<double> x;
    ASSERT_EQ(h.solve(x), 0);
    EXPECT_LT(max_rel_diff(x_base, x), 1e-12);
    EXPECT_LT(h.eta(x), 1e-12);

    // And off again after an abort.
    EXPECT_EQ(h.factor(10), FactorStatus::InertiaExceeded);
    ASSERT_EQ(h.factor(-1), FactorStatus::Success);
    EXPECT_EQ(h.info.num_negative, neg);
    EXPECT_EQ(h.info.delayed_pivots, base.info.delayed_pivots);
    ASSERT_EQ(h.solve(x), 0);
    EXPECT_LT(max_rel_diff(x_base, x), 1e-12);
    EXPECT_LT(h.eta(x), 1e-12);
}

TEST(IpmHooks, SerialGridArmedIsBitIdentical) {
    // The serial engine on the grid KKT (many fronts, delays): armed at the
    // true count, or aborted first, the factor is bitwise the unarmed one.
    Harness base(grid_kkt(110, 7), grid_ctrl(1));
    ASSERT_EQ(base.factor(-1), FactorStatus::Success);
    const int neg = base.info.num_negative;
    std::vector<double> x_base;
    ASSERT_EQ(base.solve(x_base), 0);

    Harness h(grid_kkt(110, 7), grid_ctrl(1));
    EXPECT_EQ(h.factor(neg - 1), FactorStatus::InertiaExceeded);
    EXPECT_EQ(h.factor(10), FactorStatus::InertiaExceeded);
    ASSERT_EQ(h.factor(neg), FactorStatus::Success);
    EXPECT_EQ(h.info.delayed_pivots, base.info.delayed_pivots);
    EXPECT_TRUE(bit_equal(base.fk.factor_values, h.fk.factor_values));
    std::vector<double> x;
    ASSERT_EQ(h.solve(x), 0);
    EXPECT_TRUE(bit_equal(x_base, x));
}

// --- Control::user_perm / user pairs (OrderingMethod::UserSupplied) ---

namespace {

Control user_ctrl(const std::vector<Int> &perm) {
    Control c = indef_ctrl();
    c.ordering = OrderingMethod::UserSupplied;
    c.user_perm = perm.data();
    return c;
}

std::vector<Int> amd_perm(const CscLower &A) {
    Control c = indef_ctrl();
    c.ordering = OrderingMethod::AMD;
    Solver s;
    Info info;
    auto ak = s.analyse(A, c, info);
    return ak ? ak->perm : std::vector<Int>{};
}

// The AMD order of small_kkt with each constraint row kN + i moved to just
// after its primal column 3 + i (coupled by A(kN + i, 3 + i) = 1 + 0.1 i).
std::vector<Int> paired_order(std::vector<Int> &pairs) {
    const std::vector<Int> amd = amd_perm(small_kkt());
    std::vector<Int> perm;
    pairs.clear();
    for (Int old : amd) {
        if (old >= kN)
            continue;
        perm.push_back(old);
        if (old >= 3 && old < 3 + kM) {
            perm.push_back(kN + old - 3);
            pairs.push_back(old);
            pairs.push_back(kN + old - 3);
        }
    }
    return perm;
}

} // namespace

TEST(IpmHooks, UserPermMatchesAmdFactor) {
    // AMD's final order fed back in gives the same elimination tree and
    // fill. Relaxed amalgamation is greedy and sees the tree in a new
    // postorder, so its stored zeros can differ slightly: exact on the
    // small KKT, within 5% on the grid KKT (measured 6095 vs 6175).
    bool exact = true;
    for (const CscLower &A : {small_kkt(), grid_kkt(40, 7)}) {
        Control amd = indef_ctrl();
        amd.ordering = OrderingMethod::AMD;
        Harness ref(A, amd);
        ASSERT_NE(ref.ak.get(), nullptr);
        const std::vector<Int> perm = ref.ak->perm;
        const LongInt ref_entries = ref.info.predicted_factor_entries;

        Harness h(A, user_ctrl(perm));
        ASSERT_NE(h.ak.get(), nullptr);
        EXPECT_EQ(h.info.status, ErrorCode::Success);
        EXPECT_EQ(h.info.ordering_used, static_cast<int>(OrderingMethod::UserSupplied));
        if (exact)
            EXPECT_EQ(h.info.predicted_factor_entries, ref_entries);
        else
            EXPECT_LE(std::abs(static_cast<double>(h.info.predicted_factor_entries - ref_entries)),
                      0.05 * static_cast<double>(ref_entries));
        exact = false;
        EXPECT_TRUE(h.ak->matching_scale.empty());

        ASSERT_EQ(ref.factor(-1), FactorStatus::Success);
        ASSERT_EQ(h.factor(-1), FactorStatus::Success);
        std::vector<double> x_ref, x;
        ASSERT_EQ(ref.solve(x_ref), 0);
        ASSERT_EQ(h.solve(x), 0);
        EXPECT_LT(max_rel_diff(x_ref, x), 1e-12);
    }
}

TEST(IpmHooks, UserPermIsUsedAsGiven) {
    // A shuffled order fills far more than AMD: proves the perm is read,
    // not recomputed. The factor is still correct.
    const CscLower A = grid_kkt(40, 7);
    Control amd = indef_ctrl();
    amd.ordering = OrderingMethod::AMD;
    Harness ref(A, amd);
    ASSERT_NE(ref.ak.get(), nullptr);

    std::vector<Int> perm(static_cast<std::size_t>(A.n));
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), std::mt19937(11));
    Harness h(A, user_ctrl(perm));
    ASSERT_NE(h.ak.get(), nullptr);
    EXPECT_GT(h.info.predicted_factor_entries, 2 * ref.info.predicted_factor_entries);
    ASSERT_EQ(h.factor(-1), FactorStatus::Success);
    std::vector<double> x;
    ASSERT_EQ(h.solve(x), 0);
    EXPECT_LT(h.eta(x), 1e-12);
}

TEST(IpmHooks, PairsStayInOneSupernode) {
    std::vector<Int> pairs;
    const std::vector<Int> perm = paired_order(pairs);
    ASSERT_EQ(perm.size(), static_cast<std::size_t>(kN + kM));
    ASSERT_EQ(pairs.size(), static_cast<std::size_t>(2 * kM));
    // nemin 1 turns amalgamation off: only the pair rule can merge them.
    for (int nemin : {1, 8}) {
        Control c = user_ctrl(perm);
        c.nemin = nemin;
        c.user_pair_first = pairs.data();
        c.user_num_pairs = kM;
        Harness h(small_kkt(), c);
        ASSERT_NE(h.ak.get(), nullptr) << "nemin " << nemin;
        EXPECT_EQ(h.info.status, ErrorCode::Success);
        EXPECT_EQ(h.info.ordering_used, static_cast<int>(OrderingMethod::UserSupplied));
        EXPECT_EQ(h.info.pairs_split, 0);
        const auto &sn = h.ak->supernodes;
        for (int k = 0; k < kM; ++k) {
            const Int pa = h.ak->iperm[static_cast<std::size_t>(pairs[2 * k])];
            const Int pb = h.ak->iperm[static_cast<std::size_t>(pairs[2 * k + 1])];
            const auto in = [&](const Supernode &s, Int p) { return s.col_start <= p && p < s.col_end; };
            const auto it = std::find_if(sn.begin(), sn.end(), [&](const Supernode &s) { return in(s, pa); });
            ASSERT_NE(it, sn.end());
            EXPECT_TRUE(in(*it, pb)) << "nemin " << nemin << " pair " << k;
        }
        ASSERT_EQ(h.factor(-1), FactorStatus::Success);
        EXPECT_EQ(h.info.num_positive, 37);
        EXPECT_EQ(h.info.num_negative, 18);
        std::vector<double> x;
        ASSERT_EQ(h.solve(x), 0);
        EXPECT_LT(h.eta(x), 1e-12);
    }
}

TEST(IpmHooks, UncoupledPairIsSplit) {
    // Primal columns 0 and 1 are adjacent but share no entry, so 1 is not
    // the etree parent of 0 and the pair cannot be forced together.
    std::vector<Int> perm(static_cast<std::size_t>(kN + kM));
    std::iota(perm.begin(), perm.end(), 0);
    const std::vector<Int> pairs = {1, 0};
    Control c = user_ctrl(perm);
    c.nemin = 1;
    c.user_pair_first = pairs.data();
    c.user_num_pairs = 1;
    Harness h(small_kkt(), c);
    ASSERT_NE(h.ak.get(), nullptr);
    EXPECT_EQ(h.info.status, ErrorCode::Success);
    EXPECT_EQ(h.info.pairs_split, 1);
}

TEST(IpmHooks, UserPermNullIsError) {
    Control c = indef_ctrl();
    c.ordering = OrderingMethod::UserSupplied;
    Solver s;
    Info info;
    EXPECT_EQ(s.analyse(small_kkt(), c, info), nullptr);
    EXPECT_EQ(info.status, ErrorCode::IllegalValue);
}

TEST(IpmHooks, UserPermInvalidIsError) {
    const CscLower A = small_kkt();
    std::vector<Int> perm(static_cast<std::size_t>(A.n));
    std::iota(perm.begin(), perm.end(), 0);
    Solver s;
    Info info;

    std::vector<Int> dup = perm;
    dup[5] = 4;
    EXPECT_EQ(s.analyse(A, user_ctrl(dup), info), nullptr);
    EXPECT_EQ(info.status, ErrorCode::IllegalValue);

    std::vector<Int> out = perm;
    out[5] = A.n;
    EXPECT_EQ(s.analyse(A, user_ctrl(out), info), nullptr);
    EXPECT_EQ(info.status, ErrorCode::IllegalValue);

    // A pair must be adjacent in the order.
    const std::vector<Int> far = {3, kN};
    Control c = user_ctrl(perm);
    c.user_pair_first = far.data();
    c.user_num_pairs = 1;
    EXPECT_EQ(s.analyse(A, c, info), nullptr);
    EXPECT_EQ(info.status, ErrorCode::IllegalValue);

    // Pairs without their array.
    c.user_pair_first = nullptr;
    EXPECT_EQ(s.analyse(A, c, info), nullptr);
    EXPECT_EQ(info.status, ErrorCode::IllegalValue);
}

TEST(IpmHooks, TaskParallelEngineReportsSameStatus) {
    // Control::tree_parallel selects the older task-parallel engine (when
    // built with SMF_PARALLEL): no early stop there, but the limit is
    // checked on its finished factor so the status is the same.
    Control c = indef_ctrl();
    c.num_threads = 4;
    c.tree_parallel = true;
    Harness h(small_kkt(), c);
    ASSERT_NE(h.ak.get(), nullptr);
    EXPECT_EQ(h.factor(15), FactorStatus::InertiaExceeded);
    EXPECT_GT(h.info.num_negative, 15);
    ASSERT_EQ(h.factor(18), FactorStatus::Success);
    EXPECT_EQ(h.info.num_negative, 18);
}

// --- realized_pivot_order ---

namespace {

bool is_permutation_of_n(const std::vector<Int> &p, Int n) {
    std::vector<char> seen(static_cast<std::size_t>(n), 0);
    if (p.size() != static_cast<std::size_t>(n)) return false;
    for (Int v : p) {
        if (v < 0 || v >= n || seen[static_cast<std::size_t>(v)]) return false;
        seen[static_cast<std::size_t>(v)] = 1;
    }
    return true;
}

// Delay-prone setting probed for ReanalysisRemovesDelays.
Control delay_ctrl(const std::vector<Int> *perm = nullptr) {
    Control c = grid_ctrl(1);
    c.nemin = 1;
    c.pivot_u = 0.5;
    if (perm) {
        c.ordering = OrderingMethod::UserSupplied;
        c.user_perm = perm->data();
    }
    return c;
}

} // namespace

TEST(IpmHooks, RealizedOrderIsPermutation) {
    // CrossFrontNonsingular (test_delayed_pivots.cpp): A(0,0) = 0, so column
    // 0 is delayed from its leaf front into the parent and eliminated later.
    CscLower A;
    A.n = 3;
    A.col_ptr = {0, 2, 4, 5};
    A.row_idx = {0, 2, 1, 2, 2};
    A.values = {0.0, 1.0, 4.0, 1.0, 2.0};
    Control c = indef_ctrl();
    c.nemin = 1;
    c.pivot_u = 0.01;
    Harness h(A, c);
    ASSERT_NE(h.ak.get(), nullptr);
    ASSERT_EQ(h.factor(-1), FactorStatus::Success);
    ASSERT_GT(h.info.delayed_pivots, 0);

    const RealizedOrder ro = realized_pivot_order(*h.ak, h.fk);
    EXPECT_TRUE(is_permutation_of_n(ro.perm, 3));
    EXPECT_NE(ro.perm[0], 0);
    EXPECT_EQ(ro.zero_pivots, 0);
    // Every 2x2 pair is adjacent in perm.
    for (std::size_t k = 0; k < ro.pairs.size(); k += 2) {
        const auto it = std::find(ro.perm.begin(), ro.perm.end(), ro.pairs[k]);
        ASSERT_NE(it, ro.perm.end());
        ASSERT_NE(it + 1, ro.perm.end());
        EXPECT_EQ(*(it + 1), ro.pairs[k + 1]);
    }
}

TEST(IpmHooks, RealizedOrderPutsZeroPivotsLast) {
    // diag(0, 1, 2): column 0 is a root with a zero diagonal and no coupling.
    CscLower A;
    A.n = 3;
    A.col_ptr = {0, 1, 2, 3};
    A.row_idx = {0, 1, 2};
    A.values = {0.0, 1.0, 2.0};
    Control c = indef_ctrl();
    c.nemin = 1;
    c.scaling = ScalingMethod::None;
    Harness h(A, c);
    ASSERT_NE(h.ak.get(), nullptr);
    h.factor(-1);
    ASSERT_EQ(h.info.num_zero, 1);
    const RealizedOrder ro = realized_pivot_order(*h.ak, h.fk);
    ASSERT_TRUE(is_permutation_of_n(ro.perm, 3));
    EXPECT_EQ(ro.zero_pivots, 1);
    EXPECT_EQ(ro.perm[2], 0);
    EXPECT_TRUE(ro.pairs.empty());
}

namespace {

// Factor A with delays, re-analyse from its realized order, factor again.
// Returns the delays left; fills the first run's counts.
struct ReanalysisResult {
    int first_delays = 0, second_delays = 0, pairs = 0, parallel_subtrees = 0;
};

ReanalysisResult reanalyse_and_check(const CscLower &A, int threads) {
    ReanalysisResult r;
    Control c = delay_ctrl();
    c.num_threads = threads;
    Harness first(A, c);
    EXPECT_NE(first.ak.get(), nullptr);
    if (!first.ak) return r;
    EXPECT_EQ(first.factor(-1), FactorStatus::Success);
    r.first_delays = first.info.delayed_pivots;
    r.parallel_subtrees = first.info.parallel_subtrees;
    std::vector<double> x1;
    EXPECT_EQ(first.solve(x1), 0);

    const RealizedOrder ro = realized_pivot_order(*first.ak, first.fk);
    EXPECT_TRUE(is_permutation_of_n(ro.perm, A.n));
    if (ro.perm.empty()) return r;
    r.pairs = static_cast<int>(ro.pairs.size() / 2);

    Control c2 = delay_ctrl(&ro.perm);
    c2.num_threads = threads;
    c2.user_pair_first = ro.pairs.empty() ? nullptr : ro.pairs.data();
    c2.user_num_pairs = static_cast<Int>(ro.pairs.size() / 2);
    Harness second(A, c2);
    EXPECT_NE(second.ak.get(), nullptr);
    if (!second.ak) return r;
    EXPECT_EQ(second.info.pairs_split, 0);
    EXPECT_EQ(second.factor(-1), FactorStatus::Success);
    r.second_delays = second.info.delayed_pivots;
    std::vector<double> x2;
    EXPECT_EQ(second.solve(x2), 0);
    EXPECT_LT(max_rel_diff(x1, x2), 1e-10);
    EXPECT_LT(second.eta(x2), 1e-10);
    return r;
}

} // namespace

TEST(IpmHooks, ReanalysisRemovesDelays) {
    // grid_kkt(40, 7) with u = 0.5, nemin = 1 delays heavily (the Task 6 KKT
    // does not: its AMD order already puts each constraint after its primal).
    const ReanalysisResult serial = reanalyse_and_check(grid_kkt(40, 7), 1);
    EXPECT_GT(serial.first_delays, 0);
    EXPECT_EQ(serial.second_delays, 0);
}

TEST(IpmHooks, ReanalysisRemovesDelaysTreeParallel) {
    // Delayed columns that cross the subtree/top boundary of the tree-parallel
    // engine: diag_entries is the concatenation of passes.
    const ReanalysisResult par = reanalyse_and_check(grid_kkt(110, 7), 4);
    ASSERT_GE(par.parallel_subtrees, 2) << "parallel path not taken";
    EXPECT_GT(par.first_delays, 0);
    EXPECT_EQ(par.second_delays, 0);
}

namespace {
// Zero-diagonal grid adjacency (bipartite, values 1 + small variation): all
// pivots must be 2x2 or delayed.
CscLower zero_diag_grid(int m) {
    CscLower A;
    A.n = m * m;
    A.col_ptr.push_back(0);
    for (int j = 0; j < A.n; ++j) {
        A.row_idx.push_back(j);
        A.values.push_back(j % 11 == 0 ? 0.3 : 0.0);
        if ((j % m) + 1 < m) { A.row_idx.push_back(j + 1); A.values.push_back(1.0 + 0.01 * (j % 7)); }
        if (j + m < A.n) { A.row_idx.push_back(j + m); A.values.push_back(-1.0 - 0.02 * (j % 5)); }
        A.col_ptr.push_back(static_cast<Int>(A.row_idx.size()));
    }
    return A;
}
}

TEST(IpmHooks, ReanalysisKeepsTwoByTwoPivots) {
    // Zero-diagonal matrix: pivots are 2x2 blocks (and delays). The realized
    // order's pairs are passed back, none is split, and no delay remains.
    for (int threads : {1, 4}) {
        const ReanalysisResult r = reanalyse_and_check(zero_diag_grid(30), threads);
        EXPECT_GT(r.pairs, 0) << "threads " << threads;
        EXPECT_GT(r.first_delays, 0) << "threads " << threads;
        EXPECT_EQ(r.second_delays, 0) << "threads " << threads;
    }
}
