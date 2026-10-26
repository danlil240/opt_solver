// Feature 3b: IpmSession in StaticMode::Regularized (the quasidefinite mode).
//
// The session adds any missing diagonal entries to the pattern, classifies
// rows at the first factorization of an analysis (zero or missing diagonal:
// constraint, -rho; anything else, negative included: primal, +rho),
// factors in AMD order with 1x1 pivots and no delays (a pivot below rho is
// raised and counted as a static pivot), and refines every full solve
// against the unregularized matrix.

#include "../benchmarks/common/validation.hpp"
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/error.hpp"
#include "smf/info.hpp"
#include "smf/ipm_session.hpp"
#include "smf/matrix_market.hpp"
#include "smf/solver.hpp"
#include "smf/types.hpp"
#include "smf/workspace.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace smf;

namespace {

namespace fs = std::filesystem;

// A recorded IPOPT KKT sequence (as in test_ipm_session.cpp).
struct Replay {
    CscLower A;
    std::vector<std::vector<double>> steps;
};

bool load_replay(const std::string &model, Replay &r) {
    const fs::path dir = fs::path(__FILE__).parent_path() / ".." / ".." / "benchmarks" / "fixtures" /
                         "ipopt_kkt" / model;
    const fs::path mtx = dir / (model + ".mtx");
    const fs::path vals = dir / (model + ".mtx.vals");
    std::error_code ec;
    if (!fs::exists(mtx, ec) || !fs::exists(vals, ec)) return false;
    auto mm = read_matrix_market(mtx.string());
    if (!mm.ok) return false;
    r.A = std::move(mm.matrix);
    const std::size_t step_bytes = r.A.values.size() * sizeof(double);
    std::ifstream in(vals, std::ios::binary | std::ios::ate);
    if (!in.good() || step_bytes == 0) return false;
    const std::size_t nsteps = static_cast<std::size_t>(in.tellg()) / step_bytes;
    in.seekg(0);
    r.steps.assign(nsteps, std::vector<double>(r.A.values.size()));
    for (auto &s : r.steps) {
        in.read(reinterpret_cast<char *>(s.data()), static_cast<std::streamsize>(step_bytes));
        if (!in.good()) return false;
    }
    return !r.steps.empty();
}

// The MA97 plugin's control on the IPOPT track.
Control ipopt_control() {
    Control c;
    c.matrix_type = MatrixType::RealSymmetricIndefinite;
    c.ordering = OrderingMethod::AutoParallel;
    c.scaling = ScalingMethod::None;
    c.pivot_u = 1e-8;
    c.num_threads = 1;
    c.continue_on_singular = true;
    c.expected_refactors = 10;
    c.solve_use_blas3_single_rhs = true;
    c.solve_multifrontal_forward = true;
    return c;
}

IpmControl reg_mode(bool as_zero = false) {
    IpmControl ipm;
    ipm.learn_inertia = false;
    ipm.abort_on_wrong_inertia = false;
    ipm.reanalyse = false;
    ipm.static_mode = StaticMode::Regularized;
    ipm.static_as_zero = as_zero;
    return ipm;
}

IpmControl off_mode() {
    IpmControl ipm = reg_mode();
    ipm.static_mode = StaticMode::Off;
    return ipm;
}

std::vector<double> rhs(Int n) {
    std::vector<double> b(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = std::sin(0.013 * static_cast<double>(i)) + 0.5;
    return b;
}

// Lower CSC from (row, col, value) triplets, row >= col; the order of
// triplets within a column is kept (duplicates allowed).
CscLower from_triplets(Int n, std::vector<std::tuple<Int, Int, double>> t) {
    std::stable_sort(t.begin(), t.end(), [](const auto &x, const auto &y) {
        return std::get<1>(x) != std::get<1>(y) ? std::get<1>(x) < std::get<1>(y)
                                                : std::get<0>(x) < std::get<0>(y);
    });
    CscLower A;
    A.n = n;
    A.col_ptr.assign(static_cast<std::size_t>(n) + 1, 0);
    for (const auto &[i, j, v] : t) {
        ++A.col_ptr[static_cast<std::size_t>(j) + 1];
        A.row_idx.push_back(i);
        A.values.push_back(v);
    }
    for (std::size_t j = 0; j < static_cast<std::size_t>(n); ++j) A.col_ptr[j + 1] += A.col_ptr[j];
    return A;
}

std::vector<std::tuple<Int, Int, double>> triplets(const CscLower &A) {
    std::vector<std::tuple<Int, Int, double>> t;
    for (Int j = 0; j < A.n; ++j)
        for (Int p = A.col_ptr[static_cast<std::size_t>(j)]; p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p)
            t.emplace_back(A.row_idx[static_cast<std::size_t>(p)], j, A.values[static_cast<std::size_t>(p)]);
    return t;
}

// KKT [[H, J^T], [J, 0]] with kN = 40 primal and kM = 15 constraint rows
// (the Task 6 fixture of test_ipm_hooks.cpp), stored without the
// constraint-block diagonal. J has no entry in columns 0..2 and full row rank
// (row i alone touches column 3 + i).
constexpr int kN = 40;
constexpr int kM = 15;

// `neg_cols`: the H diagonal entries set to -1 (the others are 2 + 0.01 j).
CscLower kkt(std::initializer_list<int> neg_cols) {
    std::vector<std::tuple<Int, Int, double>> t;
    for (int j = 0; j < kN; ++j) {
        const bool neg = std::find(neg_cols.begin(), neg_cols.end(), j) != neg_cols.end();
        t.emplace_back(j, j, neg ? -1.0 : 2.0 + 0.01 * j);
    }
    for (int i = 0; i < kM; ++i) {
        t.emplace_back(kN + i, 3 + i, 1.0 + 0.1 * i);
        t.emplace_back(kN + i, 18 + i, 0.5);
        t.emplace_back(kN + i, 33 + i % 7, 0.25);
    }
    return from_triplets(kN + kM, std::move(t));
}

// H positive definite on null(J) but indefinite overall: the -1 entries sit
// on columns 3, 4, 5, which J couples (with weight 1.x) to columns of H ~ 2,
// so x^T H x > 0 for J x = 0. Inertia (40, 15, 0).
CscLower good_kkt() { return kkt({3, 4, 5}); }

// The Task 6 KKT: H = -1 on columns 0..2, which lie in null(J). The reduced
// Hessian is indefinite; inertia (37, 18, 0).
CscLower bad_kkt() { return kkt({0, 1, 2}); }

// A with an explicit zero diagonal entry in every column that has none.
CscLower with_zero_diagonal(const CscLower &A) {
    auto t = triplets(A);
    for (Int j = 0; j < A.n; ++j) {
        bool has = false;
        for (Int p = A.col_ptr[static_cast<std::size_t>(j)]; p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p)
            has = has || A.row_idx[static_cast<std::size_t>(p)] == j;
        if (!has) t.emplace_back(j, j, 0.0);
    }
    return from_triplets(A.n, std::move(t));
}

// A with some entries stored as two duplicates that sum to the original
// value: diagonal (3, 3), off-diagonal (kN, 3), and the explicit zero
// diagonal of row kN + 1 when there is one.
CscLower with_duplicates(const CscLower &A) {
    std::vector<std::tuple<Int, Int, double>> t;
    for (const auto &[i, j, v] : triplets(A)) {
        const bool split = (i == 3 && j == 3) || (i == kN && j == 3) || (i == kN + 1 && j == kN + 1);
        if (split) {
            t.emplace_back(i, j, 0.25 * v);
            t.emplace_back(i, j, 0.75 * v);
        } else {
            t.emplace_back(i, j, v);
        }
    }
    return from_triplets(A.n, std::move(t));
}

// diag(1, d, 0.5): max|A| = 1, so rho = 1e-8 exactly (no scaling).
CscLower tiny_diag(double d) { return from_triplets(3, {{0, 0, 1.0}, {1, 1, d}, {2, 2, 0.5}}); }

struct Run {
    IpmFactorStatus status = IpmFactorStatus::Error;
    InertiaCounts in{};
    int delayed = -1;
    int statics = -1;
    long stats_static = -1;
    double eta = 0.0;
    bool qualified = false;
};

// Analyse A's pattern in Regularized mode, factor A's values (or `values`),
// and solve with rhs(n); the solution is validated against the
// unregularized matrix A (with those values).
Run run_regularized(const CscLower &A, const IpmControl &ipm = reg_mode(),
                    const std::vector<double> *values = nullptr) {
    Run r;
    IpmSession s;
    EXPECT_EQ(s.analyse(A, {}, ipopt_control(), ipm), ErrorCode::Success);
    const std::vector<double> &v = values ? *values : A.values;
    r.status = s.factor(v);
    r.in = s.inertia();
    r.delayed = s.info().delayed_pivots;
    r.statics = s.info().static_pivots;
    r.stats_static = s.stats().static_pivots;
    if (r.status == IpmFactorStatus::Ok || r.status == IpmFactorStatus::Singular) {
        std::vector<double> x = rhs(A.n);
        EXPECT_EQ(s.solve(x.data(), static_cast<int>(A.n), 1), 0);
        CscLower Av = A;
        Av.values = v;
        const auto val = smf::bench::validate(Av, rhs(A.n), x, 1);
        r.eta = val.eta_inf;
        r.qualified = smf::bench::is_qualified_validation(val);
    }
    return r;
}

// The Regularized factorization of A run directly through Solver, with an
// explicit row classification `sign` (+1 primal, -1 constraint).
struct Direct {
    FactorStatus status;
    InertiaCounts in;
    int statics;
};

Direct run_explicit(const CscLower &A, const std::vector<int8_t> &sign, bool as_zero = false) {
    Control c = ipopt_control();
    c.ordering = OrderingMethod::AMD;
    c.pivot_u = 0.0;
    c.static_pivot_tol = 1e-8;
    c.static_pivot_in_order = true;
    c.static_zero_stays_zero = true;
    c.static_pivots_as_zero = as_zero;
    c.reg_diag_sign = sign.data();
    c.reg_diag_tol = 1e-8;
    Solver solver;
    Info info;
    auto ak = solver.analyse(A, c, info);
    EXPECT_TRUE(ak != nullptr);
    if (!ak) return {FactorStatus::Singular, {}, -1};
    FactorKeep fk;
    NumericWorkspace nws;
    const FactorStatus st = solver.factor(*ak, A, c, info, fk, nws);
    EXPECT_EQ(info.delayed_pivots, 0);
    return {st, {info.num_positive, info.num_negative, info.num_zero}, info.static_pivots};
}

std::string str(const InertiaCounts &c) {
    return "(" + std::to_string(c.positive) + ", " + std::to_string(c.negative) + ", " +
           std::to_string(c.zero) + ")";
}

bool same(const InertiaCounts &a, const InertiaCounts &b) {
    return a.positive == b.positive && a.negative == b.negative && a.zero == b.zero;
}

} // namespace

TEST(IpmRegularized, FixturesHaveExpectedInertia) {
    // With every feature off (threshold pivoting): the true inertia.
    const std::pair<CscLower, InertiaCounts> cases[] = {
        {with_zero_diagonal(good_kkt()), {kN, kM, 0}}, {with_zero_diagonal(bad_kkt()), {37, 18, 0}}};
    for (const auto &[A, want] : cases) {
        IpmSession s;
        ASSERT_EQ(s.analyse(A, {}, ipopt_control(), off_mode()), ErrorCode::Success);
        ASSERT_EQ(s.factor(A.values), IpmFactorStatus::Ok);
        EXPECT_TRUE(same(s.inertia(), want)) << str(s.inertia());
    }
}

TEST(IpmRegularized, InertiaMatchesReducedHessian) {
    const Run r = run_regularized(with_zero_diagonal(good_kkt()));
    EXPECT_EQ(r.status, IpmFactorStatus::Ok);
    EXPECT_EQ(r.in.negative, kM) << str(r.in);
    EXPECT_EQ(r.in.zero, 0) << str(r.in);
    EXPECT_EQ(r.in.positive, kN) << str(r.in);
    EXPECT_EQ(r.delayed, 0);
}

TEST(IpmRegularized, WrongInertiaWhenReducedHessianIndefinite) {
    const Run r = run_regularized(with_zero_diagonal(bad_kkt()));
    EXPECT_EQ(r.status, IpmFactorStatus::Ok);
    EXPECT_GT(r.in.negative, kM) << str(r.in);
    EXPECT_EQ(r.in.positive + r.in.negative + r.in.zero, kN + kM);
    EXPECT_EQ(r.delayed, 0);
}

TEST(IpmRegularized, RefinedSolveMeetsGate) {
    for (const CscLower &A : {with_zero_diagonal(good_kkt()), good_kkt(), with_zero_diagonal(bad_kkt())}) {
        const Run r = run_regularized(A);
        EXPECT_TRUE(r.qualified) << "eta " << r.eta;
        EXPECT_EQ(r.delayed, 0);
    }

    // The recorded IPOPT steps, scaled as IPOPT runs them (MC64 scaling,
    // MA97 scaling = 1, smf's matching scaling): rho = reg_rho x max|A|
    // after scaling. (Unscaled, the barrier terms of the later steps reach
    // max|A| = 3e11, so rho ~ 3e3 and refinement cannot recover the
    // componentwise gate.)
    Replay rp;
    if (!load_replay("acopf_grid_80", rp)) GTEST_SKIP() << "acopf_grid_80 fixture not present";
    ASSERT_GE(rp.steps.size(), 2u);
    Control ctrl = ipopt_control();
    ctrl.scaling = ScalingMethod::Matching;
    IpmSession s;
    ASSERT_EQ(s.analyse(rp.A, {}, ctrl, reg_mode()), ErrorCode::Success);
    for (std::size_t k = 0; k < rp.steps.size(); ++k) {
        const IpmFactorStatus st = s.factor(rp.steps[k]);
        ASSERT_TRUE(st == IpmFactorStatus::Ok || st == IpmFactorStatus::Singular) << "step " << k;
        EXPECT_EQ(s.info().delayed_pivots, 0) << "step " << k;
        std::vector<double> x = rhs(rp.A.n);
        ASSERT_EQ(s.solve(x.data(), static_cast<int>(rp.A.n), 1), 0) << "step " << k;
        CscLower Av = rp.A;
        Av.values = rp.steps[k];
        const auto val = smf::bench::validate(Av, rhs(rp.A.n), x, 1);
        EXPECT_TRUE(smf::bench::is_qualified_validation(val))
            << "step " << k << ": eta " << val.eta_inf << " omega " << val.omega
            << " static " << s.info().static_pivots << " inertia " << str(s.inertia());
    }
}

TEST(IpmRegularized, MissingDiagonalAugmented) {
    // The constraint block stored without its diagonal (good_kkt as built)
    // and with explicit zeros: same classification, inertia and solution.
    const CscLower missing = good_kkt();
    const CscLower zeros = with_zero_diagonal(missing);
    ASSERT_LT(missing.row_idx.size(), zeros.row_idx.size());

    IpmSession s;
    ASSERT_EQ(s.analyse(missing, missing.values, ipopt_control(), reg_mode()), ErrorCode::Success);
    ASSERT_EQ(s.factor(missing.values), IpmFactorStatus::Ok);
    const Run z = run_regularized(zeros);
    EXPECT_TRUE(same(s.inertia(), z.in)) << str(s.inertia()) << " vs " << str(z.in);
    EXPECT_EQ(s.info().delayed_pivots, 0);
    for (Int i = 0; i < missing.n; ++i) EXPECT_EQ(s.row_is_constraint(i), i >= kN) << "row " << i;
    std::vector<double> x = rhs(missing.n);
    ASSERT_EQ(s.solve(x.data(), static_cast<int>(missing.n), 1), 0);
    EXPECT_TRUE(smf::bench::is_qualified_validation(smf::bench::validate(missing, rhs(missing.n), x, 1)));

    // Duplicate entries map to their own slots and sum as before, with and
    // without the missing diagonals.
    for (const CscLower &D : {with_duplicates(missing), with_duplicates(zeros)}) {
        ASSERT_GT(D.row_idx.size(), missing.row_idx.size());
        IpmSession d;
        ASSERT_EQ(d.analyse(D, {}, ipopt_control(), reg_mode()), ErrorCode::Success);
        ASSERT_EQ(d.factor(D.values), IpmFactorStatus::Ok);
        EXPECT_TRUE(same(d.inertia(), z.in)) << str(d.inertia());
        EXPECT_EQ(d.info().delayed_pivots, 0);
        for (Int i = 0; i < D.n; ++i) EXPECT_EQ(d.row_is_constraint(i), i >= kN) << "row " << i;
        std::vector<double> y = rhs(D.n);
        ASSERT_EQ(d.solve(y.data(), static_cast<int>(D.n), 1), 0);
        EXPECT_TRUE(smf::bench::is_qualified_validation(smf::bench::validate(zeros, rhs(D.n), y, 1)));
    }
}

TEST(IpmRegularized, NegativePrimalDiagonalIsPrimal) {
    // good_kkt has H = -1 on rows 3, 4, 5: primal rows (+rho).
    const CscLower A = with_zero_diagonal(good_kkt());
    IpmSession s;
    for (Int i = 0; i < A.n; ++i) EXPECT_FALSE(s.row_is_constraint(i));
    ASSERT_EQ(s.analyse(A, {}, ipopt_control(), reg_mode()), ErrorCode::Success);
    for (Int i = 0; i < A.n; ++i) EXPECT_FALSE(s.row_is_constraint(i)) << "before the first factor";
    ASSERT_EQ(s.factor(A.values), IpmFactorStatus::Ok);
    for (Int i = 0; i < A.n; ++i) EXPECT_EQ(s.row_is_constraint(i), i >= kN) << "row " << i;
    std::vector<int8_t> sign(static_cast<std::size_t>(A.n), 1);
    for (Int i = kN; i < A.n; ++i) sign[static_cast<std::size_t>(i)] = -1;
    const Direct d = run_explicit(A, sign);
    EXPECT_TRUE(same(s.inertia(), d.in)) << str(s.inertia()) << " vs " << str(d.in);

    // Discriminating case: a primal diagonal of -rho / 2 gets +rho
    // (d = +rho / 2, raised to +rho and counted by its sign: positive);
    // with -rho it would be -1.5 rho, a negative pivot.
    const CscLower T = tiny_diag(-0.5e-8);
    const Run r = run_regularized(T);
    EXPECT_TRUE(same(r.in, InertiaCounts{3, 0, 0})) << str(r.in);
    const Direct as_primal = run_explicit(T, {1, 1, 1});
    const Direct as_constraint = run_explicit(T, {1, -1, 1});
    EXPECT_TRUE(same(r.in, as_primal.in)) << str(as_primal.in);
    EXPECT_TRUE(same(as_constraint.in, InertiaCounts{2, 1, 0})) << str(as_constraint.in);
    EXPECT_EQ(r.delayed, 0);
}

TEST(IpmRegularized, ForcedPivotAccounting) {
    // Exact zero: -rho + rho = 0, raised to +rho. A zero eigenvalue under
    // both accountings; the factor is singular.
    for (const bool as_zero : {false, true}) {
        const Run r = run_regularized(tiny_diag(-1e-8), reg_mode(as_zero));
        EXPECT_EQ(r.status, IpmFactorStatus::Singular) << "as_zero " << as_zero;
        EXPECT_TRUE(same(r.in, InertiaCounts{2, 0, 1})) << "as_zero " << as_zero << " " << str(r.in);
        EXPECT_EQ(r.statics, 1) << "as_zero " << as_zero;
        EXPECT_EQ(r.stats_static, 1) << "as_zero " << as_zero;
        EXPECT_EQ(r.delayed, 0);
    }
    // Nonzero tiny: -rho / 2 + rho = rho / 2, raised to +rho. Its sign by
    // default (hybrid), a zero eigenvalue only with static_as_zero.
    const Run hybrid = run_regularized(tiny_diag(-0.5e-8), reg_mode(false));
    EXPECT_EQ(hybrid.status, IpmFactorStatus::Ok);
    EXPECT_TRUE(same(hybrid.in, InertiaCounts{3, 0, 0})) << str(hybrid.in);
    EXPECT_EQ(hybrid.statics, 1);
    const Run zero = run_regularized(tiny_diag(-0.5e-8), reg_mode(true));
    EXPECT_EQ(zero.status, IpmFactorStatus::Singular);
    EXPECT_TRUE(same(zero.in, InertiaCounts{2, 0, 1})) << str(zero.in);
    EXPECT_EQ(zero.statics, 1);
}

TEST(IpmRegularized, StaticPivotsAccumulate) {
    const CscLower T = tiny_diag(-0.5e-8);
    IpmSession s;
    ASSERT_EQ(s.analyse(T, {}, ipopt_control(), reg_mode()), ErrorCode::Success);
    for (int k = 1; k <= 3; ++k) {
        ASSERT_EQ(s.factor(T.values), IpmFactorStatus::Ok);
        EXPECT_EQ(s.stats().static_pivots, k);
    }
}

TEST(IpmRegularized, ClassificationFrozenUntilAnalyse) {
    // First factor: constraint diagonals 0 -> constraint rows. Second
    // factor: delta_c = 1e-4 on the constraint block and H(0, 0) = 0; the
    // classification does not change. A new analysis classifies again.
    const CscLower A = with_zero_diagonal(good_kkt());
    CscLower B = A;
    for (Int j = 0; j < B.n; ++j)
        for (Int p = B.col_ptr[static_cast<std::size_t>(j)]; p < B.col_ptr[static_cast<std::size_t>(j) + 1]; ++p)
            if (B.row_idx[static_cast<std::size_t>(p)] == j) {
                if (j >= kN) B.values[static_cast<std::size_t>(p)] = -1e-4;
                if (j == 0) B.values[static_cast<std::size_t>(p)] = 0.0;
            }

    IpmSession s;
    ASSERT_EQ(s.analyse(A, {}, ipopt_control(), reg_mode()), ErrorCode::Success);
    ASSERT_EQ(s.factor(A.values), IpmFactorStatus::Ok);
    EXPECT_FALSE(s.row_is_constraint(0));
    EXPECT_TRUE(s.row_is_constraint(kN));
    s.factor(B.values);
    EXPECT_EQ(s.info().delayed_pivots, 0);
    EXPECT_FALSE(s.row_is_constraint(0));
    EXPECT_TRUE(s.row_is_constraint(kN));

    ASSERT_EQ(s.analyse(B, {}, ipopt_control(), reg_mode()), ErrorCode::Success);
    EXPECT_FALSE(s.row_is_constraint(kN)) << "reset by analyse()";
    s.factor(B.values);
    EXPECT_EQ(s.info().delayed_pivots, 0);
    EXPECT_TRUE(s.row_is_constraint(0));
    EXPECT_FALSE(s.row_is_constraint(kN));
}

TEST(IpmRegularized, NoDelays) {
    const CscLower cases[] = {with_zero_diagonal(good_kkt()), good_kkt(), with_zero_diagonal(bad_kkt()),
                              bad_kkt(), with_duplicates(good_kkt()), tiny_diag(-1e-8),
                              tiny_diag(-0.5e-8), tiny_diag(-1.0)};
    for (const CscLower &A : cases) {
        for (const bool as_zero : {false, true}) {
            const Run r = run_regularized(A, reg_mode(as_zero));
            EXPECT_EQ(r.delayed, 0) << "n " << A.n << " as_zero " << as_zero;
            EXPECT_TRUE(r.status == IpmFactorStatus::Ok || r.status == IpmFactorStatus::Singular);
        }
    }
    Replay rp;
    if (!load_replay("acopf_grid_80", rp)) GTEST_SKIP() << "acopf_grid_80 fixture not present";
    IpmSession s;
    ASSERT_EQ(s.analyse(rp.A, {}, ipopt_control(), reg_mode()), ErrorCode::Success);
    for (std::size_t k = 0; k < rp.steps.size(); ++k) {
        s.factor(rp.steps[k]);
        EXPECT_EQ(s.info().delayed_pivots, 0) << "step " << k;
    }
}

