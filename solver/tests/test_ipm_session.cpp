// IpmSession with every IPM feature off: a pass-through to Solver that must
// reproduce it exactly on recorded IPOPT KKT sequences. Then Feature 1:
// inertia learning and early abort; Feature 2: re-analysis from the realized
// pivot order.

#include "../benchmarks/common/validation.hpp"
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/dense_indef.hpp"
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
#include <filesystem>
#include <fstream>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace smf;

namespace {

namespace fs = std::filesystem;

// A recorded IPOPT KKT sequence: the pattern (with the first step's values)
// and every factorization's values in the same entry order.
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

IpmControl features_off() {
    IpmControl ipm;
    ipm.learn_inertia = false;
    ipm.abort_on_wrong_inertia = false;
    ipm.reanalyse = false;
    ipm.static_mode = StaticMode::Off;
    return ipm;
}

std::vector<double> rhs(Int n) {
    std::vector<double> b(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = std::sin(0.013 * static_cast<double>(i)) + 0.5;
    return b;
}

double max_rel_diff(const std::vector<double> &a, const std::vector<double> &b) {
    double d = 0.0, s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        d = std::max(d, std::abs(a[i] - b[i]));
        s = std::max(s, std::abs(b[i]));
    }
    return s > 0.0 ? d / s : d;
}

// KKT [[H, J^T], [J, 0]] with kN = 40 primal and kM = 15 constraint rows
// (the Task 6 fixture of test_ipm_hooks.cpp). H = diag(-1, -1, -1, 2, ...):
// the three negative coordinates lie in null(J), J has full row rank, so the
// inertia is (37, 18, 0).
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

// Value of entry (row, col) of a lower CSC matrix (it must be in the pattern).
double &entry(CscLower &A, Int row, Int col) {
    for (Int p = A.col_ptr[static_cast<std::size_t>(col)]; p < A.col_ptr[static_cast<std::size_t>(col) + 1]; ++p)
        if (A.row_idx[static_cast<std::size_t>(p)] == row) return A.values[static_cast<std::size_t>(p)];
    ADD_FAILURE() << "entry (" << row << ", " << col << ") not in the pattern";
    static double dummy = 0.0;
    return dummy;
}

// small_kkt with H shifted so that 6 directions of null(J) are negative:
// H_jj = -20 for j = 3, 4, 5 outweighs the positive columns J couples them
// to. Same pattern; inertia (34, 21, 0).
CscLower bad_kkt() {
    CscLower A = small_kkt();
    for (Int j = 3; j < 6; ++j) entry(A, j, j) = -20.0;
    return A;
}

// small_kkt with the last row of J zeroed: J is rank-deficient by 1 and the
// KKT has one zero eigenvalue (inertia (37, 17, 1)).
CscLower singular_kkt() {
    CscLower A = small_kkt();
    const Int r = kN + kM - 1;
    entry(A, r, 3 + kM - 1) = 0.0;
    entry(A, r, 18 + kM - 1) = 0.0;
    entry(A, r, 33 + (kM - 1) % 7) = 0.0;
    return A;
}

// 2x2 saddle point [[1, 1], [1, 0]] plus an SPD diagonal block: n = 3,
// inertia (2, 1, 0).
CscLower tiny_kkt() {
    CscLower A;
    A.n = 3;
    A.col_ptr = {0, 2, 3, 4};
    A.row_idx = {0, 1, 1, 2};
    A.values = {1.0, 1.0, 0.0, 2.0};
    return A;
}

} // namespace

TEST(IpmSession, MatchesSolverOnReplay) {
    Replay r;
    if (!load_replay("acopf_grid_80", r)) GTEST_SKIP() << "acopf_grid_80 fixture not present";
    const Control ctrl = ipopt_control();
    const int n = static_cast<int>(r.A.n);

    // Reference: Solver, analysed on the first step's values (as the plugin's
    // deferred analysis does), one FactorKeep / workspace pair reused.
    Solver solver;
    Info rinfo;
    CscLower Ar = r.A;
    Ar.values = r.steps[0];
    auto ak = solver.analyse(Ar, ctrl, rinfo);
    ASSERT_NE(ak.get(), nullptr);
    FactorKeep fk;
    NumericWorkspace nws;
    SolveWorkspace sws;

    // Session: pattern only, analysis deferred to the first factor.
    IpmSession session;
    ASSERT_EQ(session.analyse(r.A, {}, ctrl, features_off()), ErrorCode::Success);
    EXPECT_EQ(session.analysis_keep(), nullptr);

    const std::vector<double> b = rhs(r.A.n);
    for (std::size_t k = 0; k < r.steps.size(); ++k) {
        SCOPED_TRACE("step " + std::to_string(k));
        Ar.values = r.steps[k];
        const FactorStatus fs = solver.factor(*ak, Ar, ctrl, rinfo, fk, nws);
        ASSERT_TRUE(fs == FactorStatus::Success || fs == FactorStatus::MaxPivotDelays);
        std::vector<double> xr = b;
        ASSERT_EQ(solver.solve(fk, ctrl, rinfo, xr.data(), n, 1, sws), 0);

        ASSERT_EQ(session.factor(r.steps[k]), IpmFactorStatus::Ok);
        ASSERT_NE(session.analysis_keep(), nullptr);
        std::vector<double> xs = b;
        ASSERT_EQ(session.solve(xs.data(), n, 1), 0);

        EXPECT_LE(max_rel_diff(xs, xr), 1e-14);
        const InertiaCounts in = session.inertia();
        EXPECT_EQ(in.positive, rinfo.num_positive);
        EXPECT_EQ(in.negative, rinfo.num_negative);
        EXPECT_EQ(in.zero, rinfo.num_zero);
        EXPECT_EQ(session.info().delayed_pivots, rinfo.delayed_pivots);
        EXPECT_EQ(session.info().num_negative, rinfo.num_negative);
        EXPECT_EQ(session.info().factor_status, fs);
    }
    const IpmStats &st = session.stats();
    EXPECT_EQ(st.factorizations, static_cast<int>(r.steps.size()));
    EXPECT_EQ(st.accepted, static_cast<int>(r.steps.size()));
    EXPECT_EQ(st.aborted, 0);
    EXPECT_EQ(st.reanalyses_tried, 0);
    EXPECT_EQ(st.static_pivots, 0);
}

TEST(IpmSession, NewPatternResets) {
    Replay ra, rb;
    if (!load_replay("acopf_grid_80", ra) || !load_replay("pde_control_120", rb))
        GTEST_SKIP() << "acopf_grid_80 / pde_control_120 fixtures not present";
    ASSERT_NE(ra.A.n, rb.A.n);
    const Control ctrl = ipopt_control();

    IpmSession session;
    ASSERT_EQ(session.analyse(ra.A, ra.steps[0], ctrl, features_off()), ErrorCode::Success);
    ASSERT_NE(session.analysis_keep(), nullptr); // values given: analysed now
    ASSERT_EQ(session.factor(ra.steps[0]), IpmFactorStatus::Ok);
    std::vector<double> xa = rhs(ra.A.n);
    ASSERT_EQ(session.solve(xa.data(), static_cast<int>(ra.A.n), 1), 0);
    EXPECT_EQ(session.stats().factorizations, 1);
    EXPECT_EQ(session.stats().accepted, 1);

    // A different pattern (different n): everything starts over.
    ASSERT_EQ(session.analyse(rb.A, {}, ctrl, features_off()), ErrorCode::Success);
    const IpmStats &st = session.stats();
    EXPECT_EQ(st.factorizations, 0);
    EXPECT_EQ(st.accepted, 0);
    EXPECT_EQ(st.aborted, 0);
    EXPECT_EQ(st.reanalyses_tried, 0);
    EXPECT_EQ(st.reanalyses_adopted, 0);
    EXPECT_EQ(st.reanalyses_reverted, 0);
    EXPECT_EQ(st.static_pivots, 0);
    EXPECT_EQ(st.flops_avoided, 0.0);

    const int n = static_cast<int>(rb.A.n);
    ASSERT_EQ(session.factor(rb.steps[0]), IpmFactorStatus::Ok);
    ASSERT_NE(session.analysis_keep(), nullptr);
    EXPECT_EQ(session.analysis_keep()->n, rb.A.n);
    EXPECT_EQ(session.factor_keep().n, rb.A.n);
    const InertiaCounts in = session.inertia();
    EXPECT_EQ(in.positive + in.negative + in.zero, n);

    // Same answer as a fresh Solver on pattern B.
    Solver solver;
    Info info;
    CscLower B = rb.A;
    B.values = rb.steps[0];
    auto ak = solver.analyse(B, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);
    FactorKeep fk;
    NumericWorkspace nws;
    SolveWorkspace sws;
    ASSERT_EQ(solver.factor(*ak, B, ctrl, info, fk, nws), FactorStatus::Success);
    std::vector<double> xr = rhs(rb.A.n), xs = rhs(rb.A.n);
    ASSERT_EQ(solver.solve(fk, ctrl, info, xr.data(), n, 1, sws), 0);
    ASSERT_EQ(session.solve(xs.data(), n, 1), 0);
    EXPECT_LE(max_rel_diff(xs, xr), 1e-14);
    EXPECT_EQ(in.negative, info.num_negative);
    EXPECT_EQ(session.stats().factorizations, 1);
}

// Calls out of order and malformed input are errors, never crashes; a
// one-shot user scale applies to the next factor only.
TEST(IpmSession, DeferredAnalysisAndErrors) {
    Control ctrl = ipopt_control();
    IpmSession session;
    const std::vector<double> v = {1.0, 1.0, 0.0, 2.0};
    EXPECT_EQ(session.factor(v), IpmFactorStatus::Error); // no analyse yet
    double x0[3] = {1.0, 1.0, 1.0};
    EXPECT_NE(session.solve(x0, 3, 1), 0);

    const CscLower A = tiny_kkt();
    const std::vector<double> short_values = {1.0, 2.0};
    EXPECT_NE(session.analyse(A, short_values, ctrl, features_off()), ErrorCode::Success);

    ASSERT_EQ(session.analyse(A, {}, ctrl, features_off()), ErrorCode::Success);
    EXPECT_EQ(session.analysis_keep(), nullptr);
    EXPECT_EQ(session.factor(short_values), IpmFactorStatus::Error);

    ASSERT_EQ(session.factor(v), IpmFactorStatus::Ok);
    ASSERT_NE(session.analysis_keep(), nullptr);
    const InertiaCounts in = session.inertia();
    EXPECT_EQ(in.positive, 2);
    EXPECT_EQ(in.negative, 1);
    EXPECT_EQ(in.zero, 0);
    double x[3] = {2.0, 1.0, 2.0}; // A * [1, 1, 1]
    ASSERT_EQ(session.solve(x, 3, 1), 0);
    for (double xi : x) EXPECT_NEAR(xi, 1.0, 1e-14);

    // User scale: a zero factor is rejected; the next factor is unscaled again.
    const double bad_scale[3] = {1.0, 0.0, 1.0};
    ctrl.scaling = ScalingMethod::UserSupplied;
    session.set_control(ctrl);
    session.set_user_scale(bad_scale);
    EXPECT_NE(session.factor(v), IpmFactorStatus::Ok);
    ctrl.scaling = ScalingMethod::None;
    session.set_control(ctrl);
    EXPECT_EQ(session.factor(v), IpmFactorStatus::Ok);
    EXPECT_EQ(session.stats().factorizations, 3);
    EXPECT_EQ(session.stats().accepted, 1);
}

// ---------------------------------------------------------------------------
// Feature 1: inertia learning and early abort.

namespace {

// Backward error of x as a solution of A x = b.
double eta(const CscLower &A, const std::vector<double> &b, const std::vector<double> &x) {
    return smf::bench::validate(A, b, x, 1).eta_inf;
}

// A session on the small KKT pattern with Feature 1 on (IpmControl defaults).
struct AbortHarness {
    Control ctrl = ipopt_control();
    CscLower good = small_kkt();
    CscLower bad = bad_kkt();
    IpmSession session;
    int n = static_cast<int>(good.n);

    AbortHarness() { EXPECT_EQ(session.analyse(good, {}, ctrl, IpmControl{}), ErrorCode::Success); }

    // Factor the good KKT and solve with it: the session learns 18.
    void accept_good() {
        ASSERT_EQ(session.factor(good.values), IpmFactorStatus::Ok);
        ASSERT_EQ(session.inertia().negative, kM + 3);
        std::vector<double> x = rhs(good.n);
        ASSERT_EQ(session.solve(x.data(), n, 1), 0);
        EXPECT_LE(eta(good, rhs(good.n), x), 1e-10);
    }
};

} // namespace

TEST(IpmSession, FixtureInertia) {
    // The fixtures' inertia, with every feature off.
    const Control ctrl = ipopt_control();
    const std::pair<CscLower, InertiaCounts> cases[] = {
        {small_kkt(), {37, 18, 0}}, {bad_kkt(), {34, 21, 0}}, {singular_kkt(), {37, 17, 1}}};
    for (const auto &[A, want] : cases) {
        IpmSession s;
        ASSERT_EQ(s.analyse(A, {}, ctrl, features_off()), ErrorCode::Success);
        const IpmFactorStatus st = s.factor(A.values);
        EXPECT_TRUE(st == IpmFactorStatus::Ok || st == IpmFactorStatus::Singular);
        const InertiaCounts got = s.inertia();
        EXPECT_TRUE(got.positive == want.positive && got.negative == want.negative && got.zero == want.zero)
            << "inertia (" << got.positive << ", " << got.negative << ", " << got.zero << ")";
    }
}

TEST(IpmSession, LearnsAndAborts) {
    AbortHarness h;
    EXPECT_FALSE(h.session.expected_negatives().has_value());
    h.accept_good();
    ASSERT_TRUE(h.session.expected_negatives().has_value());
    EXPECT_EQ(*h.session.expected_negatives(), 18);

    EXPECT_EQ(h.session.factor(h.bad.values), IpmFactorStatus::WrongInertia);
    EXPECT_EQ(h.session.info().factor_status, FactorStatus::InertiaExceeded);
    EXPECT_GT(h.session.info().num_negative, 18); // partial count
    EXPECT_LE(h.session.info().num_negative, 21);
    EXPECT_GT(h.session.inertia().negative, 18);
    const IpmStats &st = h.session.stats();
    EXPECT_EQ(st.factorizations, 2);
    EXPECT_EQ(st.accepted, 1);
    EXPECT_EQ(st.aborted, 1);
    EXPECT_GE(st.flops_avoided, 0.0);
    EXPECT_EQ(st.flops_avoided, h.session.info().aborted_flops_skipped);

    // A refactor with the right inertia (IPOPT's next delta) is not aborted.
    EXPECT_EQ(h.session.factor(h.good.values), IpmFactorStatus::Ok);
    EXPECT_EQ(h.session.inertia().negative, 18);
    EXPECT_EQ(h.session.stats().aborted, 1);
}

TEST(IpmSession, NoAbortBeforeFirstAccept) {
    AbortHarness h;
    EXPECT_EQ(h.session.factor(h.bad.values), IpmFactorStatus::Ok);
    EXPECT_EQ(h.session.inertia().negative, 21);
    // A factor that is never solved with teaches nothing.
    EXPECT_EQ(h.session.factor(h.good.values), IpmFactorStatus::Ok);
    EXPECT_EQ(h.session.factor(h.bad.values), IpmFactorStatus::Ok);
    EXPECT_EQ(h.session.inertia().negative, 21);
    EXPECT_FALSE(h.session.expected_negatives().has_value());
    EXPECT_EQ(h.session.stats().aborted, 0);
    EXPECT_EQ(h.session.stats().accepted, 0);
}

TEST(IpmSession, SolveAfterAbortCompletes) {
    AbortHarness h;
    h.accept_good();
    ASSERT_EQ(h.session.factor(h.bad.values), IpmFactorStatus::WrongInertia);

    const std::vector<double> b = rhs(h.bad.n);
    std::vector<double> x = b;
    ASSERT_EQ(h.session.solve(x.data(), h.n, 1), 0);
    EXPECT_LE(eta(h.bad, b, x), 1e-10);
    // The completed factor is the one accepted: full inertia, learned.
    EXPECT_EQ(h.session.inertia().negative, 21);
    EXPECT_EQ(h.session.inertia().positive, 34);
    EXPECT_EQ(h.session.stats().accepted, 2);
    EXPECT_EQ(*h.session.expected_negatives(), 21);
    // Further solves reuse the completed factor.
    std::vector<double> y = b;
    ASSERT_EQ(h.session.solve(y.data(), h.n, 1), 0);
    EXPECT_EQ(max_rel_diff(y, x), 0.0);
    EXPECT_EQ(h.session.stats().accepted, 2);
}

// Ruling 1: the completion refactors with the FACTOR's Control and user
// scale, not with what a caller set for the solve (the MA97 plugin sets the
// solve's control block, with no scale, right before solving).
TEST(IpmSession, SolveAfterAbortUsesFactorScale) {
    AbortHarness h;
    Control fctrl = h.ctrl;
    fctrl.scaling = ScalingMethod::UserSupplied;
    std::vector<double> scale(static_cast<std::size_t>(h.n));
    for (std::size_t i = 0; i < scale.size(); ++i) scale[i] = 0.5 + 0.01 * static_cast<double>(i);

    h.session.set_control(fctrl);
    h.session.set_user_scale(scale.data());
    ASSERT_EQ(h.session.factor(h.good.values), IpmFactorStatus::Ok);
    h.session.set_control(fctrl); // solve-time control: no user scale
    std::vector<double> x = rhs(h.good.n);
    ASSERT_EQ(h.session.solve(x.data(), h.n, 1), 0);
    ASSERT_EQ(*h.session.expected_negatives(), 18);

    h.session.set_control(fctrl);
    h.session.set_user_scale(scale.data());
    ASSERT_EQ(h.session.factor(h.bad.values), IpmFactorStatus::WrongInertia);
    // The caller's buffer is gone (a zero scale would be rejected).
    std::fill(scale.begin(), scale.end(), 0.0);
    Control sctrl = fctrl;
    sctrl.solve_use_blas3_single_rhs = false;
    h.session.set_control(sctrl);

    const std::vector<double> b = rhs(h.bad.n);
    x = b;
    ASSERT_EQ(h.session.solve(x.data(), h.n, 1), 0);
    EXPECT_LE(eta(h.bad, b, x), 1e-10);
    EXPECT_EQ(h.session.inertia().negative, 21);
}

TEST(IpmSession, ExplicitExpectationAndSwitches) {
    {
        // An explicit expectation arms the abort before any accept ...
        AbortHarness h;
        EXPECT_EQ(h.session.factor(h.bad.values, 18), IpmFactorStatus::WrongInertia);
        EXPECT_EQ(h.session.stats().aborted, 1);
        // ... and overrides the learned one.
        h.accept_good();
        EXPECT_EQ(h.session.factor(h.bad.values, 21), IpmFactorStatus::Ok);
        EXPECT_EQ(h.session.factor(h.good.values, 10), IpmFactorStatus::WrongInertia);
    }
    {
        // learn_inertia off: no learned expectation, no abort.
        IpmControl ipm;
        ipm.learn_inertia = false;
        AbortHarness h;
        ASSERT_EQ(h.session.analyse(h.good, {}, h.ctrl, ipm), ErrorCode::Success);
        h.accept_good();
        EXPECT_FALSE(h.session.expected_negatives().has_value());
        EXPECT_EQ(h.session.factor(h.bad.values), IpmFactorStatus::Ok);
    }
    {
        // abort off: never aborts, even with an explicit expectation.
        AbortHarness h;
        ASSERT_EQ(h.session.analyse(h.good, {}, h.ctrl, features_off()), ErrorCode::Success);
        h.accept_good();
        EXPECT_EQ(h.session.factor(h.bad.values), IpmFactorStatus::Ok);
        EXPECT_EQ(h.session.factor(h.bad.values, 18), IpmFactorStatus::Ok);
        EXPECT_EQ(h.session.stats().aborted, 0);
    }
    {
        // A new analysis forgets the expectation.
        AbortHarness h;
        h.accept_good();
        ASSERT_EQ(h.session.analyse(h.good, {}, h.ctrl, IpmControl{}), ErrorCode::Success);
        EXPECT_FALSE(h.session.expected_negatives().has_value());
        EXPECT_EQ(h.session.factor(h.bad.values), IpmFactorStatus::Ok);
    }
}

// Review Focus 3: a singular KKT has fewer negatives than expected, so the
// abort never fires; its zero pivots are still reported.
TEST(IpmSession, SingularStillReportsZeros) {
    const CscLower S = singular_kkt();
    {
        AbortHarness h; // no accepted factor
        const IpmFactorStatus st = h.session.factor(S.values);
        EXPECT_TRUE(st == IpmFactorStatus::Singular || st == IpmFactorStatus::Ok);
        EXPECT_NE(st, IpmFactorStatus::WrongInertia);
        EXPECT_GE(h.session.inertia().zero, 1);
    }
    {
        AbortHarness h; // expectation 18 learned, then the singular one
        h.accept_good();
        const IpmFactorStatus st = h.session.factor(S.values);
        EXPECT_TRUE(st == IpmFactorStatus::Singular || st == IpmFactorStatus::Ok);
        EXPECT_NE(st, IpmFactorStatus::WrongInertia);
        EXPECT_GE(h.session.inertia().zero, 1);
        EXPECT_EQ(h.session.stats().aborted, 0);
    }
}

// ---------------------------------------------------------------------------
// Feature 2: re-analysis from the realized pivot order.

namespace {

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

// Saddle point [L B^T; B 0] (the Task 8 matrix of test_ipm_hooks.cpp): L a
// 2D Laplacian on m x m; constraint r couples primal v = r * stride (b1) and
// v + 1 (-b2), b1, b2 in [0.5, 2). With u = 0.5 and nemin = 1 its AMD order
// delays heavily, as robot_arm's KKT does under IPOPT.
CscLower grid_kkt(int m, int stride) {
    const int nv = m * m;
    const int nc = nv / stride;
    std::vector<std::vector<std::pair<int, double>>> cols(static_cast<std::size_t>(nv + nc));
    const CscLower L = laplacian_2d(m);
    for (int j = 0; j < nv; ++j)
        for (Int p = L.col_ptr[static_cast<std::size_t>(j)]; p < L.col_ptr[static_cast<std::size_t>(j) + 1]; ++p)
            cols[static_cast<std::size_t>(j)].push_back(
                {L.row_idx[static_cast<std::size_t>(p)], L.values[static_cast<std::size_t>(p)]});
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

// A later KKT on grid_kkt's pattern for which A1's realized order is bad:
// each constraint's coupling to its first primal v (the one it was
// eliminated after on A1) is 100x weaker and H is 10x smaller (Sigma -> 0).
// Under A1's realized order it delays more than A1 did (and more than AMD).
CscLower shifted_kkt(const CscLower &A1, int nv, int stride) {
    CscLower A2 = A1;
    for (Int j = 0; j < A2.n; ++j)
        for (Int p = A2.col_ptr[static_cast<std::size_t>(j)]; p < A2.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
            const Int i = A2.row_idx[static_cast<std::size_t>(p)];
            double &v = A2.values[static_cast<std::size_t>(p)];
            if (i >= nv && j < nv && j % stride == 0) v *= 0.01;
            if (i == j && j < nv) v *= 0.1;
        }
    return A2;
}

// The delay-prone setting of Task 8 (AMD, u = 0.5, nemin = 1, no scaling).
Control delay_ctrl() {
    Control c;
    c.matrix_type = MatrixType::RealSymmetricIndefinite;
    c.ordering = OrderingMethod::AMD;
    c.scaling = ScalingMethod::None;
    c.num_threads = 1;
    c.nemin = 1;
    c.pivot_u = 0.5;
    return c;
}

// Feature 2 alone, triggered by any delay at every factorization.
IpmControl reanalyse_eager() {
    IpmControl ipm = features_off();
    ipm.reanalyse = true;
    ipm.reanalyse_delay_frac = 0.0;
    ipm.reanalyse_min_gap = 1;
    return ipm;
}

// factor() then solve() (the IPM accepts the factor); checks the solution.
void factor_and_solve(IpmSession &s, const CscLower &A) {
    ASSERT_EQ(s.factor(A.values), IpmFactorStatus::Ok);
    const std::vector<double> b = rhs(A.n);
    std::vector<double> x = b;
    ASSERT_EQ(s.solve(x.data(), static_cast<int>(A.n), 1), 0);
    EXPECT_LE(eta(A, b, x), 1e-10);
}

} // namespace

TEST(IpmSession, ReanalysisAdoptedRemovesDelays) {
    const CscLower A = grid_kkt(40, 7);
    IpmSession s;
    ASSERT_EQ(s.analyse(A, {}, delay_ctrl(), reanalyse_eager()), ErrorCode::Success);

    factor_and_solve(s, A);
    const AnalysisKeep *first = s.analysis_keep();
    ASSERT_NE(first, nullptr);
    const int first_delays = s.info().delayed_pivots;
    EXPECT_GT(first_delays, 0);
    EXPECT_EQ(s.stats().reanalyses_tried, 0);

    // The accepted factor delayed: the next factor() re-analyses from its
    // realized order, adopts it and factors without delays.
    factor_and_solve(s, A);
    EXPECT_EQ(s.stats().reanalyses_tried, 1);
    EXPECT_EQ(s.stats().reanalyses_adopted, 1);
    EXPECT_EQ(s.stats().reanalyses_reverted, 0);
    EXPECT_EQ(s.info().delayed_pivots, 0);
    EXPECT_NE(s.analysis_keep(), first);
    EXPECT_EQ(s.info().ordering_used, static_cast<int>(OrderingMethod::UserSupplied));
    EXPECT_EQ(s.factor_keep().analysis, s.analysis_keep());
    EXPECT_GT(s.stats().reanalyse_seconds, 0.0);

    // Probation passed; no delays left, so nothing more is tried.
    const AnalysisKeep *adopted = s.analysis_keep();
    for (int k = 0; k < 3; ++k) factor_and_solve(s, A);
    EXPECT_EQ(s.stats().reanalyses_tried, 1);
    EXPECT_EQ(s.stats().reanalyses_reverted, 0);
    EXPECT_EQ(s.analysis_keep(), adopted);
    EXPECT_EQ(s.info().delayed_pivots, 0);
}

TEST(IpmSession, ReanalysisRevertsWhenDelaysGrow) {
    // A2 weakens the couplings A1's realized order relied on (shifted_kkt).
    const CscLower A1 = grid_kkt(40, 7);
    const CscLower A2 = shifted_kkt(A1, 40 * 40, 7);
    IpmSession s;
    ASSERT_EQ(s.analyse(A1, {}, delay_ctrl(), reanalyse_eager()), ErrorCode::Success);
    factor_and_solve(s, A1);
    const AnalysisKeep *original = s.analysis_keep();
    const int a1_delays = s.info().delayed_pivots;
    ASSERT_GT(a1_delays, 0);

    // Probation factor: re-analysed from A1's order, it delays more on A2.
    factor_and_solve(s, A2); // the probation factor stays solvable
    EXPECT_EQ(s.stats().reanalyses_tried, 1);
    EXPECT_EQ(s.stats().reanalyses_adopted, 1);
    EXPECT_GT(s.info().delayed_pivots, a1_delays);
    EXPECT_EQ(s.stats().reanalyses_reverted, 1);
    EXPECT_NE(s.analysis_keep(), original);
    // A second solve with the probation factor (its analysis is still alive).
    {
        const std::vector<double> b = rhs(A2.n);
        std::vector<double> x = b;
        ASSERT_EQ(s.solve(x.data(), static_cast<int>(A2.n), 1), 0);
        EXPECT_LE(eta(A2, b, x), 1e-10);
    }

    // Reverted: the next factor runs on the original analysis, and this
    // pattern is never re-analysed again although every factor delays.
    for (int k = 0; k < 4; ++k) {
        factor_and_solve(s, k % 2 ? A2 : A1);
        EXPECT_EQ(s.analysis_keep(), original);
        EXPECT_EQ(s.factor_keep().analysis, original);
        EXPECT_GT(s.info().delayed_pivots, 0);
    }
    EXPECT_EQ(s.stats().reanalyses_tried, 1);
    EXPECT_EQ(s.stats().reanalyses_adopted, 1);
    EXPECT_EQ(s.stats().reanalyses_reverted, 1);
}

TEST(IpmSession, ReanalysisRejectedOnFlops) {
    const CscLower A = grid_kkt(40, 7);
    IpmControl ipm = reanalyse_eager();
    ipm.reanalyse_flops_ratio = 0.0;
    IpmSession s;
    ASSERT_EQ(s.analyse(A, {}, delay_ctrl(), ipm), ErrorCode::Success);
    factor_and_solve(s, A);
    const AnalysisKeep *first = s.analysis_keep();
    const int first_delays = s.info().delayed_pivots;
    ASSERT_GT(first_delays, 0);

    factor_and_solve(s, A);
    EXPECT_EQ(s.stats().reanalyses_tried, 1);
    EXPECT_EQ(s.stats().reanalyses_adopted, 0);
    EXPECT_EQ(s.stats().reanalyses_reverted, 0);
    EXPECT_EQ(s.analysis_keep(), first);
    EXPECT_EQ(s.info().delayed_pivots, first_delays);
}

TEST(IpmSession, ReanalysisWaitsForTheGap) {
    // reanalyse_min_gap = 3 after a rejected attempt at factorization 2: the
    // next attempt is at factorization 5, then 8.
    const CscLower A = grid_kkt(40, 7);
    IpmControl ipm = reanalyse_eager();
    ipm.reanalyse_flops_ratio = 0.0;
    ipm.reanalyse_min_gap = 3;
    IpmSession s;
    ASSERT_EQ(s.analyse(A, {}, delay_ctrl(), ipm), ErrorCode::Success);
    const int want_tried[] = {0, 1, 1, 1, 2, 2, 2, 3};
    for (int k = 0; k < 8; ++k) {
        factor_and_solve(s, A);
        EXPECT_EQ(s.stats().reanalyses_tried, want_tried[k]) << "factorization " << k + 1;
    }
}

TEST(IpmSession, ReanalysisNeedsAnAcceptedFactorAndEnoughDelays) {
    const CscLower A = grid_kkt(40, 7);
    {
        // Factors that are never solved with do not trigger.
        IpmSession s;
        ASSERT_EQ(s.analyse(A, {}, delay_ctrl(), reanalyse_eager()), ErrorCode::Success);
        for (int k = 0; k < 3; ++k) ASSERT_EQ(s.factor(A.values), IpmFactorStatus::Ok);
        EXPECT_EQ(s.stats().reanalyses_tried, 0);
    }
    {
        // Delays at or below reanalyse_delay_frac * n do not trigger.
        IpmSession s;
        IpmControl ipm = reanalyse_eager();
        ipm.reanalyse_delay_frac = 10.0;
        ASSERT_EQ(s.analyse(A, {}, delay_ctrl(), ipm), ErrorCode::Success);
        for (int k = 0; k < 3; ++k) factor_and_solve(s, A);
        EXPECT_EQ(s.stats().reanalyses_tried, 0);
    }
    {
        // Feature off: the analysis never changes.
        IpmSession s;
        IpmControl ipm = reanalyse_eager();
        ipm.reanalyse = false;
        ASSERT_EQ(s.analyse(A, {}, delay_ctrl(), ipm), ErrorCode::Success);
        factor_and_solve(s, A);
        const AnalysisKeep *first = s.analysis_keep();
        const int d = s.info().delayed_pivots;
        for (int k = 0; k < 3; ++k) factor_and_solve(s, A);
        EXPECT_EQ(s.stats().reanalyses_tried, 0);
        EXPECT_EQ(s.analysis_keep(), first);
        EXPECT_EQ(s.info().delayed_pivots, d);
    }
}

TEST(IpmSession, NewPatternRestartsReanalysis) {
    // A revert stops re-analysis for that pattern only: analyse() starts over.
    const CscLower A1 = grid_kkt(40, 7);
    const CscLower A2 = shifted_kkt(A1, 40 * 40, 7);
    IpmSession s;
    ASSERT_EQ(s.analyse(A1, {}, delay_ctrl(), reanalyse_eager()), ErrorCode::Success);
    factor_and_solve(s, A1);
    factor_and_solve(s, A2);
    ASSERT_EQ(s.stats().reanalyses_reverted, 1);
    ASSERT_EQ(s.analyse(A1, {}, delay_ctrl(), reanalyse_eager()), ErrorCode::Success);
    EXPECT_EQ(s.stats().reanalyses_tried, 0);
    factor_and_solve(s, A1);
    factor_and_solve(s, A1);
    EXPECT_EQ(s.stats().reanalyses_adopted, 1);
    EXPECT_EQ(s.info().delayed_pivots, 0);
}

TEST(IpmSession, ReanalysisProbationWaitsForACompleteFactor) {
    // The adopted analysis' first factor stops early (wrong inertia): it is
    // not judged. The completion inside solve() is, on the same analysis.
    const CscLower A = grid_kkt(40, 7);
    const CscLower A2 = shifted_kkt(A, 40 * 40, 7);
    IpmControl ipm = reanalyse_eager();
    ipm.abort_on_wrong_inertia = true;
    {
        IpmSession s;
        ASSERT_EQ(s.analyse(A, {}, delay_ctrl(), ipm), ErrorCode::Success);
        factor_and_solve(s, A);
        const AnalysisKeep *first = s.analysis_keep();
        ASSERT_EQ(s.factor(A.values, 10), IpmFactorStatus::WrongInertia);
        EXPECT_EQ(s.stats().reanalyses_adopted, 1);
        const AnalysisKeep *adopted = s.analysis_keep();
        EXPECT_NE(adopted, first);
        const std::vector<double> b = rhs(A.n);
        std::vector<double> x = b;
        ASSERT_EQ(s.solve(x.data(), static_cast<int>(A.n), 1), 0); // completes, then judged
        EXPECT_LE(eta(A, b, x), 1e-10);
        EXPECT_EQ(s.info().delayed_pivots, 0);
        EXPECT_EQ(s.stats().reanalyses_reverted, 0);
        factor_and_solve(s, A);
        EXPECT_EQ(s.analysis_keep(), adopted);
    }
    {
        // Aborted, then refactored (IPOPT's next delta) with more delays: the
        // refactor is the judged one, and it reverts.
        IpmSession s;
        ASSERT_EQ(s.analyse(A, {}, delay_ctrl(), ipm), ErrorCode::Success);
        factor_and_solve(s, A);
        const AnalysisKeep *first = s.analysis_keep();
        ASSERT_EQ(s.factor(A.values, 10), IpmFactorStatus::WrongInertia);
        EXPECT_EQ(s.stats().reanalyses_reverted, 0);
        ASSERT_EQ(s.factor(A2.values), IpmFactorStatus::Ok);
        EXPECT_EQ(s.stats().reanalyses_reverted, 1);
        const std::vector<double> b = rhs(A2.n);
        std::vector<double> x = b;
        ASSERT_EQ(s.solve(x.data(), static_cast<int>(A2.n), 1), 0);
        EXPECT_LE(eta(A2, b, x), 1e-10);
        factor_and_solve(s, A);
        EXPECT_EQ(s.analysis_keep(), first);
        EXPECT_EQ(s.stats().reanalyses_tried, 1);
    }
}

TEST(IpmSession, ReanalysisKeepsTheMatchingScale) {
    // The matching ordering's scaling (applied by ScalingMethod::Auto) is
    // carried over: a user-supplied order computes none. (shifted_kkt delays
    // under the matching ordering; grid_kkt does not.)
    const CscLower A = shifted_kkt(grid_kkt(40, 7), 40 * 40, 7);
    Control c = delay_ctrl();
    c.ordering = OrderingMethod::Matching;
    c.scaling = ScalingMethod::Auto;
    IpmSession s;
    ASSERT_EQ(s.analyse(A, A.values, c, reanalyse_eager()), ErrorCode::Success);
    const std::vector<double> scale0 = s.analysis_keep()->matching_scale;
    ASSERT_FALSE(scale0.empty());
    factor_and_solve(s, A);
    ASSERT_GT(s.info().delayed_pivots, 0);
    const std::vector<double> fscale0 = s.factor_keep().scale;
    factor_and_solve(s, A);
    ASSERT_EQ(s.stats().reanalyses_adopted, 1);
    EXPECT_EQ(s.analysis_keep()->matching_scale, scale0);
    EXPECT_EQ(s.factor_keep().scale, fscale0);
    EXPECT_EQ(s.info().delayed_pivots, 0);
}

namespace {

// Flops of a delay-free factor of ak in Info::actual_flops units (the kernel
// count IpmSession's adoption test predicts with).
double kernel_flops(const AnalysisKeep &ak) {
    double t = 0.0;
    for (std::size_t s = 0; s < ak.supernodes.size(); ++s) {
        const double f = static_cast<double>(ak.fronts[s].front_size());
        for (Int k = 0; k < ak.supernodes[s].width(); ++k) {
            const double r = f - static_cast<double>(k) - 1.0;
            t += r + r * (r + 1.0);
        }
    }
    return t;
}

// factor_keep().analysis is one of the live analyses (or null).
void expect_factor_points_at_a_live_analysis(const IpmSession &s) {
    const AnalysisKeep *a = s.factor_keep().analysis;
    EXPECT_TRUE(a == nullptr || a == s.analysis_keep() || a == s.previous_analysis_keep())
        << "factor_keep().analysis points at a freed analysis";
}

} // namespace

TEST(IpmSession, ReanalysisPredictsTheKernelFlops) {
    // The adoption test's units: a delay-free factor's actual flops equal
    // the kernel-count prediction of its analysis.
    const CscLower A = grid_kkt(40, 7);
    IpmSession s;
    ASSERT_EQ(s.analyse(A, {}, delay_ctrl(), reanalyse_eager()), ErrorCode::Success);
    factor_and_solve(s, A);
    factor_and_solve(s, A);
    ASSERT_EQ(s.stats().reanalyses_adopted, 1);
    ASSERT_EQ(s.info().delayed_pivots, 0);
    EXPECT_NEAR(kernel_flops(*s.analysis_keep()), s.info().actual_flops, 1e-9 * s.info().actual_flops);
}

TEST(IpmSession, ProbationIgnoresAFactorThatNeverRan) {
    // The probation factor fails in the matching scaling (a zero row has no
    // matching), before the engine runs: Singular with SingularMatrix, but
    // fk_ is still the accepted factor of the previous analysis. It must not
    // be judged, and that analysis must stay alive.
    const CscLower A = grid_kkt(40, 7);
    CscLower Z = A;
    const Int r = A.n - 1; // the last constraint row: zero its couplings
    for (Int j = 0; j < Z.n; ++j)
        for (Int p = Z.col_ptr[static_cast<std::size_t>(j)]; p < Z.col_ptr[static_cast<std::size_t>(j) + 1]; ++p)
            if (Z.row_idx[static_cast<std::size_t>(p)] == r) Z.values[static_cast<std::size_t>(p)] = 0.0;
    Control none = delay_ctrl();
    Control matching = none;
    matching.scaling = ScalingMethod::Matching;
    IpmSession s;
    ASSERT_EQ(s.analyse(A, {}, none, reanalyse_eager()), ErrorCode::Success);
    factor_and_solve(s, A);
    const AnalysisKeep *original = s.analysis_keep();

    s.set_control(matching);
    const IpmFactorStatus st = s.factor(Z.values);
    ASSERT_EQ(s.info().status, ErrorCode::SingularMatrix);
    EXPECT_EQ(st, IpmFactorStatus::Singular);
    EXPECT_EQ(s.stats().reanalyses_adopted, 1);
    EXPECT_EQ(s.stats().reanalyses_reverted, 0);
    EXPECT_EQ(s.previous_analysis_keep(), original); // probation still open
    EXPECT_EQ(s.factor_keep().analysis, original);
    expect_factor_points_at_a_live_analysis(s);

    // The next factor that runs is the judged one (no delays: pass).
    s.set_control(none);
    factor_and_solve(s, A);
    EXPECT_EQ(s.info().delayed_pivots, 0);
    EXPECT_EQ(s.previous_analysis_keep(), nullptr);
    EXPECT_EQ(s.factor_keep().analysis, s.analysis_keep());
    EXPECT_EQ(s.stats().reanalyses_reverted, 0);
    expect_factor_points_at_a_live_analysis(s);
}

// ---------------------------------------------------------------------------
// Feature 3a: StaticMode::Fallback.

namespace {

IpmControl fallback_mode(bool as_zero = false) {
    IpmControl ipm = features_off();
    ipm.static_mode = StaticMode::Fallback;
    ipm.static_as_zero = as_zero;
    return ipm;
}

struct FactorCounts {
    IpmFactorStatus status;
    InertiaCounts in;
    int static_pivots;
    long stats_static;
};

FactorCounts factor_once(const CscLower &A, const IpmControl &ipm) {
    IpmSession s;
    EXPECT_EQ(s.analyse(A, {}, ipopt_control(), ipm), ErrorCode::Success);
    const IpmFactorStatus st = s.factor(A.values);
    return {st, s.inertia(), s.info().static_pivots, s.stats().static_pivots};
}

} // namespace

TEST(IpmSession, StaticAsZeroDefaultsOff) {
    EXPECT_FALSE(IpmControl{}.static_as_zero);
}

TEST(IpmSession, FallbackStaticPivotAccounting) {
    // singular_kkt: the zeroed constraint row reaches the root unpivoted.
    // Off: a zero pivot (37, 17, 1). Fallback: a static pivot instead (d = 0
    // raised to +1e-8 max|A|). Its original d is an exact zero, so the
    // default (hybrid) accounting still counts it as a zero and the factor
    // stays singular, as with Off: IPOPT sees the rank deficiency and
    // applies delta_c. Zero accounting (static_as_zero) counts it the same.
    // (A static pivot raised from a nonzero d keeps its sign by default:
    // IpmHooks.FallbackHybridAccounting.)
    const CscLower S = singular_kkt();
    const int n = static_cast<int>(S.n);
    const auto sum = [](const InertiaCounts &c) { return c.positive + c.negative + c.zero; };

    const FactorCounts off = factor_once(S, features_off());
    EXPECT_EQ(off.status, IpmFactorStatus::Singular);
    EXPECT_EQ(off.static_pivots, 0);
    EXPECT_EQ(off.in.positive, 37);
    EXPECT_EQ(off.in.negative, 17);
    EXPECT_EQ(off.in.zero, 1);

    for (const bool as_zero : {false, true}) {
        const FactorCounts fb = factor_once(S, fallback_mode(as_zero));
        EXPECT_EQ(fb.status, IpmFactorStatus::Singular) << "as_zero " << as_zero;
        EXPECT_EQ(fb.static_pivots, 1) << "as_zero " << as_zero;
        EXPECT_EQ(fb.stats_static, 1) << "as_zero " << as_zero;
        EXPECT_GE(fb.in.zero, 1) << "as_zero " << as_zero;
        EXPECT_EQ(sum(fb.in), n);
        EXPECT_EQ(fb.in.positive, 37) << "as_zero " << as_zero;
        EXPECT_EQ(fb.in.negative, 17) << "as_zero " << as_zero;
        EXPECT_EQ(fb.in.zero, 1) << "as_zero " << as_zero;
    }
}

TEST(IpmSession, FallbackModeLeavesANonsingularKktAlone) {
    const CscLower A = small_kkt();
    const Control ctrl = ipopt_control();
    IpmSession s;
    ASSERT_EQ(s.analyse(A, {}, ctrl, fallback_mode()), ErrorCode::Success);
    ASSERT_EQ(s.factor(A.values), IpmFactorStatus::Ok);
    const InertiaCounts in = s.inertia();
    EXPECT_EQ(in.positive, 37);
    EXPECT_EQ(in.negative, 18);
    EXPECT_EQ(in.zero, 0);
    EXPECT_EQ(s.info().static_pivots, 0);
    std::vector<double> x = rhs(A.n);
    ASSERT_EQ(s.solve(x.data(), static_cast<int>(A.n), 1), 0);
    EXPECT_LE(eta(A, rhs(A.n), x), 1e-10);
}
