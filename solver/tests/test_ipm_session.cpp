// IpmSession with every IPM feature off: a pass-through to Solver that must
// reproduce it exactly on recorded IPOPT KKT sequences.

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
