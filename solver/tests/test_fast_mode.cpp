// The opt-in fast mode (Control::pivoting = Pivoting::Static): its plumbing
// (parse_pivoting, the IPOPT plugin's SMF_PIVOTING), the backward-error
// refinement trigger on every solve path (Solver::factor_solve, refine(),
// IpmSession), Auto refinement on an already-good static factor (no FGMRES
// escalation), the static path under tree_parallel, and the default
// (Threshold) path left as it was.
#include "../benchmarks/common/validation.hpp"
#include "kkt_test_matrices.hpp"
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/determinism.hpp"
#include "smf/factor_indef.hpp"
#include "smf/info.hpp"
#include "smf/ipm_session.hpp"
#include "smf/ipopt_plugin_control.hpp"
#include "smf/matrix_market.hpp"
#include "smf/refinement.hpp"
#include "smf/solver.hpp"
#include "smf/workspace.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <vector>

using namespace smf;

namespace {

void set_env(const char *name, const char *value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    if (*value) setenv(name, value, 1); else unsetenv(name);
#endif
}

std::vector<double> rhs_of(int n) {
    std::vector<double> b(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) b[static_cast<std::size_t>(i)] = 1.0 + 0.01 * (i % 97);
    return b;
}

Control indef_control() {
    Control c;
    c.matrix_type = MatrixType::RealSymmetricIndefinite;
    c.num_threads = 1;
    return c;
}

// A KKT whose unscaled threshold-pivoting delays cascade (work ratio ~21
// against the rescue's 16) but whose unscaled static factorization perturbs
// no pivot: the pde_control pattern (test_scale_on_delay's static_ok_kkt).
CscLower cascading_kkt() { return cascade_kkt(90, 300, 2.5, 1.5); }

} // namespace

// --- Plumbing ---

TEST(FastMode, DefaultIsThreshold) {
    const Control c;
    EXPECT_EQ(c.pivoting, Pivoting::Threshold);
    EXPECT_FALSE(c.static_fallback_after_cascade);
    EXPECT_EQ(c.refine_omega_target, 1e-10);
    EXPECT_EQ(c.refine_growth_threshold, std::numeric_limits<double>::infinity());
}

TEST(FastMode, ParsePivoting) {
    Pivoting p = Pivoting::Static;
    EXPECT_TRUE(parse_pivoting("threshold", p));
    EXPECT_EQ(p, Pivoting::Threshold);
    EXPECT_TRUE(parse_pivoting("static", p));
    EXPECT_EQ(p, Pivoting::Static);
    for (const char *bad : {"", "Static", "STATIC", "fast", "static ", "thresh", "1"}) {
        Pivoting q = Pivoting::Threshold;
        EXPECT_FALSE(parse_pivoting(bad, q)) << bad;
        EXPECT_EQ(q, Pivoting::Threshold) << "untouched on " << bad;
    }
    Pivoting q = Pivoting::Static;
    EXPECT_FALSE(parse_pivoting(nullptr, q));
    EXPECT_EQ(q, Pivoting::Static);
}

// SMF_PIVOTING as the plugin resolves it: unset / empty -> Threshold;
// threshold / static; anything else -> Threshold, flagged invalid.
TEST(FastMode, PluginPivotingResolve) {
    EXPECT_EQ(plugin_pivoting_resolve(nullptr).pivoting, Pivoting::Threshold);
    EXPECT_TRUE(plugin_pivoting_resolve(nullptr).valid);
    EXPECT_EQ(plugin_pivoting_resolve("").pivoting, Pivoting::Threshold);
    EXPECT_TRUE(plugin_pivoting_resolve("").valid);
    EXPECT_EQ(plugin_pivoting_resolve("static").pivoting, Pivoting::Static);
    EXPECT_TRUE(plugin_pivoting_resolve("static").valid);
    EXPECT_EQ(plugin_pivoting_resolve("threshold").pivoting, Pivoting::Threshold);
    EXPECT_TRUE(plugin_pivoting_resolve("threshold").valid);
    EXPECT_EQ(plugin_pivoting_resolve("fast").pivoting, Pivoting::Threshold);
    EXPECT_FALSE(plugin_pivoting_resolve("fast").valid);

    // plugin_pivoting_from_env() reads the environment now.
    set_env("SMF_PIVOTING", "static");
    EXPECT_EQ(plugin_pivoting_from_env().pivoting, Pivoting::Static);
    set_env("SMF_PIVOTING", "threshold");
    EXPECT_EQ(plugin_pivoting_from_env().pivoting, Pivoting::Threshold);
    set_env("SMF_PIVOTING", "bogus");
    EXPECT_FALSE(plugin_pivoting_from_env().valid);
    set_env("SMF_PIVOTING", "");
    EXPECT_EQ(plugin_pivoting_from_env().pivoting, Pivoting::Threshold);
    EXPECT_TRUE(plugin_pivoting_from_env().valid);
}

// The plugin's Control takes the pivoting given, else plugin_pivoting()
// (SMF_PIVOTING, read once per process); every other field is the same.
TEST(FastMode, PluginControlCarriesPivoting) {
    const auto def = ipopt_plugin_control(Ma97Options{}, -1, 1);
    ASSERT_TRUE(def.has_value());
    EXPECT_EQ(def->pivoting, plugin_pivoting());
    const auto st = ipopt_plugin_control(Ma97Options{}, -1, 1, Pivoting::Static);
    const auto th = ipopt_plugin_control(Ma97Options{}, -1, 1, Pivoting::Threshold);
    ASSERT_TRUE(st.has_value());
    ASSERT_TRUE(th.has_value());
    EXPECT_EQ(st->pivoting, Pivoting::Static);
    EXPECT_EQ(th->pivoting, Pivoting::Threshold);
    EXPECT_EQ(st->pivot_u, th->pivot_u);
    EXPECT_EQ(st->scaling, th->scaling);
    EXPECT_EQ(st->ordering, th->ordering);
    EXPECT_EQ(st->delay_budget, th->delay_budget);
    // apply_ma97_options() on an existing Control sets it the same way.
    Control c;
    ASSERT_TRUE(apply_ma97_options(c, Ma97Options{}, -1, 1, Pivoting::Static));
    EXPECT_EQ(c.pivoting, Pivoting::Static);
}

// --- Static mode on a cascading KKT ---

// A KKT whose threshold-pivoting delays cascade: fast mode factors it with
// no delay and no perturbed pivot (exact inertia, the threshold factor's),
// and every solve path refines it to refine_omega_target.
TEST(FastMode, StaticReachesOmegaTargetOnCascadingKkt) {
    const CscLower A = cascading_kkt();
    const int n = static_cast<int>(A.n);
    Solver s;

    // Premise: threshold pivoting, rescue off, cascades.
    Info ith;
    {
        Control t = indef_control();
        t.amalgamation = Amalgamation::Smf;
        t.scale_on_delay_ratio = -1.0;
        Info ia;
        auto ak = s.analyse(A, t, ia);
        ASSERT_NE(ak.get(), nullptr);
        FactorKeep fk;
        Info &i = ith;
        ASSERT_EQ(s.factor(*ak, t, i, fk), FactorStatus::Success);
        std::printf("[FastMode] threshold: %d delays, work ratio %.2f\n", i.delayed_pivots,
                    i.front_work / delay_free_front_work(*ak));
        EXPECT_GT(i.delayed_pivots, 0);
        EXPECT_FALSE(fk.static_path);
        EXPECT_TRUE(i.inertia_exact);
    }

    Control ctrl = indef_control();
    ctrl.amalgamation = Amalgamation::Smf;
    ctrl.pivoting = Pivoting::Static;
    Info ia;
    auto ak = s.analyse(A, ctrl, ia);
    ASSERT_NE(ak.get(), nullptr);
    FactorKeep fk;
    Info i;
    ASSERT_EQ(s.factor(*ak, ctrl, i, fk), FactorStatus::Success);
    std::printf("[FastMode] static: %d delays, %d perturbed, growth %.3g\n", i.delayed_pivots,
                i.static_pivots, i.pivot_growth);
    EXPECT_EQ(i.delayed_pivots, 0);
    EXPECT_EQ(i.static_pivots, 0);
    EXPECT_EQ(i.static_fallback, 1);
    EXPECT_EQ(i.delay_rescales, 0);
    EXPECT_TRUE(i.inertia_exact);
    EXPECT_EQ(i.num_positive, ith.num_positive);
    EXPECT_EQ(i.num_negative, ith.num_negative);
    EXPECT_EQ(i.num_zero, ith.num_zero);
    EXPECT_TRUE(fk.static_path);
    EXPECT_TRUE(refinement_advised(fk));

    const std::vector<double> b = rhs_of(n);
    // Solver::solve + refine(), as the docs ask after a plain factor.
    {
        std::vector<double> x = b;
        Info si;
        ASSERT_EQ(s.solve(fk, ctrl, si, x.data(), n, 1), 0);
        SolveWorkspace ws;
        const RefinementResult rr = refine(A, fk, ctrl, b, x, 1, ws);
        std::printf("[FastMode] refine: omega %.3g -> %.3g in %d steps (method %d)\n", rr.omega_start,
                    rr.omega, rr.iterations, static_cast<int>(rr.method_used));
        EXPECT_EQ(rr.status, ErrorCode::Success);
        EXPECT_LE(rr.omega, ctrl.refine_omega_target);
        const auto val = smf::bench::validate(A, b, x, 1);
        EXPECT_LE(val.omega, 10.0 * ctrl.refine_omega_target) << "independent check";
    }
    // Solver::factor_solve refines on its own.
    {
        std::vector<double> x = b;
        Info fi;
        auto fk2 = s.factor_solve(A, ctrl, fi, x.data(), n, 1);
        ASSERT_NE(fk2.get(), nullptr);
        EXPECT_TRUE(fk2->static_path);
        const auto val = smf::bench::validate(A, b, x, 1);
        std::printf("[FastMode] factor_solve: omega %.3g, %d steps\n", val.omega, fi.refinement_iterations);
        EXPECT_LE(val.omega, 10.0 * ctrl.refine_omega_target);
    }
    // IpmSession (the IPOPT plugin's path) refines a static-path factor's
    // full solves against the factored values: classical refinement, as
    // refine() does it. A tight omega target makes it take steps here.
    {
        Control sc = ctrl;
        sc.refine_omega_target = 1e-15;
        IpmSession sess;
        ASSERT_EQ(sess.analyse(A, A.values, sc, IpmControl{}), ErrorCode::Success);
        ASSERT_EQ(sess.factor(A.values), IpmFactorStatus::Ok);
        EXPECT_TRUE(sess.factor_keep().static_path);
        EXPECT_EQ(sess.inertia().negative, ith.num_negative);
        // The reference: the same factor's solve, refined by refine().
        std::vector<double> x0 = b;
        Info si;
        ASSERT_EQ(s.solve(sess.factor_keep(), sc, si, x0.data(), n, 1), 0);
        std::vector<double> xr = x0;
        Control rc = sc;
        rc.refine_method = RefineMethod::Classical;
        SolveWorkspace ws;
        const RefinementResult rr = refine(A, sess.factor_keep(), rc, b, xr, 1, ws);
        std::vector<double> x = b;
        ASSERT_EQ(sess.solve(x.data(), n, 1), 0);
        const auto val = smf::bench::validate(A, b, x, 1);
        std::printf("[FastMode] IpmSession: reference %d steps, omega %.3g -> %.3g; session omega %.3g\n",
                    rr.iterations, rr.omega_start, rr.omega, val.omega);
        EXPECT_GE(rr.iterations, 1);
        EXPECT_EQ(x, xr) << "the session refines as refine() does";
        EXPECT_LE(val.omega, 10.0 * ctrl.refine_omega_target);
        // A second solve with the same factor (refinement values set once
        // per factor) gives the same answer.
        std::vector<double> y = b;
        ASSERT_EQ(sess.solve(y.data(), n, 1), 0);
        EXPECT_EQ(x, y);
        // New values in the same session: refined against the new ones.
        CscLower A1 = A;
        for (double &v : A1.values) v *= 2.0;
        ASSERT_EQ(sess.factor(A1.values), IpmFactorStatus::Ok);
        std::vector<double> z0 = b;
        ASSERT_EQ(s.solve(sess.factor_keep(), sc, si, z0.data(), n, 1), 0);
        std::vector<double> zr = z0;
        refine(A1, sess.factor_keep(), rc, b, zr, 1, ws);
        std::vector<double> z = b;
        ASSERT_EQ(sess.solve(z.data(), n, 1), 0);
        EXPECT_EQ(z, zr) << "refined against the new values";
        const auto v1 = smf::bench::validate(A1, b, z, 1);
        EXPECT_LE(v1.omega, 10.0 * ctrl.refine_omega_target);
    }
    // Threshold IpmSession: no static path, no refinement (as before).
    {
        Control t = ctrl;
        t.pivoting = Pivoting::Threshold;
        IpmSession sess;
        ASSERT_EQ(sess.analyse(A, A.values, t, IpmControl{}), ErrorCode::Success);
        ASSERT_EQ(sess.factor(A.values), IpmFactorStatus::Ok);
        EXPECT_FALSE(sess.factor_keep().static_path);
        std::vector<double> x0 = b;
        Info si;
        ASSERT_EQ(s.solve(sess.factor_keep(), t, si, x0.data(), n, 1), 0);
        std::vector<double> x = b;
        ASSERT_EQ(sess.solve(x.data(), n, 1), 0);
        EXPECT_EQ(x, x0) << "threshold-pivoting session solves are not refined";
    }
}

// The real thing, when the (untracked) fixture is there: pde_control_120's
// last IPOPT step on the recommended track's Control at 1 thread. Threshold
// pivoting delays; fast mode delays nothing, perturbs nothing, keeps the
// threshold factor's inertia and refines to the accuracy gate.
TEST(FastMode, PdeControlFixture) {
    namespace fs = std::filesystem;
    const fs::path dir = fs::path(__FILE__).parent_path() / ".." / ".." / "benchmarks" / "fixtures" /
                         "ipopt_kkt" / "pde_control_120";
    const fs::path mtx = dir / "pde_control_120.mtx";
    const fs::path vals_path = dir / "pde_control_120.mtx.vals";
    if (!fs::exists(mtx) || !fs::exists(vals_path)) GTEST_SKIP() << "fixture missing";
    auto mm = read_matrix_market(mtx.string());
    ASSERT_TRUE(mm.ok) << mm.error_message;
    CscLower A = std::move(mm.matrix);
    const std::size_t nnz = A.values.size();
    const auto bytes = fs::file_size(vals_path);
    const std::size_t steps = static_cast<std::size_t>(bytes) / (nnz * sizeof(double));
    ASSERT_GE(steps, 1u);
    std::ifstream in(vals_path, std::ios::binary);
    in.seekg(static_cast<std::streamoff>((steps - 1) * nnz * sizeof(double)));
    in.read(reinterpret_cast<char *>(A.values.data()), static_cast<std::streamsize>(nnz * sizeof(double)));
    ASSERT_TRUE(static_cast<bool>(in));
    const int n = static_cast<int>(A.n);
    const std::vector<double> b = rhs_of(n);
    Solver s;

    Control t = indef_control();
    Info ia;
    auto ak = s.analyse(A, t, ia);
    ASSERT_NE(ak.get(), nullptr);
    FactorKeep ft;
    Info it;
    ASSERT_EQ(s.factor(*ak, t, it, ft), FactorStatus::Success);
    std::printf("[FastMode] pde_control_120 threshold: %d delays, %d rescales, neg %d\n", it.delayed_pivots,
                it.delay_rescales, it.num_negative);

    Control c = t;
    c.pivoting = Pivoting::Static;
    FactorKeep fk;
    Info i;
    ASSERT_EQ(s.factor(*ak, c, i, fk), FactorStatus::Success);
    std::printf("[FastMode] pde_control_120 static: %d delays, %d perturbed, growth %.3g, neg %d\n",
                i.delayed_pivots, i.static_pivots, i.pivot_growth, i.num_negative);
    EXPECT_EQ(i.delayed_pivots, 0);
    EXPECT_EQ(i.static_pivots, 0);
    EXPECT_TRUE(i.inertia_exact);
    EXPECT_EQ(i.num_positive, it.num_positive);
    EXPECT_EQ(i.num_negative, it.num_negative);
    EXPECT_EQ(i.num_zero, it.num_zero);
    std::vector<double> x = b;
    Info si;
    ASSERT_EQ(s.solve(fk, c, si, x.data(), n, 1), 0);
    SolveWorkspace ws;
    const RefinementResult rr = refine(A, fk, c, b, x, 1, ws);
    std::printf("[FastMode] pde_control_120 refine: omega %.3g -> %.3g in %d steps (method %d)\n",
                rr.omega_start, rr.omega, rr.iterations, static_cast<int>(rr.method_used));
    EXPECT_EQ(rr.status, ErrorCode::Success);
    const auto val = smf::bench::validate(A, b, x, 1);
    EXPECT_TRUE(smf::bench::is_qualified_validation(val)) << "eta " << val.eta_inf << " omega " << val.omega;
}

// --- Refinement: Auto does not escalate an already-good static factor ---

namespace {

// [e 1 0; 1 1 1; 0 1 1], e = 2^-40, under the identity order with no
// amalgamation: column 0 is a front of its own with no acceptable pivot,
// so the static path perturbs it (raised to 1e-8 * max|A|). Every entry and
// A * (1, 2, 3) = (2 + 2^-40, 6, 5) are exact in binary64.
CscLower exact_tridiag() {
    CscLower A;
    A.n = 3;
    A.col_ptr = {0, 2, 4, 5};
    A.row_idx = {0, 1, 1, 2, 2};
    A.values = {std::ldexp(1.0, -40), 1.0, 1.0, 1.0, 1.0};
    return A;
}

Control exact_tridiag_control(const Int *perm) {
    Control c = indef_control();
    c.ordering = OrderingMethod::UserSupplied;
    c.user_perm = perm;
    c.nemin = 1;
    c.amalgamation = Amalgamation::Smf;
    c.scaling = ScalingMethod::None;
    c.pivoting = Pivoting::Static;
    return c;
}

} // namespace

// Open item from PR #45: under RefineMethod::Auto, a static-path factor whose
// growth forces a step from an iterate that already meets the omega target
// used to escalate to FGMRES when that step did not improve it. From the
// exact solution the forced step's correction is exactly zero: Auto now
// stops there (one classical step, no FGMRES), as Classical does.
TEST(FastMode, AutoDoesNotEscalateOnAnAlreadyGoodStaticFactor) {
    const CscLower A = exact_tridiag();
    const std::vector<Int> perm = {0, 1, 2};
    const Control ctrl = exact_tridiag_control(perm.data());
    Solver s;
    Info ia;
    auto ak = s.analyse(A, ctrl, ia);
    ASSERT_NE(ak.get(), nullptr);
    FactorKeep fk;
    Info i;
    ASSERT_EQ(s.factor(*ak, ctrl, i, fk), FactorStatus::Success);
    ASSERT_TRUE(fk.static_path);
    EXPECT_EQ(i.static_pivots, 1);
    const std::vector<double> xt = {1.0, 2.0, 3.0};
    const std::vector<double> b = {2.0 + std::ldexp(1.0, -40), 6.0, 5.0};
    for (const RefineMethod m : {RefineMethod::Auto, RefineMethod::Classical}) {
        Control c = ctrl;
        c.refine_method = m;
        c.refine_growth_threshold = -1.0;  // force the growth step
        std::vector<double> x = xt;
        SolveWorkspace ws;
        const RefinementResult rr = refine(A, fk, c, b, x, 1, ws);
        std::printf("[FastMode] exact start, method %d: %d steps, used %d, omega %.3g -> %.3g\n",
                    static_cast<int>(m), rr.iterations, static_cast<int>(rr.method_used), rr.omega_start,
                    rr.omega);
        EXPECT_EQ(rr.omega_start, 0.0);
        EXPECT_EQ(rr.status, ErrorCode::Success);
        EXPECT_EQ(rr.iterations, 1) << "the forced step only";
        EXPECT_EQ(rr.method_used, RefineMethod::Classical) << "method " << static_cast<int>(m);
        EXPECT_EQ(x, xt);
    }
    // Not already good: Auto still refines (and may escalate) from a bad
    // start, and ends at the solution.
    {
        Control c = ctrl;
        std::vector<double> x = b;
        Info si;
        ASSERT_EQ(s.solve(fk, c, si, x.data(), 3, 1), 0);
        SolveWorkspace ws;
        const RefinementResult rr = refine(A, fk, c, b, x, 1, ws);
        EXPECT_EQ(rr.status, ErrorCode::Success);
        EXPECT_GE(rr.iterations, 1);
        for (int k = 0; k < 3; ++k) EXPECT_NEAR(x[static_cast<std::size_t>(k)], xt[static_cast<std::size_t>(k)], 1e-12);
    }
}

// The same on a realistic factor: a high-growth static factor (pivot_u 0.4)
// whose first solve already meets the accuracy gate (asserted). The growth
// threshold forces a step; whether it improves the iterate or not, Auto now
// never escalates from there.
TEST(FastMode, AutoForcedStepFromConvergedStartStaysClassical) {
    const CscLower E = cascade_kkt(40, 600, 4.1, 0.0);
    Control ctrl = indef_control();
    ctrl.amalgamation = Amalgamation::Smf;
    ctrl.pivoting = Pivoting::Static;
    ctrl.pivot_u = 0.4;
    Solver s;
    Info ia;
    auto ak = s.analyse(E, ctrl, ia);
    ASSERT_NE(ak.get(), nullptr);
    FactorKeep fk;
    Info i;
    ASSERT_EQ(s.factor(*ak, ctrl, i, fk), FactorStatus::Success);
    ASSERT_TRUE(fk.static_path);
    const int n = static_cast<int>(E.n);
    const std::vector<double> b = rhs_of(n);
    std::vector<double> x0 = b;
    Info si;
    ASSERT_EQ(s.solve(fk, ctrl, si, x0.data(), n, 1), 0);
    const auto v0 = smf::bench::validate(E, b, x0, 1);
    ASSERT_TRUE(v0.eta_inf <= 1e-10 && v0.omega <= 1e-8)
        << "premise: the first solve meets the gate (eta " << v0.eta_inf << ", omega " << v0.omega << ")";
    Control c = ctrl;
    c.refine_method = RefineMethod::Auto;
    c.refine_omega_target = 1e-8;  // the gate: met before any step
    c.refine_growth_threshold = 1.0;
    ASSERT_GT(fk.pivot_growth, c.refine_growth_threshold);
    std::vector<double> x = x0;
    SolveWorkspace ws;
    const RefinementResult rr = refine(E, fk, c, b, x, 1, ws);
    EXPECT_EQ(rr.iterations, 1);
    EXPECT_EQ(rr.method_used, RefineMethod::Classical);
    EXPECT_EQ(rr.status, ErrorCode::Success);
    EXPECT_LE(rr.omega, rr.omega_start);
}

// --- Static path under tree_parallel ---

// Pivoting::Static with tree_parallel at num_threads > 1 runs the default
// subtree engine (static_path_control), on a cascading KKT: the same pivots,
// inertia and delays (none) as without tree_parallel, and the static path
// reported truthfully. Through IpmSession too.
TEST(FastMode, StaticUnderTreeParallelIsRouted) {
    const CscLower A = cascading_kkt();
    Control c = indef_control();
    c.amalgamation = Amalgamation::Smf;
    c.pivoting = Pivoting::Static;
    c.num_threads = 2;
    c.auto_threads = false;
    if (deterministic_mode(c)) GTEST_SKIP() << "deterministic mode ignores tree_parallel";
    Solver s;
    Info ia;
    auto ak = s.analyse(A, c, ia);
    ASSERT_NE(ak.get(), nullptr);
    Info ir;
    FactorKeep fr;
    ASSERT_EQ(s.factor(*ak, c, ir, fr), FactorStatus::Success);
    Control tp = c;
    tp.tree_parallel = true;
    Info it;
    FactorKeep ft;
    ASSERT_EQ(s.factor(*ak, tp, it, ft), FactorStatus::Success);
    std::printf("[FastMode] tree_parallel static: %d delays, %d perturbed (default engine %d / %d)\n",
                it.delayed_pivots, it.static_pivots, ir.delayed_pivots, ir.static_pivots);
    EXPECT_EQ(it.delayed_pivots, 0);
    EXPECT_EQ(it.static_pivots, ir.static_pivots);
    EXPECT_EQ(it.static_fallback, 1);
    EXPECT_TRUE(ft.static_path);
    EXPECT_EQ(it.num_positive, ir.num_positive);
    EXPECT_EQ(it.num_negative, ir.num_negative);
    EXPECT_EQ(it.num_zero, ir.num_zero);
    // The threshold path under tree_parallel delays on this matrix: the
    // routing is what keeps the static run delay-free.
    {
        Control th = tp;
        th.pivoting = Pivoting::Threshold;
        Info i2;
        FactorKeep f2;
        ASSERT_EQ(s.factor(*ak, th, i2, f2), FactorStatus::Success);
        EXPECT_GT(i2.delayed_pivots, 0);
        EXPECT_FALSE(f2.static_path);
    }
    // IpmSession with the same Control.
    IpmSession sess;
    ASSERT_EQ(sess.analyse(A, A.values, tp, IpmControl{}), ErrorCode::Success);
    ASSERT_EQ(sess.factor(A.values), IpmFactorStatus::Ok);
    EXPECT_TRUE(sess.factor_keep().static_path);
    EXPECT_EQ(sess.info().delayed_pivots, 0);
    EXPECT_EQ(sess.inertia().negative, ir.num_negative);
}

// --- Raised exact zeros, refinement only after a forced pivot ---

// A zero-diagonal KKT (the PGLib / COPS shape: opf_like_kkt, unscaled, the
// constraint rows ordered first with nemin 1, so that each zero diagonal is
// a front's only eligible column) raises exact-zero pivots in fast mode. They count by the sign
// of the perturbed value (+), as PARDISO counts them: the factor is Success
// with no zero eigenvalue instead of Singular, its inertia is that of the
// perturbed matrix (inexact), and the refined solve meets the omega target.
// Threshold pivoting of the same matrix is the reference inertia.
TEST(FastMode, RaisedExactZerosCountBySign) {
    const CscLower A = opf_like_kkt(300, 120, 7u);
    const int n = static_cast<int>(A.n);
    const int n_x = 300;
    std::vector<Int> perm;
    for (int k = n_x; k < n; ++k) perm.push_back(k);
    for (int k = 0; k < n_x; ++k) perm.push_back(k);
    Control base = indef_control();
    base.ordering = OrderingMethod::UserSupplied;
    base.user_perm = perm.data();
    base.nemin = 1;
    base.scaling = ScalingMethod::None;
    Solver s;

    Info it;
    {
        Info ia;
        auto ak = s.analyse(A, base, ia);
        ASSERT_NE(ak.get(), nullptr);
        FactorKeep fk;
        ASSERT_EQ(s.factor(*ak, base, it, fk), FactorStatus::Success);
        ASSERT_EQ(it.num_zero, 0);
    }

    Control ctrl = base;
    ctrl.pivoting = Pivoting::Static;
    Info ia;
    auto ak = s.analyse(A, ctrl, ia);
    ASSERT_NE(ak.get(), nullptr);
    FactorKeep fk;
    Info i;
    ASSERT_EQ(s.factor(*ak, ctrl, i, fk), FactorStatus::Success);
    std::printf("[FastMode] zero-diagonal KKT: %d perturbed, growth %.3g, inertia (%d, %d, %d) vs "
                "threshold (%d, %d, %d)\n",
                i.static_pivots, i.pivot_growth, i.num_positive, i.num_negative, i.num_zero,
                it.num_positive, it.num_negative, it.num_zero);
    ASSERT_GT(i.static_pivots, 0) << "premise: exact zeros raised";
    EXPECT_EQ(i.num_zero, 0);
    EXPECT_FALSE(i.inertia_exact);
    EXPECT_EQ(i.num_positive + i.num_negative, n);
    EXPECT_TRUE(fk.static_path);
    EXPECT_TRUE(refinement_advised(fk));

    const std::vector<double> b = rhs_of(n);
    std::vector<double> x = b;
    Info si;
    ASSERT_EQ(s.solve(fk, ctrl, si, x.data(), n, 1), 0);
    SolveWorkspace ws;
    const RefinementResult rr = refine(A, fk, ctrl, b, x, 1, ws);
    std::printf("[FastMode] zero-diagonal KKT refine: omega %.3g -> %.3g in %d steps\n",
                rr.omega_start, rr.omega, rr.iterations);
    EXPECT_EQ(rr.status, ErrorCode::Success);
    EXPECT_LE(rr.omega, ctrl.refine_omega_target);

    // The plugin's path: IpmSession reports Ok (not Singular), with the
    // inexact inertia that the plugin turns into MA97 flag 7.
    IpmSession sess;
    ASSERT_EQ(sess.analyse(A, A.values, ctrl, IpmControl{}), ErrorCode::Success);
    ASSERT_EQ(sess.factor(A.values), IpmFactorStatus::Ok);
    EXPECT_EQ(sess.inertia().zero, 0);
    std::vector<double> y = b;
    ASSERT_EQ(sess.solve(y.data(), n, 1), 0);
    const auto val = smf::bench::validate(A, b, y, 1);
    EXPECT_LE(val.omega, 10.0 * ctrl.refine_omega_target);
}

// A static factor that forced no pivot (every pivot passed pivot_u: a
// diagonally dominant indefinite matrix) is a threshold factorization:
// static_path, growth 0, nothing perturbed, and no refinement advised. An
// IpmSession full solve with it is the plain solve of its factor, bitwise.
TEST(FastMode, NoForcedPivotNoRefinement) {
    const int n = 60;
    CscLower A;
    A.n = n;
    A.col_ptr.assign(static_cast<std::size_t>(n) + 1, 0);
    for (int j = 0; j < n; ++j) {
        A.row_idx.push_back(j);
        A.values.push_back(j % 2 == 0 ? 4.0 : -4.0);
        if (j + 1 < n) {
            A.row_idx.push_back(j + 1);
            A.values.push_back(1.0);
        }
        A.col_ptr[static_cast<std::size_t>(j) + 1] = static_cast<Int>(A.row_idx.size());
    }
    Control ctrl = indef_control();
    ctrl.pivoting = Pivoting::Static;
    Solver s;
    Info ia;
    auto ak = s.analyse(A, ctrl, ia);
    ASSERT_NE(ak.get(), nullptr);
    FactorKeep fk;
    Info i;
    ASSERT_EQ(s.factor(*ak, ctrl, i, fk), FactorStatus::Success);
    EXPECT_TRUE(fk.static_path);
    EXPECT_EQ(i.static_pivots, 0);
    EXPECT_EQ(fk.pivot_growth, 0.0);
    EXPECT_TRUE(i.inertia_exact);
    EXPECT_FALSE(refinement_advised(fk));

    const std::vector<double> b = rhs_of(n);
    IpmSession sess;
    ASSERT_EQ(sess.analyse(A, A.values, ctrl, IpmControl{}), ErrorCode::Success);
    ASSERT_EQ(sess.factor(A.values), IpmFactorStatus::Ok);
    EXPECT_TRUE(sess.factor_keep().static_path);
    EXPECT_FALSE(refinement_advised(sess.factor_keep()));
    // The session's own factor, solved plainly: what an unrefined full
    // solve gives.
    std::vector<double> x = b;
    Info si;
    ASSERT_EQ(s.solve(sess.factor_keep(), ctrl, si, x.data(), n, 1), 0);
    std::vector<double> y = b;
    ASSERT_EQ(sess.solve(y.data(), n, 1), 0);
    EXPECT_EQ(x, y) << "not refined";
}

// --- The default path ---

// A default Control, one with Threshold set explicitly and one parsed from
// "threshold" factor and solve a cascading sequence bitwise alike (the
// rescue, no static path, nothing refined beyond the usual).
TEST(FastMode, ThresholdIsBitwiseTheDefault) {
    const CscLower A = cascade_kkt(40, 600, 4.1, 3.0);
    const CscLower A1 = cascade_kkt(40, 600, 4.15, 3.0);
    Control def = indef_control();
    Control expl = def;
    expl.pivoting = Pivoting::Threshold;
    Control parsed = def;
    parsed.pivoting = Pivoting::Static;
    ASSERT_TRUE(parse_pivoting("threshold", parsed.pivoting));
    std::vector<std::vector<double>> ref;
    for (const Control &ctrl : {def, expl, parsed}) {
        Solver s;
        Info ia;
        auto ak = s.analyse(A, ctrl, ia);
        ASSERT_NE(ak.get(), nullptr);
        NumericWorkspace ws;
        FactorKeep fk;
        std::vector<std::vector<double>> got;
        for (const CscLower *M : {&A, &A1, &A1}) {
            Info i;
            ASSERT_EQ(s.factor(*ak, *M, ctrl, i, fk, ws), FactorStatus::Success);
            EXPECT_EQ(i.static_fallback, 0);
            EXPECT_FALSE(fk.static_path);
            got.push_back(fk.factor_values);
            std::vector<double> x = rhs_of(static_cast<int>(M->n));
            Info si;
            ASSERT_EQ(s.solve(fk, ctrl, si, x.data(), static_cast<int>(M->n), 1), 0);
            got.push_back(x);
        }
        if (ref.empty()) ref = got;
        else EXPECT_EQ(got, ref);
    }
}
