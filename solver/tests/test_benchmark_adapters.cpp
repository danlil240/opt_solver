#include "../benchmarks/adapters/adapter.hpp"
#include "../benchmarks/common/replay_refine.hpp"
#include "../benchmarks/common/replay_steps.hpp"
#include "../benchmarks/common/validation.hpp"

#include <gtest/gtest.h>

#include <vector>

using namespace smf;
using namespace smf::bench;

namespace {

CscLower make_identity_spd_3x3() {
    CscLower A;
    A.n = 3;
    A.col_ptr = {0, 1, 2, 3};
    A.row_idx = {0, 1, 2};
    A.values = {1.0, 1.0, 1.0};
    return A;
}

CscLower make_exchange_2x2_indef() {
    CscLower A;
    A.n = 2;
    A.col_ptr = {0, 2, 3};
    A.row_idx = {0, 1, 1};
    A.values = {0.0, 1.0, 0.0};
    return A;
}

CscLower make_singular_2x2() {
    CscLower A;
    A.n = 2;
    A.col_ptr = {0, 1, 2};
    A.row_idx = {0, 1};
    A.values = {2.0, 0.0};
    return A;
}

} // namespace

TEST(BenchmarkAdapters, SmfAdapterAndAllRegisteredAdapters) {
    auto ids = all_registered_adapter_ids();
    ASSERT_FALSE(ids.empty());

    for (const auto &id : ids) {
        SCOPED_TRACE("adapter=" + id);
        auto adapter = make_solver_adapter(id);
        ASSERT_NE(adapter.get(), nullptr) << "Null adapter for " << id;
        AdapterIdentity ident = adapter->identity();
        EXPECT_EQ(ident.solver_id, id);
        EXPECT_FALSE(ident.integer_abi.empty());
        // Never label libsmf_ma97 as genuine HSL ma97!
        if (id == "ma97") {
            EXPECT_EQ(ident.library_path.find("smf_ma97"), std::string::npos);
        }

        if (!ident.available) {
            CscLower I3 = make_identity_spd_3x3();
            EXPECT_EQ(adapter->analyse(I3), Outcome::Unavailable);
            EXPECT_EQ(adapter->factor(I3.values), Outcome::Unavailable);
            std::vector<double> b = {1.0, 2.0, 3.0};
            EXPECT_EQ(adapter->solve(b, 1), Outcome::Unavailable);
            continue;
        }

        // Test 1: Identity SPD
        if (ident.supports_spd) {
            Control ctrl;
            ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
            adapter->configure(ctrl, "recommended");
            CscLower I3 = make_identity_spd_3x3();
            ASSERT_EQ(adapter->analyse(I3), Outcome::Success);
            ASSERT_EQ(adapter->factor(I3.values), Outcome::Success);
            std::vector<double> b = {1.0, 2.0, 3.0};
            std::vector<double> x = b;
            ASSERT_EQ(adapter->solve(x, 1), Outcome::Success);
            auto v = validate(I3, b, x, 1);
            EXPECT_TRUE(is_qualified_validation(v));

            // Changed-value replay on the same pattern: diag(2, 4, 8)
            std::vector<double> updated_vals = {2.0, 4.0, 8.0};
            ASSERT_EQ(adapter->factor(updated_vals), Outcome::Success);
            // Multi-RHS (nrhs = 2)
            std::vector<double> b2 = {2.0, 8.0, 24.0, 4.0, 12.0, 16.0};
            std::vector<double> x2 = b2;
            ASSERT_EQ(adapter->solve(x2, 2), Outcome::Success);
            CscLower A_upd = I3;
            A_upd.values = updated_vals;
            auto v2 = validate(A_upd, b2, x2, 2);
            EXPECT_TRUE(is_qualified_validation(v2));
        }

        // Test 2: 2x2 exchange matrix where indefinite is supported
        if (ident.supports_indefinite) {
            Control ctrl;
            ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
            // cuDSS has no 2x2 pivots (static pivoting only) and, as on the
            // recommended track, does no refinement; its exchange-matrix solve
            // is only exact with its own iterative refinement (ipopt track).
            adapter->configure(ctrl, id == "cudss" ? "ipopt" : "recommended");
            CscLower E2 = make_exchange_2x2_indef();
            ASSERT_EQ(adapter->analyse(E2), Outcome::Success);
            ASSERT_EQ(adapter->factor(E2.values), Outcome::Success);
            std::vector<double> b = {1.0, 2.0};
            std::vector<double> x = b;
            ASSERT_EQ(adapter->solve(x, 1), Outcome::Success);
            EXPECT_NEAR(x[0], 2.0, 1e-12);
            EXPECT_NEAR(x[1], 1.0, 1e-12);
            AdapterStats st = adapter->stats();
            EXPECT_EQ(st.inertia_pos, 1);
            EXPECT_EQ(st.inertia_neg, 1);
            EXPECT_EQ(st.inertia_zero, 0);
        }

        // Test 3: Singularity must never produce a falsely successful solve.
        // Factor-time detection (MUMPS/CHOLMOD/smf) is preferred, but solvers
        // that internally regularize (e.g. MKL PARDISO pivot perturbation) may
        // report factor Success; in that case the solution must fail the shared
        // backward-error qualification on the original system.
        {
            Control ctrl;
            ctrl.matrix_type = ident.supports_indefinite
                                   ? MatrixType::RealSymmetricIndefinite
                                   : MatrixType::RealSymmetricPositiveDefinite;
            adapter->configure(ctrl, "recommended");
            CscLower S2 = make_singular_2x2();
            ASSERT_EQ(adapter->analyse(S2), Outcome::Success);
            Outcome f_out = adapter->factor(S2.values);
            if (f_out == Outcome::Success) {
                std::vector<double> b = {1.0, 1.0};
                std::vector<double> x = b;
                Outcome s_out = adapter->solve(x, 1);
                if (s_out == Outcome::Success) {
                    auto v = validate(S2, b, x, 1);
                    EXPECT_FALSE(is_qualified_validation(v))
                        << "singular solve passed qualification for " << id;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// bench_replay's IPM option: the smf adapter routed through smf::IpmSession
// with a static mode (--ipm-static off|fallback|regularized,
// --ipm-static-zero), and the replayed steps of a recorded sequence
// (--seq-steps N|all).

namespace {

// diag(1, d, 0.5) with no scaling: max|A| = 1, so Regularized's rho = 1e-8.
CscLower make_tiny_diag(double d) {
    CscLower A;
    A.n = 3;
    A.col_ptr = {0, 1, 2, 3};
    A.row_idx = {0, 1, 2};
    A.values = {1.0, d, 0.5};
    return A;
}

Control replay_control() {
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    ctrl.scaling = ScalingMethod::None;
    ctrl.pivot_u = 1e-8;
    return ctrl;
}

} // namespace

TEST(BenchmarkAdapters, ParseIpmStatic) {
    EXPECT_EQ(parse_ipm_static("off"), StaticMode::Off);
    EXPECT_EQ(parse_ipm_static("fallback"), StaticMode::Fallback);
    EXPECT_EQ(parse_ipm_static("regularized"), StaticMode::Regularized);
    EXPECT_FALSE(parse_ipm_static("").has_value());
    EXPECT_FALSE(parse_ipm_static("reg").has_value());
}

TEST(BenchmarkAdapters, ReplayStepPicks) {
    // 5 evenly spaced steps (the default), first to last.
    EXPECT_EQ(replay_step_picks(9, 5), (std::vector<std::size_t>{0, 2, 4, 6, 8}));
    EXPECT_EQ(replay_step_picks(1, 5), (std::vector<std::size_t>{0, 0, 0, 0, 0}));
    // Every stored step.
    EXPECT_EQ(replay_step_picks(4, -1), (std::vector<std::size_t>{0, 1, 2, 3}));
    EXPECT_TRUE(replay_step_picks(0, -1).empty());
    EXPECT_TRUE(replay_step_picks(0, 5).empty());
}

TEST(BenchmarkAdapters, SmfIpmAdapterOffMatchesPlainSmf) {
    auto ipm = make_smf_ipm_adapter(StaticMode::Off, false);
    ASSERT_NE(ipm.get(), nullptr);
    EXPECT_EQ(ipm->identity().solver_id, "smf");
    ipm->configure(replay_control(), "ipopt");
    CscLower E2 = make_exchange_2x2_indef();
    ASSERT_EQ(ipm->analyse(E2), Outcome::Success);
    ASSERT_EQ(ipm->factor(E2.values), Outcome::Success);
    std::vector<double> x = {1.0, 2.0};
    ASSERT_EQ(ipm->solve(x, 1), Outcome::Success);
    EXPECT_NEAR(x[0], 2.0, 1e-12);
    EXPECT_NEAR(x[1], 1.0, 1e-12);
    EXPECT_EQ(ipm->stats().inertia_pos, 1);
    EXPECT_EQ(ipm->stats().inertia_neg, 1);
    EXPECT_EQ(ipm->stats().static_pivots, 0);
}

TEST(BenchmarkAdapters, SmfIpmAdapterRegularizedRaisesTinyPivot) {
    // A primal row with diagonal -rho/2: regularized to +rho/2 < rho, raised
    // to rho (one static pivot). Hybrid accounting keeps its sign; with
    // static-zero it is a zero eigenvalue (singular factor).
    const CscLower T = make_tiny_diag(-0.5e-8);

    auto plain = make_smf_adapter();
    plain->configure(replay_control(), "ipopt");
    ASSERT_EQ(plain->analyse(T), Outcome::Success);
    ASSERT_EQ(plain->factor(T.values), Outcome::Success);
    EXPECT_EQ(plain->stats().static_pivots, 0);
    EXPECT_EQ(plain->stats().inertia_neg, 1);

    auto hybrid = make_smf_ipm_adapter(StaticMode::Regularized, false);
    hybrid->configure(replay_control(), "ipopt");
    ASSERT_EQ(hybrid->analyse(T), Outcome::Success);
    ASSERT_EQ(hybrid->factor(T.values), Outcome::Success);
    EXPECT_EQ(hybrid->stats().static_pivots, 1);
    EXPECT_EQ(hybrid->stats().inertia_pos, 3);
    EXPECT_EQ(hybrid->stats().inertia_neg, 0);
    EXPECT_EQ(hybrid->stats().inertia_zero, 0);

    auto zero = make_smf_ipm_adapter(StaticMode::Regularized, true);
    zero->configure(replay_control(), "ipopt");
    ASSERT_EQ(zero->analyse(T), Outcome::Success);
    EXPECT_EQ(zero->factor(T.values), Outcome::Singular);
    EXPECT_EQ(zero->stats().static_pivots, 1);
    EXPECT_EQ(zero->stats().inertia_zero, 1);
}

TEST(BenchmarkAdapters, SmfIpmAdapterRegularizedSolveIsRefined) {
    // An indefinite KKT-like matrix (zero constraint diagonal): the factor is
    // of the regularized matrix, the solve is refined against the original.
    CscLower K;
    K.n = 3;
    K.col_ptr = {0, 3, 5, 6};
    K.row_idx = {0, 1, 2, 1, 2, 2};
    K.values = {4.0, 1.0, 1.0, 3.0, 1.0, 0.0};
    auto reg = make_smf_ipm_adapter(StaticMode::Regularized, false);
    reg->configure(replay_control(), "ipopt");
    ASSERT_EQ(reg->analyse(K), Outcome::Success);
    for (int f = 0; f < 2; ++f) {
        ASSERT_EQ(reg->factor(K.values), Outcome::Success);
        const std::vector<double> b = {1.0, -2.0, 0.5};
        std::vector<double> x = b;
        ASSERT_EQ(reg->solve(x, 1), Outcome::Success);
        EXPECT_TRUE(is_qualified_validation(validate(K, b, x, 1))) << "factor " << f;
        EXPECT_EQ(reg->stats().inertia_neg, 1);
    }
}

TEST(BenchmarkAdapters, RecordCarriesFirstFactorAndRefactorOnlyWhenMeasured) {
    RunRecord rec;
    EXPECT_EQ(rec.to_json_line().find("first_factor_ms"), std::string::npos);
    EXPECT_EQ(rec.to_json_line().find("\"refactor_ms\""), std::string::npos);
    rec.first_factor_ms = 12.5;
    rec.refactor_ms = 3.25;
    const std::string line = rec.to_json_line();
    EXPECT_NE(line.find("\"first_factor_ms\":12.5"), std::string::npos) << line;
    EXPECT_NE(line.find("\"refactor_ms\":3.25"), std::string::npos) << line;
}

// R29: the cold path on the ipopt track refines like warm_sequence does. A
// solver whose solve carries a 1e-6 relative error misses the accuracy gate
// unrefined; harness refinement (ipopt track only, nrhs == 1) recovers it.
TEST(BenchmarkAdapters, ColdIpoptTrackRefinesRecommendedDoesNot) {
    CscLower A;
    const int n = 200;
    A.n = n;
    A.col_ptr.assign(static_cast<std::size_t>(n) + 1, 0);
    for (int j = 0; j < n; ++j) {
        A.row_idx.push_back(j);
        A.values.push_back(4.0);
        if (j + 1 < n) {
            A.row_idx.push_back(j + 1);
            A.values.push_back(-1.0);
        }
        A.col_ptr[static_cast<std::size_t>(j) + 1] = static_cast<Int>(A.row_idx.size());
    }
    const std::vector<double> b(static_cast<std::size_t>(n), 1.0);
    // Exact tridiagonal solve (Thomas) times (1 + 1e-6): an inaccurate "factorization".
    auto inexact_solve = [&](std::vector<double> &r) {
        std::vector<double> c(n), d(r);
        double m = 4.0;
        c[0] = -1.0 / m;
        d[0] /= m;
        for (int i = 1; i < n; ++i) {
            m = 4.0 + c[i - 1];
            c[i] = -1.0 / m;
            d[i] = (d[i] + d[i - 1]) / m;
        }
        for (int i = n - 2; i >= 0; --i) d[i] -= c[i] * d[i + 1];
        for (int i = 0; i < n; ++i) r[i] = d[i] * (1.0 + 1e-6);
        return true;
    };

    std::vector<double> x0 = b;
    inexact_solve(x0);
    ASSERT_FALSE(is_qualified_validation(validate(A, b, x0, 1)));

    std::vector<double> xi = x0;
    const double ms = refine_timed("ipopt", 1, inexact_solve, A, A.values, b, xi, 3);
    EXPECT_GT(ms, 0.0);
    EXPECT_TRUE(is_qualified_validation(validate(A, b, xi, 1)));

    std::vector<double> xr = x0;
    EXPECT_EQ(refine_timed("recommended", 1, inexact_solve, A, A.values, b, xr, 3), 0.0);
    EXPECT_EQ(xr, x0);
    std::vector<double> xm = x0;
    EXPECT_EQ(refine_timed("ipopt", 2, inexact_solve, A, A.values, b, xm, 3), 0.0);
    EXPECT_EQ(xm, x0);
}

// ---------------------------------------------------------------------------
// cuDSS (GPU reference solver), built only with -DSMF_BENCH_CUDSS=ON.

#if defined(SMF_BENCH_CUDSS)

namespace {

// 5-point 2D Poisson on an m x m grid, lower triangle in CSC.
CscLower make_poisson2d(int m) {
    CscLower A;
    A.n = m * m;
    A.col_ptr.assign(static_cast<std::size_t>(A.n) + 1, 0);
    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < m; ++i) {
            const int col = j * m + i;
            A.row_idx.push_back(col);
            A.values.push_back(4.0);
            if (i + 1 < m) {
                A.row_idx.push_back(col + 1);
                A.values.push_back(-1.0);
            }
            if (j + 1 < m) {
                A.row_idx.push_back(col + m);
                A.values.push_back(-1.0);
            }
            A.col_ptr[static_cast<std::size_t>(col) + 1] =
                static_cast<Int>(A.row_idx.size());
        }
    }
    return A;
}

} // namespace

TEST(BenchmarkAdapters, CudssSolvesPoisson) {
    auto adapter = make_solver_adapter("cudss");
    ASSERT_NE(adapter.get(), nullptr) << "cudss is not a registered adapter";
    const AdapterIdentity ident = adapter->identity();
    EXPECT_EQ(ident.solver_id, "cudss");
    if (!ident.available) GTEST_SKIP() << "no usable CUDA device: " << adapter->stats().reason;

    const CscLower A = make_poisson2d(30);
    std::vector<double> b(static_cast<std::size_t>(A.n));
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = 1.0 + 0.001 * static_cast<double>(i % 17);

    // Both matrix types: SPD (Cholesky) and symmetric indefinite (LDL^T).
    for (MatrixType mt : {MatrixType::RealSymmetricPositiveDefinite,
                          MatrixType::RealSymmetricIndefinite}) {
        SCOPED_TRACE(mt == MatrixType::RealSymmetricIndefinite ? "ldlt" : "spd");
        Control ctrl;
        ctrl.matrix_type = mt;
        adapter->configure(ctrl, "recommended");
        ASSERT_EQ(adapter->analyse(A), Outcome::Success) << adapter->stats().reason;
        ASSERT_EQ(adapter->factor(A.values), Outcome::Success) << adapter->stats().reason;
        std::vector<double> x = b;
        ASSERT_EQ(adapter->solve(x, 1), Outcome::Success) << adapter->stats().reason;
        const Validation v = validate(A, b, x, 1);
        EXPECT_LE(v.eta_inf, 1e-12);
        EXPECT_TRUE(is_qualified_validation(v));
        EXPECT_GT(adapter->stats().factor_ms, 0.0);

        // Refactor with changed values on the same analysis (diag scaled by 1.5).
        CscLower A2 = A;
        for (Int j = 0; j < A2.n; ++j) {
            A2.values[static_cast<std::size_t>(A2.col_ptr[static_cast<std::size_t>(j)])] *= 1.5;
        }
        ASSERT_EQ(adapter->factor(A2.values), Outcome::Success);
        std::vector<double> x2 = b;
        ASSERT_EQ(adapter->solve(x2, 1), Outcome::Success);
        EXPECT_LE(validate(A2, b, x2, 1).eta_inf, 1e-12);

        // ipopt track, one rhs: cuDSS iterative refinement (3 steps) runs inside solve.
        adapter->configure(ctrl, "ipopt");
        ASSERT_EQ(adapter->analyse(A), Outcome::Success);
        ASSERT_EQ(adapter->factor(A.values), Outcome::Success);
        std::vector<double> x3 = b;
        ASSERT_EQ(adapter->solve(x3, 1), Outcome::Success);
        EXPECT_LE(validate(A, b, x3, 1).eta_inf, 1e-12);
    }
}

#endif // SMF_BENCH_CUDSS
