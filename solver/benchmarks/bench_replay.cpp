#include "adapters/adapter.hpp"
#include "common/result.hpp"
#include "common/validation.hpp"
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/matrix_market.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

smf::CscLower make_tridiag_spd(int n) {
    smf::CscLower A;
    A.n = n;
    A.col_ptr.resize(static_cast<std::size_t>(n) + 1, 0);
    A.row_idx.reserve(static_cast<std::size_t>(2 * n));
    A.values.reserve(static_cast<std::size_t>(2 * n));
    for (int j = 0; j < n; ++j) {
        A.row_idx.push_back(j);
        A.values.push_back(4.0);
        if (j + 1 < n) {
            A.row_idx.push_back(j + 1);
            A.values.push_back(-1.0);
        }
        A.col_ptr[static_cast<std::size_t>(j + 1)] =
            static_cast<smf::Int>(A.row_idx.size());
    }
    return A;
}

smf::CscLower make_banded_spd(int n, int bw) {
    smf::CscLower A;
    A.n = n;
    A.col_ptr.resize(static_cast<std::size_t>(n) + 1, 0);
    for (int j = 0; j < n; ++j) {
        A.row_idx.push_back(j);
        A.values.push_back(static_cast<double>(2 * bw + 4));
        for (int d = 1; d <= bw && j + d < n; ++d) {
            A.row_idx.push_back(j + d);
            A.values.push_back(-1.0 / static_cast<double>(d));
        }
        A.col_ptr[static_cast<std::size_t>(j + 1)] =
            static_cast<smf::Int>(A.row_idx.size());
    }
    return A;
}

smf::CscLower make_poisson2d_spd(int grid) {
    const int n = grid * grid;
    smf::CscLower A;
    A.n = n;
    A.col_ptr.resize(static_cast<std::size_t>(n) + 1, 0);
    A.row_idx.reserve(static_cast<std::size_t>(3 * n));
    A.values.reserve(static_cast<std::size_t>(3 * n));
    for (int r = 0; r < grid; ++r) {
        for (int c = 0; c < grid; ++c) {
            const int j = r * grid + c;
            A.row_idx.push_back(j);
            A.values.push_back(4.0 + 1e-2);
            if (c + 1 < grid) {
                A.row_idx.push_back(j + 1);
                A.values.push_back(-1.0);
            }
            if (r + 1 < grid) {
                A.row_idx.push_back(j + grid);
                A.values.push_back(-1.0);
            }
            A.col_ptr[static_cast<std::size_t>(j + 1)] =
                static_cast<smf::Int>(A.row_idx.size());
        }
    }
    return A;
}

// Build a stage-coupled KKT saddle-point matrix for optimal control / MPC:
// Each stage t in [0, horizon) has primal state/control block of size `n_prim`
// and equality constraint block of size `n_dual` coupling stage t to t+1.
// Lower-triangular CSC with positive primal curvature, off-diagonal Jacobian
// coupling, and negative dual regularization (-1e-3).
smf::CscLower make_kkt_saddle_stage(int horizon, int n_prim, int n_dual, int variant) {
    const int stage_sz = n_prim + n_dual;
    const int n = horizon * stage_sz;
    smf::CscLower A;
    A.n = n;
    A.col_ptr.resize(static_cast<std::size_t>(n) + 1, 0);
    const double base_h = 4.0 + 0.25 * static_cast<double>(variant % 5);
    const double reg_d = -1e-2 - 2e-3 * static_cast<double>(variant % 4);

    for (int t = 0; t < horizon; ++t) {
        const int base = t * stage_sz;
        // Primal columns within stage t
        for (int p = 0; p < n_prim; ++p) {
            const int col = base + p;
            A.row_idx.push_back(col);
            A.values.push_back(base_h + 0.1 * static_cast<double>((p + t) % 4));
            if (p + 1 < n_prim) {
                A.row_idx.push_back(col + 1);
                A.values.push_back(-0.35);
            }
            // Jacobian coupling J_t: connect primal var p to dual constraint rows in stage t
            const int d_row0 = base + n_prim + (p % n_dual);
            A.row_idx.push_back(d_row0);
            A.values.push_back(1.0 + 0.05 * static_cast<double>(p));
            if (n_dual > 1) {
                const int d_row1 = base + n_prim + ((p + 1) % n_dual);
                if (d_row1 > d_row0) {
                    A.row_idx.push_back(d_row1);
                    A.values.push_back(-0.4);
                }
            }
            A.col_ptr[static_cast<std::size_t>(col + 1)] =
                static_cast<smf::Int>(A.row_idx.size());
        }
        // Dual columns within stage t (couple to next stage's primal variables too)
        for (int d = 0; d < n_dual; ++d) {
            const int col = base + n_prim + d;
            A.row_idx.push_back(col);
            A.values.push_back(reg_d);
            if (t + 1 < horizon) {
                const int next_prim = (t + 1) * stage_sz + (d % n_prim);
                A.row_idx.push_back(next_prim);
                A.values.push_back(-1.0);
            }
            A.col_ptr[static_cast<std::size_t>(col + 1)] =
                static_cast<smf::Int>(A.row_idx.size());
        }
    }
    return A;
}

smf::CscLower make_indef_banded(int n, int bw) {
    smf::CscLower A;
    A.n = n;
    A.col_ptr.resize(static_cast<std::size_t>(n) + 1, 0);
    for (int j = 0; j < n; ++j) {
        A.row_idx.push_back(j);
        const double sign = (j % 3 == 2) ? -1.0 : 1.0;
        A.values.push_back(sign * static_cast<double>(2 * bw + 5));
        for (int d = 1; d <= bw && j + d < n; ++d) {
            A.row_idx.push_back(j + d);
            A.values.push_back((d % 2 == 0 ? 0.5 : -0.8) / static_cast<double>(d));
        }
        A.col_ptr[static_cast<std::size_t>(j + 1)] =
            static_cast<smf::Int>(A.row_idx.size());
    }
    return A;
}

smf::CscLower make_builtin_fixture(const std::string &case_id, bool &is_spd) {
    is_spd = false;
    if (case_id == "smoke_spd_tridiag_10") {
        is_spd = true;
        return make_tridiag_spd(10);
    }
    if (case_id == "tridiag_500") {
        is_spd = true;
        return make_tridiag_spd(500);
    }
    if (case_id == "tridiag_2000") {
        is_spd = true;
        return make_tridiag_spd(2000);
    }
    if (case_id == "tridiag_5000") {
        is_spd = true;
        return make_tridiag_spd(5000);
    }
    if (case_id == "banded_spd_500_8") {
        is_spd = true;
        return make_banded_spd(500, 8);
    }
    if (case_id == "banded_spd_2000_10") {
        is_spd = true;
        return make_banded_spd(2000, 10);
    }
    if (case_id == "poisson2d_20") {
        is_spd = true;
        return make_poisson2d_spd(20);
    }
    if (case_id == "poisson2d_31") {
        is_spd = true;
        return make_poisson2d_spd(31);
    }
    if (case_id == "poisson2d_50") {
        is_spd = true;
        return make_poisson2d_spd(50);
    }
    if (case_id == "smoke_indef_exchange_2") {
        smf::CscLower A;
        A.n = 2;
        A.col_ptr = {0, 2, 3};
        A.row_idx = {0, 1, 1};
        A.values = {0.0, 1.0, 0.0};
        return A;
    }
    if (case_id == "indef_banded_600_6") {
        return make_indef_banded(600, 6);
    }
    if (case_id == "indef_banded_1500_8") {
        return make_indef_banded(1500, 8);
    }
    if (case_id == "kkt_ocp_lqr") {
        return make_kkt_saddle_stage(40, 6, 3, 0);
    }
    if (case_id == "kkt_vtol_intercept") {
        return make_kkt_saddle_stage(35, 8, 4, 1);
    }
    if (case_id == "kkt_robot_arm_mpc") {
        return make_kkt_saddle_stage(30, 8, 4, 2);
    }
    if (case_id == "kkt_ac_opf") {
        return make_kkt_saddle_stage(45, 6, 3, 3);
    }
    if (case_id == "kkt_contact_qp") {
        return make_kkt_saddle_stage(35, 7, 3, 4);
    }
    if (case_id == "kkt_chem_process") {
        return make_kkt_saddle_stage(40, 5, 3, 5);
    }
    if (case_id == "kkt_autonomous_driving_nmpc") {
        return make_kkt_saddle_stage(36, 6, 3, 6);
    }
    if (case_id == "kkt_quadrotor_swing") {
        return make_kkt_saddle_stage(32, 8, 4, 7);
    }
    if (case_id == "kkt_fixed_wing_guidance") {
        return make_kkt_saddle_stage(40, 6, 3, 8);
    }
    if (case_id == "kkt_spacecraft_rendezvous") {
        return make_kkt_saddle_stage(38, 7, 3, 9);
    }
    // Default synthetic KKT saddle-point fixture (4x4 indefinite)
    smf::CscLower A;
    A.n = 4;
    A.col_ptr = {0, 3, 5, 7, 8};
    A.row_idx = {0, 2, 3, 1, 2, 2, 3, 3};
    A.values = {4.0, 1.0, -1.0, 3.0, 2.0, -1e-2, 0.5, -1e-2};
    return A;
}

} // namespace

int main(int argc, char **argv) {
    std::string manifest_path;
    std::string case_id = "smoke_kkt_4";
    std::string sequence_id;
    std::string family_id;
    std::string solver_id = "smf";
    std::string mode = "cold";
    std::string track = "recommended";
    std::string jsonl_out;
    std::string mtx_path;
    std::string mtype;  // "spd" | "indefinite" override for .mtx cases
    int trial_id = 0;
    int nrhs = 1;
    bool is_warmup = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--manifest" && i + 1 < argc) manifest_path = argv[++i];
        else if (arg == "--case" && i + 1 < argc) case_id = argv[++i];
        else if (arg == "--sequence-id" && i + 1 < argc) sequence_id = argv[++i];
        else if (arg == "--family-id" && i + 1 < argc) family_id = argv[++i];
        else if (arg == "--solver" && i + 1 < argc) solver_id = argv[++i];
        else if (arg == "--mode" && i + 1 < argc) mode = argv[++i];
        else if (arg == "--track" && i + 1 < argc) track = argv[++i];
        else if (arg == "--trial" && i + 1 < argc) trial_id = std::atoi(argv[++i]);
        else if (arg == "--nrhs" && i + 1 < argc) nrhs = std::max(1, std::atoi(argv[++i]));
        else if (arg == "--mtx" && i + 1 < argc) mtx_path = argv[++i];
        else if (arg == "--mtype" && i + 1 < argc) mtype = argv[++i];
        else if (arg == "--JsonlOut" && i + 1 < argc) jsonl_out = argv[++i];
        else if (arg == "--warmup") is_warmup = true;
    }

    if (sequence_id.empty()) sequence_id = case_id;
    if (family_id.empty()) family_id = sequence_id;

    auto adapter = smf::bench::make_solver_adapter(solver_id);
    smf::bench::RunRecord rec;
    rec.schema_version = 1;
    rec.matrix_id = case_id;
    rec.sequence_id = sequence_id;
    rec.family_id = family_id;
    rec.solver_id = solver_id;
    rec.mode = mode;
    rec.track = track;
    rec.trial_id = trial_id;
    rec.warmup = is_warmup;
    rec.nrhs = nrhs;
#ifdef SMF_BLAS_BACKEND_STR
    rec.blas_backend = SMF_BLAS_BACKEND_STR;
#else
    rec.blas_backend = "unknown";
#endif

    if (!adapter) {
        rec.outcome = smf::bench::Outcome::Unavailable;
        rec.reason = "unregistered solver adapter: " + solver_id;
        if (!jsonl_out.empty()) {
            smf::bench::append_run_record_jsonl(jsonl_out, rec);
        }
        return 1;
    }

    const auto ident = adapter->identity();
    rec.solver_version = ident.version;
    rec.library_path = ident.library_path;

    bool is_spd = false;
    smf::CscLower A;
    if (!mtx_path.empty()) {
        auto mm = smf::read_matrix_market(mtx_path);
        if (!mm.ok) {
            rec.outcome = smf::bench::Outcome::InvalidInput;
            rec.reason = mm.error_message;
            if (!jsonl_out.empty()) smf::bench::append_run_record_jsonl(jsonl_out, rec);
            return 1;
        }
        A = std::move(mm.matrix);
    } else {
        A = make_builtin_fixture(case_id, is_spd);
    }
    if (mtype == "spd") is_spd = true;
    else if (mtype == "indefinite") is_spd = false;

    rec.n = A.n;
    rec.nnz = A.nnz();

    if (!ident.available) {
        rec.outcome = smf::bench::Outcome::Unavailable;
        rec.reason = adapter->stats().reason;
        if (!jsonl_out.empty()) {
            smf::bench::append_run_record_jsonl(jsonl_out, rec);
        }
        return 1;
    }

    if ((is_spd && !ident.supports_spd) ||
        (!is_spd && !ident.supports_indefinite)) {
        rec.outcome = smf::bench::Outcome::Unavailable;
        rec.reason = is_spd ? "solver does not support SPD"
                            : "solver does not support indefinite matrices";
        if (!jsonl_out.empty()) {
            smf::bench::append_run_record_jsonl(jsonl_out, rec);
        }
        return 1;
    }

    smf::Control ctrl;
    ctrl.matrix_type = is_spd
                           ? smf::MatrixType::RealSymmetricPositiveDefinite
                           : smf::MatrixType::RealSymmetricIndefinite;
    adapter->configure(ctrl, track);

    const std::size_t total_len =
        static_cast<std::size_t>(A.n) * static_cast<std::size_t>(nrhs);
    std::vector<double> b(total_len, 1.0);
    std::vector<double> x = b;

    // Untimed in-process warmup so one-time runtime/DLL initialization is excluded
    // from timed solver measurements.
    {
        std::vector<double> x_warm = b;
        if (adapter->analyse(A) == smf::bench::Outcome::Success &&
            adapter->factor(A.values) == smf::bench::Outcome::Success) {
            (void)adapter->solve(x_warm, nrhs);
        }
    }

    smf::bench::Outcome out_a = smf::bench::Outcome::Success;
    smf::bench::Outcome out_f = smf::bench::Outcome::Success;
    smf::bench::Outcome out_s = smf::bench::Outcome::Success;

    if (mode == "warm_sequence") {
        constexpr int kSeqSteps = 5;
        rec.batch_count = kSeqSteps;
        rec.per_update_validation.clear();
        rec.per_update_validation.reserve(kSeqSteps);

        // Pre-build the 5 numeric value updates outside the timed region
        std::vector<std::vector<double>> seq_values(kSeqSteps, A.values);
        for (int step = 0; step < kSeqSteps; ++step) {
            const double delta = 1.0 + 0.05 * static_cast<double>(step);
            for (smf::Int j = 0; j < A.n; ++j) {
                const smf::Int p0 = A.col_ptr[static_cast<std::size_t>(j)];
                if (p0 < A.col_ptr[static_cast<std::size_t>(j) + 1] &&
                    A.row_idx[static_cast<std::size_t>(p0)] == j) {
                    const double v0 = A.values[static_cast<std::size_t>(p0)];
                    if (v0 != 0.0) {
                        seq_values[step][static_cast<std::size_t>(p0)] = v0 * delta;
                    }
                }
            }
        }

        std::vector<std::vector<double>> step_solutions(kSeqSteps, b);
        double total_factor_ms = 0.0;
        double total_solve_ms = 0.0;

        const auto t0 = std::chrono::steady_clock::now();
        out_a = adapter->analyse(A);
        const double analyse_ms = adapter->stats().analyse_ms;
        if (out_a == smf::bench::Outcome::Success) {
            for (int step = 0; step < kSeqSteps; ++step) {
                out_f = adapter->factor(seq_values[step]);
                total_factor_ms += adapter->stats().factor_ms;
                if (out_f != smf::bench::Outcome::Success) break;
                out_s = adapter->solve(step_solutions[step], nrhs);
                total_solve_ms += adapter->stats().solve_ms;
                if (out_s != smf::bench::Outcome::Success) break;
            }
        } else {
            out_f = out_a;
            out_s = out_a;
        }
        const auto t1 = std::chrono::steady_clock::now();

        const auto st = adapter->stats();
        rec.analyse_ms = analyse_ms;
        rec.factor_ms = total_factor_ms;
        rec.solve_ms = total_solve_ms;
        rec.wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        rec.peak_rss_bytes = smf::bench::read_peak_rss_bytes();
        rec.inertia_pos = st.inertia_pos;
        rec.inertia_neg = st.inertia_neg;
        rec.inertia_zero = st.inertia_zero;
        rec.delayed_pivots = st.delayed_pivots;
        rec.factor_entries = st.factor_entries;
        rec.flops = st.flops;

        if (out_s == smf::bench::Outcome::Success) {
            for (int step = 0; step < kSeqSteps; ++step) {
                smf::CscLower A_step = A;
                A_step.values = seq_values[step];
                rec.per_update_validation.push_back(
                    smf::bench::validate(A_step, b, step_solutions[step], nrhs));
            }
            smf::CscLower A_last = A;
            A_last.values = seq_values.back();
            x = step_solutions.back();
            rec.per_rhs_validation = smf::bench::validate_per_rhs(A_last, b, x, nrhs);
            rec.validation = smf::bench::validate(A_last, b, x, nrhs);
            rec.outcome = smf::bench::qualify_outcome(out_s, rec.validation);
            for (const auto &vu : rec.per_update_validation) {
                if (!smf::bench::is_qualified_validation(vu)) {
                    rec.outcome = smf::bench::Outcome::Inaccurate;
                }
            }
            if (rec.outcome != smf::bench::Outcome::Success) {
                rec.reason = "exceeded backward error threshold";
            }
        } else {
            rec.outcome = out_s;
            rec.reason = st.reason;
        }
    } else {
        const int batch_count = (A.n <= 64) ? 10 : 1;
        rec.batch_count = batch_count;
        double sum_a = 0.0, sum_f = 0.0, sum_s = 0.0;

        const auto t0 = std::chrono::steady_clock::now();
        for (int b_it = 0; b_it < batch_count; ++b_it) {
            x = b;
            out_a = adapter->analyse(A);
            sum_a += adapter->stats().analyse_ms;
            out_f = (out_a == smf::bench::Outcome::Success)
                        ? adapter->factor(A.values)
                        : out_a;
            sum_f += adapter->stats().factor_ms;
            out_s = (out_f == smf::bench::Outcome::Success)
                        ? adapter->solve(x, nrhs)
                        : out_f;
            sum_s += adapter->stats().solve_ms;
            if (out_s != smf::bench::Outcome::Success) break;
        }
        const auto t1 = std::chrono::steady_clock::now();

        const auto st = adapter->stats();
        const double inv_batch = 1.0 / static_cast<double>(batch_count);
        rec.analyse_ms = sum_a * inv_batch;
        rec.factor_ms = sum_f * inv_batch;
        rec.solve_ms = sum_s * inv_batch;
        rec.wall_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count() * inv_batch;
        rec.peak_rss_bytes = smf::bench::read_peak_rss_bytes();
        rec.inertia_pos = st.inertia_pos;
        rec.inertia_neg = st.inertia_neg;
        rec.inertia_zero = st.inertia_zero;
        rec.delayed_pivots = st.delayed_pivots;
        rec.factor_entries = st.factor_entries;
        rec.flops = st.flops;

        if (out_s == smf::bench::Outcome::Success) {
            rec.per_rhs_validation = smf::bench::validate_per_rhs(A, b, x, nrhs);
            rec.validation = smf::bench::validate(A, b, x, nrhs);
            rec.outcome = smf::bench::qualify_outcome(out_s, rec.validation);
            if (rec.outcome != smf::bench::Outcome::Success) {
                rec.reason = "exceeded backward error threshold";
            }
        } else {
            rec.outcome = out_s;
            rec.reason = st.reason;
        }
    }

    if (!jsonl_out.empty()) {
        smf::bench::append_run_record_jsonl(jsonl_out, rec);
    }

    return rec.outcome == smf::bench::Outcome::Success ? 0 : 1;
}
