#include "adapters/adapter.hpp"
#include "common/result.hpp"
#include "common/validation.hpp"
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/matrix_market.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

smf::CscLower make_builtin_fixture(const std::string &case_id, bool &is_spd) {
    is_spd = false;
    if (case_id == "smoke_spd_tridiag_10") {
        is_spd = true;
        smf::CscLower A;
        A.n = 10;
        A.col_ptr.resize(11, 0);
        for (int j = 0; j < 10; ++j) {
            A.row_idx.push_back(j);
            A.values.push_back(4.0);
            if (j + 1 < 10) {
                A.row_idx.push_back(j + 1);
                A.values.push_back(-1.0);
            }
            A.col_ptr[static_cast<std::size_t>(j + 1)] =
                static_cast<smf::Int>(A.row_idx.size());
        }
        return A;
    }
    if (case_id == "smoke_indef_exchange_2") {
        smf::CscLower A;
        A.n = 2;
        A.col_ptr = {0, 2, 3};
        A.row_idx = {0, 1, 1};
        A.values = {0.0, 1.0, 0.0};
        return A;
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
    std::string solver_id = "smf";
    std::string mode = "cold";
    std::string track = "recommended";
    std::string jsonl_out;
    std::string mtx_path;
    int trial_id = 0;
    int nrhs = 1;
    bool is_warmup = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--manifest" && i + 1 < argc) manifest_path = argv[++i];
        else if (arg == "--case" && i + 1 < argc) case_id = argv[++i];
        else if (arg == "--solver" && i + 1 < argc) solver_id = argv[++i];
        else if (arg == "--mode" && i + 1 < argc) mode = argv[++i];
        else if (arg == "--track" && i + 1 < argc) track = argv[++i];
        else if (arg == "--trial" && i + 1 < argc) trial_id = std::atoi(argv[++i]);
        else if (arg == "--nrhs" && i + 1 < argc) nrhs = std::max(1, std::atoi(argv[++i]));
        else if (arg == "--mtx" && i + 1 < argc) mtx_path = argv[++i];
        else if (arg == "--JsonlOut" && i + 1 < argc) jsonl_out = argv[++i];
        else if (arg == "--warmup") is_warmup = true;
    }

    auto adapter = smf::bench::make_solver_adapter(solver_id);
    smf::bench::RunRecord rec;
    rec.schema_version = 1;
    rec.matrix_id = case_id;
    rec.sequence_id = case_id;
    rec.family_id = case_id;
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

    smf::Control ctrl;
    ctrl.matrix_type = is_spd
                           ? smf::MatrixType::RealSymmetricPositiveDefinite
                           : smf::MatrixType::RealSymmetricIndefinite;
    adapter->configure(ctrl, track);

    const std::size_t total_len =
        static_cast<std::size_t>(A.n) * static_cast<std::size_t>(nrhs);
    std::vector<double> b(total_len, 1.0);
    std::vector<double> x = b;

    const auto t0 = std::chrono::steady_clock::now();
    smf::bench::Outcome out_a = adapter->analyse(A);
    smf::bench::Outcome out_f =
        (out_a == smf::bench::Outcome::Success)
            ? adapter->factor(A.values)
            : out_a;
    smf::bench::Outcome out_s =
        (out_f == smf::bench::Outcome::Success)
            ? adapter->solve(x, nrhs)
            : out_f;
    const auto t1 = std::chrono::steady_clock::now();

    const auto st = adapter->stats();
    rec.analyse_ms = st.analyse_ms;
    rec.factor_ms = st.factor_ms;
    rec.solve_ms = st.solve_ms;
    rec.wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
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

    if (!jsonl_out.empty()) {
        smf::bench::append_run_record_jsonl(jsonl_out, rec);
    }

    return rec.outcome == smf::bench::Outcome::Success ? 0 : 1;
}
