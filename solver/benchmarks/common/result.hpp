#pragma once

#if __has_include("smf/csc_matrix.hpp")
#include "smf/csc_matrix.hpp"
#include "smf/info.hpp"
#include "smf/types.hpp"
#else
#include "../../include/smf/csc_matrix.hpp"
#include "../../include/smf/info.hpp"
#include "../../include/smf/types.hpp"
#endif

#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace smf::bench {

enum class Outcome {
    Success = 0,
    Inaccurate,
    Singular,
    InvalidInput,
    Unavailable,
    Timeout,
    OutOfMemory,
    Failed
};

struct Validation {
    bool finite = false;
    double eta_inf = std::numeric_limits<double>::infinity();
    double omega = std::numeric_limits<double>::infinity();
    double relative_residual = std::numeric_limits<double>::infinity();
};

struct RunRecord {
    int schema_version = 1;
    std::string matrix_id;
    std::string sequence_id;
    std::string family_id;
    std::string matrix_sha256;
    std::string rhs_sha256;
    std::string solver_id;
    std::string solver_version;
    std::string library_path;
    std::string blas_backend;
    std::string mode = "cold";
    std::string track = "recommended";
    int threads = 1;
    int blas_threads = 1;
    int n = 0;
    int64_t nnz = 0;
    int nrhs = 1;
    int trial_id = 0;
    bool warmup = false;
    int batch_count = 1;
    Outcome outcome = Outcome::Failed;
    std::string reason;
    double analyse_ms = 0.0;
    bool has_analyse_phases = false;  // smf solvers only
    AnalysePhaseSeconds analyse_phases;  // seconds, as Info::analyse_phases
    double factor_ms = 0.0;
    // warm_sequence only (negative = not measured, left out of the record):
    // the first factorization after analyse, and the median of the warm
    // refactorizations that follow it.
    double first_factor_ms = -1.0;
    double refactor_ms = -1.0;
    double solve_ms = 0.0;
    double refine_ms = 0.0;
    double wall_ms = 0.0;
    int64_t peak_rss_bytes = -1;
    Validation validation{};
    std::vector<Validation> per_rhs_validation;
    std::vector<Validation> per_update_validation;
    int inertia_pos = 0;
    int inertia_neg = 0;
    int inertia_zero = 0;
    int delayed_pivots = 0;
    int aptp_failed_columns = -1;  // smf only (Info::aptp_failed_columns); -1 = not reported
    int posdef_lapack_fronts = -1; // smf only (Info::posdef_lapack_fronts); -1 = not reported
    int static_pivots = -1;        // smf only (Info::static_pivots); -1 = not reported
    // smf only: factorizations of the run whose scale-on-delay trigger
    // computed a matching scaling (Info::delay_rescales, summed over a
    // warm sequence's steps); -1 = not reported.
    int delay_rescales = -1;
    // smf only: of those, tries stopped at the early crossing
    // (Info::delay_early_stops, summed likewise); -1 = not reported.
    int delay_early_stops = -1;
    // smf only: time of the stopped or failed try and of the rescue matching
    // (Info::delay_try_seconds, delay_matching_seconds), ms; -1 = not reported.
    double delay_try_ms = -1.0;
    double delay_matching_ms = -1.0;
    // smf only, Control::skip_trial_gate: the analysis skipped the auto
    // matching trial (0/1; -1 = not reported), the base AMD flops per row the
    // gate decided on (-1 = not reported, 0 = not evaluated), and the
    // factorizations whose gate-armed rescue fired (Info::skip_trial_rescues,
    // summed over a warm sequence's steps; -1 = not reported).
    int trial_skipped = -1;
    double gate_flops_per_row = -1.0;
    int skip_trial_rescues = -1;
    // smf only: smf::refine() inside the adapter's solve (static pivots, or
    // SMF_BENCH_REFINE_ALWAYS); -1 / empty = it did not run.
    int refine_iterations = -1;
    std::string refine_method;     // "classical" | "fgmres"
    // smf only, Control::static_fallback_after_cascade / Pivoting::Static:
    // factorizations on the static path (Info::static_fallback summed over a
    // warm sequence's steps), the largest Info::pivot_growth, the smf::refine()
    // steps summed over the solves, and the largest omega before refinement
    // (RefinementResult::omega_start); -1 = not reported.
    int static_fallbacks = -1;
    int static_rejects = -1;  // Info::static_rejects summed over the steps
    double pivot_growth = -1.0;
    int refine_steps_total = -1;
    double omega_unrefined = -1.0;
    int64_t factor_entries = 0;
    double flops = 0.0;
    // smf only: AdapterStats::order_hash as 16 hex digits; empty = left out.
    std::string order_hash;

    std::string to_json_line() const;
};

const char* outcome_to_string(Outcome outcome) noexcept;
bool append_run_record_jsonl(const std::string& path, const RunRecord& record);
int64_t read_peak_rss_bytes() noexcept;

} // namespace smf::bench
