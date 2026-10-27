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
    int64_t factor_entries = 0;
    double flops = 0.0;

    std::string to_json_line() const;
};

const char* outcome_to_string(Outcome outcome) noexcept;
bool append_run_record_jsonl(const std::string& path, const RunRecord& record);
int64_t read_peak_rss_bytes() noexcept;

} // namespace smf::bench
