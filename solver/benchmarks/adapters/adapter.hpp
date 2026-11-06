#pragma once

#include "../common/result.hpp"
#include "../common/validation.hpp"
#if __has_include("smf/control.hpp")
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/ipm_session.hpp"
#else
#include "../../include/smf/control.hpp"
#include "../../include/smf/csc_matrix.hpp"
#include "../../include/smf/ipm_session.hpp"
#endif

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace smf::bench {

struct AdapterIdentity {
    std::string solver_id;
    std::string version;
    std::string library_path;
    std::string integer_abi = "int32";
    bool available = false;
    bool supports_spd = true;
    bool supports_indefinite = true;
    bool is_external_memory = false;
};

struct AdapterStats {
    Outcome outcome = Outcome::Failed;
    std::string reason;
    double analyse_ms = 0.0;
    bool has_analyse_phases = false;  // smf adapters: Info::analyse_phases of the last analyse
    AnalysePhaseSeconds analyse_phases;
    double factor_ms = 0.0;
    double solve_ms = 0.0;
    int inertia_pos = 0;
    int inertia_neg = 0;
    int inertia_zero = 0;
    int delayed_pivots = 0;
    int64_t factor_entries = 0;
    double flops = 0.0;
    // smf adapters: an FNV-style hash (FNV-1a's prime; truncated offset
    // basis) over the fill-reducing perm (AnalysisKeep::perm)
    // of every distinct analysis the factorizations since analyse() used, in
    // order of first use; 0 = no factorization. Compares orderings across
    // builds (e.g. two METIS builds) without dumping them.
    std::uint64_t order_hash = 0;
    int static_pivots = 0;  // Info::static_pivots of the last factor (smf adapters)
    int aptp_failed_columns = 0;  // Info::aptp_failed_columns of the last factor (smf adapters)
    int posdef_lapack_fronts = 0;  // Info::posdef_lapack_fronts of the last factor (smf adapters)
    int delay_rescales = 0;  // Info::delay_rescales of the last factor (smf adapters)
    int delay_early_stops = 0;  // Info::delay_early_stops of the last factor (smf adapters)
    double delay_try_ms = 0.0;       // Info::delay_try_seconds of the last factor, in ms (smf adapters)
    double delay_matching_ms = 0.0;  // Info::delay_matching_seconds of the last factor, in ms (smf adapters)
    // smf adapters, Control::skip_trial_gate: the analysis skipped the trial
    // (AnalysisKeep::matching_trial_skipped; -1 = not reported), the base AMD
    // flops per row it decided on (0 = not evaluated), and Info::
    // skip_trial_rescues of the last factor.
    int trial_skipped = -1;
    double gate_flops_per_row = 0.0;
    int skip_trial_rescues = 0;
    // smf adapters, Control::slack_elimination: the last factor ran through
    // the slack plan (0/1), or tried it and took the full matrix (0/1).
    int slack_armed = 0;
    int slack_fallbacks = 0;
    // smf adapters: the subtree engine's shape and phase times (Info) of the
    // last factor; subtree/top/gather are 0 when it ran serially.
    int threads_used = 1;
    int parallel_subtrees = 0;
    int top_part_fronts = 0;
    int top_part_queue_pops = 0;  // of those, taken from the scheduler's shared list
    double subtree_ms = 0.0;
    double top_ms = 0.0;
    double gather_ms = 0.0;
    int solve_threads = 1;  // FactorKeep::solve_threads of the last factor
    // smf adapter: smf::refine() of the last solve, -1 when it did not run.
    int refine_iterations = -1;
    bool refine_fgmres = false;  // RefinementResult::method_used == Fgmres
    // smf adapter: RefinementResult::omega_start of the last solve (omega
    // before refinement), -1 when it did not run.
    double refine_omega_start = -1.0;
    int static_fallback = 0;   // Info::static_fallback of the last factor (smf adapters)
    int static_rejects = 0;    // Info::static_rejects of the last factor (smf adapters)
    double pivot_growth = 0.0; // Info::pivot_growth of the last factor (smf adapters)
};

class SolverAdapter {
public:
    virtual ~SolverAdapter() = default;
    virtual AdapterIdentity identity() const = 0;
    virtual void configure(const Control &ctrl, const std::string &track = "recommended") = 0;
    // A, and the values of the latest factor(), must stay alive while the
    // adapter uses them: adapters may reference rather than copy them.
    virtual Outcome analyse(const CscLower &A) = 0;
    virtual Outcome factor(std::span<const double> values) = 0;
    virtual Outcome solve(std::span<double> b_and_x, int nrhs = 1) = 0;
    virtual AdapterStats stats() const = 0;
};

std::unique_ptr<SolverAdapter> make_smf_adapter();
/// smf through smf::IpmSession, as the MA97 plugin runs it inside IPOPT, with
/// only the static mode on (no inertia learning, abort or re-analysis):
/// `mode` = IpmControl::static_mode, `static_zero` = IpmControl::
/// static_as_zero. analyse() defers the analysis to the first factor(), as
/// the plugin does; a Singular session factor is Outcome::Singular.
std::unique_ptr<SolverAdapter> make_smf_ipm_adapter(StaticMode mode, bool static_zero);
/// smf exactly as the MA97 plugin (libsmf_ma97) runs it inside IPOPT:
/// smf::IpmSession with `ipm` (bench_replay passes smf::
/// ipopt_plugin_ipm_control()), configured with the plugin's Control
/// (smf::ipopt_plugin_control()). Same deferred analysis as
/// make_smf_ipm_adapter; bench_replay's --track ipopt-plugin.
std::unique_ptr<SolverAdapter> make_smf_ipopt_plugin_adapter(const IpmControl &ipm);
/// bench_replay's --ipm-static value: off | fallback | regularized.
std::optional<StaticMode> parse_ipm_static(const std::string &name);
std::unique_ptr<SolverAdapter> make_hsl_adapter(const std::string &hsl_id);
/// MA97 / MA57 loaded at runtime from the library named by SMF_HSL_LIBRARY.
std::unique_ptr<SolverAdapter> make_hsl_ma97_adapter();
std::unique_ptr<SolverAdapter> make_hsl_ma57_adapter();
std::unique_ptr<SolverAdapter> make_spral_ssids_adapter();
std::unique_ptr<SolverAdapter> make_mumps_adapter();
std::unique_ptr<SolverAdapter> make_pardiso_adapter(const std::string &variant = "mkl_pardiso");
std::unique_ptr<SolverAdapter> make_cholmod_adapter();
/// NVIDIA cuDSS (GPU, FP64). Real only with -DSMF_BENCH_CUDSS=ON and a CUDA
/// device; otherwise an Unavailable stub.
std::unique_ptr<SolverAdapter> make_cudss_adapter();
std::unique_ptr<SolverAdapter> make_solver_adapter(const std::string &solver_id);
std::vector<std::string> all_registered_adapter_ids();

} // namespace smf::bench
