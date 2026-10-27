#pragma once
#include "smf/analysis.hpp"
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/dense_indef.hpp"
#include "smf/error.hpp"
#include "smf/factor_posdef.hpp"
#include "smf/info.hpp"
#include "smf/refinement.hpp"
#include "smf/solver.hpp"
#include "smf/types.hpp"
#include "smf/workspace.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace smf {

enum class StaticMode { Off, Fallback, Regularized };

/// Interior-point options of an IpmSession.
///
/// abort_on_wrong_inertia and reanalyse default to false. The spec's struct
/// (docs/superpowers/specs/2026-10-26-ipm-mode-design.md) had
/// abort_on_wrong_inertia = true, and reanalyse (added beside it) was true
/// until the gates ran. Both features failed their gates on the IPOPT tuning
/// and held-out models (docs/performance/2026-10-26-ipm-mode.md), so the
/// measured decision is off and a caller opts in explicitly.
struct IpmControl {
  bool       learn_inertia          = true;  // infer the expected #neg from accepted factors
  bool       abort_on_wrong_inertia = false; // stop once #neg exceeds the expectation (gate failed: off)
  bool       reanalyse              = false; // Feature 2: re-analysis from the realized order (gate failed: off)
  double     reanalyse_delay_frac   = 0.01;  // delays > frac * n -> try realized-order re-analysis
  double     reanalyse_flops_ratio  = 1.0;   // adopt only if predicted flops <= ratio * last actual
  int        reanalyse_min_gap      = 5;     // factorizations between re-analysis attempts
  StaticMode static_mode            = StaticMode::Off;
  // Static modes: every pivot raised by static pivoting counts as a zero
  // eigenvalue (Control::static_pivots_as_zero). Off by default, where only
  // a pivot raised from an exact zero counts as zero and one raised from a
  // nonzero value keeps its sign: with all of them zero, IPOPT's delta_c
  // cannot lift pivots raised relative to a max|A| that grows with delta_x
  // (cops_elec_200 fails).
  bool       static_as_zero         = false;
  double     reg_rho                = 1e-8;  // Regularized: rho relative to max|A| after scaling
};

enum class IpmFactorStatus { Ok, WrongInertia, Singular, Error };

struct IpmStats {
  int factorizations = 0, accepted = 0, aborted = 0;
  int reanalyses_tried = 0, reanalyses_adopted = 0, reanalyses_reverted = 0;
  long static_pivots = 0;  // Info::static_pivots summed over the complete factorizations
  double flops_avoided = 0.0;  // predicted flops of the fronts skipped by aborts
  double reanalyse_seconds = 0.0;  // wall time of the re-analyses (inside factor())
};

/// One KKT pattern factorized repeatedly by an interior-point method.
///
/// The session owns the Solver, the AnalysisKeep, the FactorKeep and the
/// workspaces, and reuses them across refactorizations.
///
/// Feature 1, inertia learning and early abort (IpmControl::learn_inertia,
/// abort_on_wrong_inertia): an interior-point method accepts a factor only
/// with one fixed number of negative pivots per pattern. The session learns
/// it from the last factor followed by a successful solve (or takes it from
/// factor()'s `expected_negatives`) and stops a later factorization as soon
/// as it has more negative pivots (IpmFactorStatus::WrongInertia). A solve
/// after such an abort first completes the factorization, so a wrong
/// expectation costs time, never correctness.
///
/// Feature 2, re-analysis from the realized pivot order (IpmControl::
/// reanalyse): when an accepted factor (one followed by a successful solve)
/// delayed more than reanalyse_delay_frac * n pivots (Info::delayed_pivots)
/// and at least reanalyse_min_gap factorizations separate this attempt from
/// the last one, the next factor() first analyses the pattern again from
/// that factor's realized elimination order and 2x2 pairs
/// (realized_pivot_order, OrderingMethod::UserSupplied). It adopts the new
/// analysis only if its predicted flops, counted as the factorization kernel
/// counts Info::actual_flops (sum over the fronts' pivots of r^2 + 2r, r the
/// rows below the pivot; not Info::predicted_flops' p f^2), are at most
/// reanalyse_flops_ratio x the accepted factor's actual flops. An empty
/// realized order or a failed analysis is a rejected attempt. The previous
/// analysis is kept while the adopted one is on probation: its first
/// complete factorization. If that delays more pivots than the accepted
/// factor did, the session reverts to the previous analysis at the next
/// factor() (the probation factor, which points at the new analysis, stays
/// usable for solves until then) and never re-analyses this pattern again.
/// analyse() starts a new history.
///
/// Feature 3, IpmControl::static_mode. Fallback: threshold pivoting with
/// delays, static pivots only where delaying fails (root fronts and fronts
/// over their delay budget). Both static modes factor with the serial
/// engine (Control::tree_parallel is forced off: the task-parallel engine
/// ignores the fallback, static and in-order settings). Regularized, the
/// quasidefinite mode:
///  - analyse() adds every structurally missing diagonal entry to the
///    pattern (values keep the input entry order and count) and analyses
///    with AMD (no matching ordering);
///  - the first factor() after analyse() classifies the rows, then they stay
///    fixed: a zero or missing diagonal is a constraint row, anything else
///    (negative included) a primal row (row_is_constraint());
///  - every factor() adds +rho to the primal and -rho to the constraint
///    diagonal, rho = reg_rho x max|A| after scaling, and factors with 1x1
///    pivots in order (no search, no delays); a pivot with |d| < rho is
///    raised to rho and counted in Info::static_pivots (IpmStats sums them);
///  - a raised pivot counts as a zero eigenvalue if it was raised from an
///    exact zero, else by its sign (hybrid), or always as a zero with
///    IpmControl::static_as_zero;
///  - every full solve is refined against the unregularized matrix, up to
///    Control::refinement_max_iters classical steps (RefineMethod::Classical
///    whatever Control::refine_method says: the plugin's semantics stay
///    those of the refinement before FGMRES).
class IpmSession {
 public:
  /// Store the pattern (lower CSC; its own values are ignored) and start a
  /// new history: stats() is zeroed and the previous analysis and factor are
  /// dropped. With `values` (in the pattern's entry order, size nnz) the
  /// analysis runs now; with an empty span it runs inside the first factor(),
  /// which has values (the analysis needs them to recognise saddle points).
  ErrorCode analyse(const CscLower& pattern, std::span<const double> values,
                    const Control& ctrl, const IpmControl& ipm);

  /// Factor new values in the pattern's entry order with the current
  /// Control (set_control / set_user_scale). Ok also covers
  /// FactorStatus::MaxPivotDelays; Singular is a factor with zero pivots
  /// (ErrorCode::SingularMatrix); every other failure, including a failed
  /// deferred analysis (then analysis_keep() is null), is Error.
  /// info().factor_status is the engine's FactorStatus of this call.
  /// info().inertia_exact is false whenever a pivot was static-pivoted
  /// (the static modes, Control::static_pivot_tol or delay_budget); the
  /// session reports it and does not act on it (the MA97 plugin turns it
  /// into a failure outside its own static modes).
  ///
  /// WrongInertia: the factorization stopped once its negative pivots
  /// exceeded the expectation (`expected_negatives` when given, else the
  /// learned one; armed only with IpmControl::abort_on_wrong_inertia and an
  /// indefinite matrix). inertia() / info().num_negative then hold the
  /// partial counts, num_negative > the expectation.
  IpmFactorStatus factor(std::span<const double> values,
                         std::optional<int> expected_negatives = {});

  /// Solve with the last factor, in place (b: n x nrhs, column-major).
  /// Returns 0 on success, nonzero (ErrorCode::NotInitialised) when there
  /// is no factor to solve with: none yet, a failed one, or a Singular one
  /// the engine never ran (a failure before it, e.g. the matching scaling
  /// finding no matching), which leaves no factor of these values; such a
  /// solve neither accepts the factor nor learns its inertia. A successful
  /// solve marks the last factor "accepted" (and, with learn_inertia, its
  /// negative count becomes the expectation). `job` selects a partial solve (MA97 jobs 1-4). The solve
  /// options come from the current Control. After a WrongInertia factor the
  /// stored values are first factorized in full, with the Control and user
  /// scale of that factor() call and no negative-pivot limit.
  int solve(double* b, int n, int nrhs, SolveJob job = SolveJob::Full);

  /// Inertia of the last factor (n positive for a successful SPD factor).
  InertiaCounts inertia() const;
  /// Statistics of the analysis and the last factor (solves keep their own).
  /// analyse_seconds is the analysis' wall time, also when it ran inside
  /// factor().
  const Info& info() const { return info_; }
  const IpmStats& stats() const { return stats_; }
  /// The learned expectation: the negative count of the most recent factor
  /// followed by a successful solve; unset before one, and after analyse().
  std::optional<int> expected_negatives() const { return expected_neg_; }
  /// Increases on every analyse() and factor() call (never reset): a caller
  /// holding several handles to one session (the MA97 plugin's fkeeps)
  /// records it at factor time to tell whether the session's factor is
  /// still the one its handle was given.
  std::uint64_t factor_generation() const { return factor_generation_; }

  /// Replace the Control used by the next factor and solve (an MA97 caller
  /// passes a control block on every call: pivot threshold, matrix type,
  /// scaling and solve options may change between factorizations). It does
  /// not change the last factor: factor() keeps its own copy of the Control
  /// (and of the user scale) for completing an aborted factorization.
  void set_control(const Control& ctrl);
  /// Control::user_scale for the next factor only (cleared after it): the
  /// MA97 "reuse" scaling. Call after set_control. factor() copies the n
  /// values, so the buffer need only live until factor() returns.
  void set_user_scale(const double* scale);

  /// Feature 3b (StaticMode::Regularized): whether row i is a constraint row
  /// (regularized by -rho). Decided at the first factorization after
  /// analyse() and then fixed; false before it, out of range, or in another
  /// mode.
  bool row_is_constraint(Int i) const;

  const FactorKeep& factor_keep() const { return fk_; }
  /// The current analysis; null before it has run or after it failed.
  const AnalysisKeep* analysis_keep() const { return ak_.get(); }
  /// Feature 2: the analysis kept while an adopted one is on probation, or
  /// until a revert takes effect; null otherwise. factor_keep().analysis is
  /// always analysis_keep(), this one, or null.
  const AnalysisKeep* previous_analysis_keep() const { return prev_ak_.get(); }

 private:
  ErrorCode run_analysis();  // analyse mat_ with ctrl_; times it in info_.analyse_seconds
  // One engine factorization of mat_ with factor_ctrl_ and the given
  // negative-pivot limit (-1 off); sets the factor state, settles a
  // probation and returns the IpmFactorStatus.
  IpmFactorStatus run_factor(int max_negative_pivots);
  // Feature 2. After an accepted factor: arm a re-analysis for the next
  // factor() if this one delayed enough and the gap allows it.
  void consider_reanalysis();
  // At the start of factor(): carry out a pending revert or re-analysis.
  void prepare_analysis();
  // Free prev_ak_ after a passed probation (kept if fk_ still points at it).
  void drop_previous_analysis();
  // Copy input values (input entry order) into mat_ (through entry_map_).
  void load_values(std::span<const double> values);
  bool regularized() const { return ipm_.static_mode == StaticMode::Regularized; }

  // Feature 3b state (StaticMode::Regularized). mat_ is the input pattern
  // with every missing diagonal entry added (value 0) at the head of its
  // column; entry_map_[k] is input entry k's slot in it (empty: none added,
  // the identity). reg_sign_: per row, +1 primal / -1 constraint, set at the
  // first factorization after analyse() (empty before).
  std::vector<Int> entry_map_;
  std::size_t input_nnz_ = 0;
  std::vector<int8_t> reg_sign_;
  std::vector<double> refine_rhs_;  // the right-hand sides of the solve being refined
  // The unregularized matrix refinement checks against: mat_'s canonical
  // pattern, built once per analyse(); values set on every factor().
  RefineMatrix refine_mat_;

  // Feature 2 state.
  std::unique_ptr<AnalysisKeep> prev_ak_;  // the analysis before an adoption (probation)
  Info prev_info_;                         // info_ when prev_ak_ was the current analysis
  std::vector<Int> order_perm_;            // realized order armed for the next factor()
  std::vector<Int> order_pairs_;
  bool reanalyse_armed_ = false;
  int trigger_delays_ = 0;        // delays of the accepted factor that armed it
  double trigger_flops_ = 0.0;    // its actual flops
  int last_attempt_ = -1;         // stats_.factorizations index of the last attempt
  bool probation_ = false;        // adopted analysis not yet judged
  int probation_delays_ = 0;      // the delays it must not exceed
  bool revert_pending_ = false;   // switch back to prev_ak_ at the next factor()
  bool reanalyse_stopped_ = false;  // reverted once: never again for this pattern

  Solver solver_;
  Control ctrl_;         // current Control: next factor, and every solve's options
  Control factor_ctrl_;  // the last factor()'s Control; user_scale -> factor_scale_
  std::vector<double> factor_scale_;  // copy of the last factor()'s user scale
  IpmControl ipm_;
  CscLower mat_;  // pattern + the values being factored
  std::unique_ptr<AnalysisKeep> ak_;
  FactorKeep fk_;
  NumericWorkspace nws_;
  SolveWorkspace sws_;
  Info info_;
  Info solve_info_;
  IpmStats stats_;
  std::optional<int> expected_neg_;
  std::uint64_t factor_generation_ = 0;
  bool user_scale_once_ = false;
  bool solvable_ = false;        // last factor Ok or Singular
  bool aborted_ = false;         // last factor WrongInertia: complete it before solving
  bool pending_accept_ = false;  // last factor usable and not yet solved with
  bool last_spd_ok_ = false;     // last factor a successful SPD one
};

} // namespace smf
