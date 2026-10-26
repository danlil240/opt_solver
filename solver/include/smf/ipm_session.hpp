#pragma once
#include "smf/analysis.hpp"
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/dense_indef.hpp"
#include "smf/error.hpp"
#include "smf/factor_posdef.hpp"
#include "smf/info.hpp"
#include "smf/solver.hpp"
#include "smf/types.hpp"
#include "smf/workspace.hpp"

#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace smf {

enum class StaticMode { Off, Fallback, Regularized };

/// Interior-point options of an IpmSession.
struct IpmControl {
  bool       learn_inertia          = true;  // infer the expected #neg from accepted factors
  bool       abort_on_wrong_inertia = true;  // stop once #neg exceeds the expectation
  double     reanalyse_delay_frac   = 0.01;  // delays > frac * n -> try realized-order re-analysis
  double     reanalyse_flops_ratio  = 1.0;   // adopt only if predicted flops <= ratio * last actual
  int        reanalyse_min_gap      = 5;     // factorizations between re-analysis attempts
  StaticMode static_mode            = StaticMode::Off;
  double     reg_rho                = 1e-8;  // Regularized: rho relative to max|A| after scaling
};

enum class IpmFactorStatus { Ok, WrongInertia, Singular, Error };

struct IpmStats {
  int factorizations = 0, accepted = 0, aborted = 0;
  int reanalyses_tried = 0, reanalyses_adopted = 0, reanalyses_reverted = 0;
  long static_pivots = 0;
  double flops_avoided = 0.0;  // predicted flops of the fronts skipped by aborts
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
/// expectation costs time, never correctness. The other features are not
/// implemented yet.
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
  ///
  /// WrongInertia: the factorization stopped once its negative pivots
  /// exceeded the expectation (`expected_negatives` when given, else the
  /// learned one; armed only with IpmControl::abort_on_wrong_inertia and an
  /// indefinite matrix). inertia() / info().num_negative then hold the
  /// partial counts, num_negative > the expectation.
  IpmFactorStatus factor(std::span<const double> values,
                         std::optional<int> expected_negatives = {});

  /// Solve with the last factor, in place (b: n x nrhs, column-major).
  /// Returns 0 on success. A successful solve marks the last factor
  /// "accepted" (and, with learn_inertia, its negative count becomes the
  /// expectation). `job` selects a partial solve (MA97 jobs 1-4). The solve
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

  const FactorKeep& factor_keep() const { return fk_; }
  /// The current analysis; null before it has run or after it failed.
  const AnalysisKeep* analysis_keep() const { return ak_.get(); }

 private:
  ErrorCode run_analysis();  // analyse mat_ with ctrl_; times it in info_.analyse_seconds
  // One engine factorization of mat_ with factor_ctrl_ and the given
  // negative-pivot limit (-1 off); sets the factor state and returns the
  // IpmFactorStatus.
  IpmFactorStatus run_factor(int max_negative_pivots);

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
  bool user_scale_once_ = false;
  bool solvable_ = false;        // last factor Ok or Singular
  bool aborted_ = false;         // last factor WrongInertia: complete it before solving
  bool pending_accept_ = false;  // last factor usable and not yet solved with
  bool last_spd_ok_ = false;     // last factor a successful SPD one
};

} // namespace smf
