// Slack-pair elimination (Control::slack_elimination).
//
// IPOPT's augmented system carries, per inequality constraint d_i(x) - s_i = 0,
// a slack s_i whose only off-diagonal is its -1 to the multiplier y_i, and
// the multiplier y_i, coupled further only to the k states b_i of the
// constraint's Jacobian row:
//
//          x     s_i     y_i
//   s_i    0   a_ss    a_ys
//   y_i   b_i  a_ys    a_yy
//
// analyse() finds these pairs; when they are a large share of the rows it
// also analyses the reduced matrix (the rows that are not in a pair, with
// every pair's k x k state block added to the pattern). factor() then
// eliminates each pair as its 2x2 block P = [a_ss a_ys; a_ys a_yy] in one
// flat loop (the states receive -(P^-1)_yy b b^T; exact also for a zero
// a_ss, a variable in one constraint only), factors the reduced matrix with
// the normal engine (its own ordering, scaling and pivoting), and solve()
// wraps the reduced solve with the pairs' forward and back substitutions.
// The inertia is the reduced factor's plus each pair's (Sylvester: the
// reduced matrix is a Schur complement). A pair whose 2x2 pivot fails the
// growth test makes that factorization take the full matrix as without the
// reduction.
#pragma once

#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/error.hpp"
#include "smf/types.hpp"

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

namespace smf {

struct AnalysisKeep;
struct CscLower;
class Solver;
/// The reduction runs at pivot thresholds up to this (IPOPT's 1e-8 to its
/// raised ones); above it, condensing active constraints is not accurate
/// enough (the u = 0.01 front tier), and nothing changes.
constexpr double kSlackMaxPivotU = 1e-6;
/// Fallbacks in a row after which factor() no longer tries the plan
/// (AnalysisKeep::slack_fallbacks_in_row).
constexpr int kSlackMaxFallbacks = 3;
/// Refinement steps of a full solve through the plan whose accuracy check
/// fails (Control::slack_omega_target), before the full matrix's factor.
constexpr int kSlackRefineSteps = 3;
/// Matrices of lower order never take the reduction.
constexpr Int kSlackMinOrder = 256;
struct FactorKeep;
struct Info;
struct SolveWorkspace;

/// One slack/multiplier pair: its rows (original indices), the slots of its
/// 2x2 block in the cleaned values (yy < 0: no stored diagonal, 0), and its
/// state entries [b0, b0 + nb) in SlackPlan::b_slot / b_row and the k(k+1)/2
/// slots of its state block from blk0 in SlackPlan::blk.
struct SlackPair {
    Int s = -1, y = -1;
    LongInt ss = -1, ys = -1, yy = -1;
    Int nb = 0;
    LongInt b0 = 0, blk0 = 0;
};

/// The analysis' pair elimination: the pairs, the reduced matrix's layout
/// and its analysis.
struct SlackPlan {
    Int n_full = 0;                 ///< order of the full matrix
    Int nr = 0;                     ///< order of the reduced matrix
    std::vector<Int> red_to_full;   ///< reduced index -> original index
    std::vector<SlackPair> pairs;
    std::vector<LongInt> b_slot;    ///< cleaned value slot of each state entry
    std::vector<Int> b_row;         ///< its state row (reduced index)
    std::vector<LongInt> sdiag;     ///< cleaned slot of each reduced row's diagonal (-1: none)
    std::vector<LongInt> blk;       ///< reduced value slots of the pairs' state blocks (lower, row-major)
    std::vector<LongInt> keep_src;  ///< cleaned value slots of the kept entries
    std::vector<LongInt> keep_dst;  ///< their reduced value slots
    std::size_t reduced_nnz = 0;    ///< entries of the reduced matrix (its input order)
    std::shared_ptr<AnalysisKeep> reduced;
    /// The input and analyse()'s control, for the full matrix's own analysis
    /// (exactly analyse() without the plan), made on the first factorization
    /// that does not run through the plan.
    CscLower input;
    Control actrl;
    /// The cleaned pattern (the cleaned values' layout), for the solve's
    /// residual on the full matrix.
    std::vector<LongInt> clean_ptr;
    std::vector<Int> clean_row;
    mutable std::mutex full_mu;
    mutable std::shared_ptr<const AnalysisKeep> full;
};

/// A factorization through the plan: the pairs' 2x2 inverses (3 per pair:
/// (P^-1)_ss, _sy, _yy) and their state entries' values.
struct SlackFactor {
    std::shared_ptr<const SlackPlan> plan;
    std::vector<double> pinv;
    std::vector<double> bval;
    /// The full matrix (its cleaned values, with the factorization's
    /// control) for the solve's accuracy check: a full solve
    /// whose componentwise backward error on the full matrix stays above
    /// Control::slack_omega_target after refinement factors the full matrix
    /// (its own analysis) once and solves with that from then on.
    std::vector<double> cv;
    Control ctrl;
    mutable std::mutex full_mu;
    mutable std::shared_ptr<FactorKeep> full;
    /// The full refactor's failure (its status, returned by this and later
    /// solves of the factor); Success while it was not tried or succeeded.
    mutable ErrorCode full_error = ErrorCode::Success;
    /// The last pair pass: pairs whose pivot failed, the largest growth
    /// bound among the others, the largest |a_ss|.
    int fails = 0;
    double max_growth = 0.0;
    double max_slack = 0.0;
};

/// Builds the plan of `keep` (analysed from A with `ctrl`), or null when the
/// pairs are fewer than Control::slack_min_share of the rows (Auto) or none
/// (On), or the reduction does not apply.
std::shared_ptr<const SlackPlan> build_slack_plan(const CscLower &cleaned, const Control &ctrl);

/// analyse() through a slack plan: when it applies (indefinite, an auto
/// ordering, pivot_u <= kSlackMaxPivotU, enough pairs), the plan with the
/// reduced matrix's analysis, attached to an analysis of the full matrix in
/// the order pairs first, then the reduced matrix's (no ordering of its own;
/// it describes the matrix, factorizations off the plan take
/// slack_full_analysis()). Null otherwise: analyse() then runs as without
/// the plan.
std::unique_ptr<AnalysisKeep> analyse_with_slack_plan(Solver &solver, const CscLower &A,
                                                      const Control &ctrl, Info &info);

/// The analysis a factorization of `keep` that does not run through its plan
/// takes: the plan's full analysis (made on first use), `keep` itself
/// without a plan.
const AnalysisKeep &slack_full_analysis(const AnalysisKeep &keep);

/// The plan's full analysis (made on first use; null when it fails).
const AnalysisKeep *slack_plan_full_analysis(const SlackPlan &plan);

/// Whether `analysis` is one of `keep`'s plan's analyses (reduced or full).
bool slack_owns(const AnalysisKeep &keep, const AnalysisKeep *analysis);

/// Whether a factor() of `keep` with `ctrl` runs through its plan.
bool slack_armed(const AnalysisKeep &keep, const Control &ctrl);

/// The pair pass: the reduced values (`rv`, the reduced matrix's input
/// order) and the pairs' inverses of cleaned values `cv`; adds the pairs'
/// inertia to pos / neg. False when a pair's 2x2 pivot fails the growth
/// guard (Control::slack_update_limit) or is singular.
bool slack_pair_pass(const SlackPlan &plan, const double *cv, const Control &ctrl,
                     std::vector<double> &rv, SlackFactor &sf, int &pos, int &neg);

namespace detail {
/// Test seam: the solve-time full refactor reports failure (Singular).
inline std::atomic<bool> slack_force_full_refactor_failure{false};
} // namespace detail

} // namespace smf
