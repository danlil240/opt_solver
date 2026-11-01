#pragma once

#include "smf/csc_matrix.hpp"
#include "smf/error.hpp"
#include "smf/types.hpp"

#include <memory>
#include <vector>

namespace smf {

struct MatchingResult {
    ErrorCode status = ErrorCode::Success;
    std::vector<double> scale;      ///< positive symmetric diagonal scaling S (length n)
    std::vector<Int> row_to_col;    ///< matched column for each row i (or -1 if unmatched)
    std::vector<Int> col_to_row;    ///< matched row for each col j (or -1 if unmatched)
    int matched_rank = 0;           ///< number of matched columns
};

/// Compute maximum-product bipartite matching and symmetric positive diagonal scaling
/// S = diag(scale) such that S * A * S has unit-bounded entries and unit-magnitude
/// matched entries, preserving symmetry and Sylvester inertia.
/// Returns ErrorCode::SingularMatrix if A is structurally singular (matched_rank < n).
MatchingResult compute_symmetric_matching(const CscLower &A);

/// compute_symmetric_matching in resumable steps. The constructor runs the
/// cheap initial phase (MC64's greedy matching on tight edges plus length-2
/// augmentations); advance() runs the shortest augmenting path searches,
/// optionally under a work budget, and resumes where it stopped. Run to the
/// end in MatchingSearchOrder::Column, the result is bitwise
/// compute_symmetric_matching's.
///
/// The searches start from the free columns in column order, or with
/// MatchingSearchOrder::Degree by decreasing degree (stable). Both give a
/// maximum-product matching; they differ where ties let several be optimal.
/// Degree order scans 3-11x fewer edges on KKT systems whose constraint
/// columns come last (case39, the TSOPF_FS family, cops_catmix_800).
enum class MatchingSearchOrder { Column, Degree };

class SymmetricMatcher {
public:
    explicit SymmetricMatcher(const CscLower &A, MatchingSearchOrder order = MatchingSearchOrder::Column);
    ~SymmetricMatcher();
    SymmetricMatcher(const SymmetricMatcher &) = delete;
    SymmetricMatcher &operator=(const SymmetricMatcher &) = delete;

    /// Success, or why the matching failed (IllegalValue, SingularMatrix).
    ErrorCode status() const;
    /// Runs searches until none is left (true; status() says whether the
    /// matching is perfect) or, with work_budget >= 0, until the edges
    /// scanned by all searches so far reach work_budget (false). The budget
    /// is checked between searches, so stopping never changes the result.
    bool advance(long long work_budget = -1);
    /// Edges scanned by the searches so far.
    long long work() const;
    /// Whether the initial phase matched every column but the dense ones
    /// (more than 10 sqrt(n) entries) to its own diagonal. The searches left
    /// are then one per dense column (phase 3).
    bool diagonal() const;
    /// Requires diagonal(): the diagonal matching, the dense columns left
    /// free, and the scaling of its duals, the dense columns' made feasible
    /// (as before the searches over them). No search runs.
    MatchingResult diagonal_result();
    /// The current matching (free columns and rows -1 while searches are
    /// left) and the scaling of the current duals, as
    /// compute_symmetric_matching reports them; unit scaling once failed.
    MatchingResult current() const;
    /// The initial phase's partial matching and the scaling of its duals.
    MatchingResult initial() const;

private:
    struct State;
    std::unique_ptr<State> s_;
};

ErrorCode compute_matching_scale(const CscLower &A,
                                 std::vector<double> &scale,
                                 int print_level = 0);

} // namespace smf
