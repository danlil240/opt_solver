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

/// Tuning of the epsilon-scaling auction on the sparse part (the
/// MatchingAlgorithm::Auction matcher, ahead of its exact shortest-path
/// repair). All costs and epsilons are in log units (log col_max - log|a|).
/// The defaults are provisional: Task 4 of the auction-matching plan tunes
/// them on edges scanned over the 50-matrix corpus.
struct AuctionOptions {
    /// Epsilon divides by this per scaling phase. 5 is in the 4-10 range
    /// Bertsekas recommends for epsilon-scaling; provisional.
    double eps_factor = 5.0;
    /// The last phase's epsilon, in log units: the matching's product is
    /// within n * eps_final of the optimum (1e-3: a factor 1.001 per column)
    /// before the repair makes it exact; provisional.
    double eps_final = 1e-3;
    /// Start from the constructor's duals and greedy matching (0-CS); false:
    /// from its duals and an empty matching; provisional.
    bool warm = true;
    /// Max bids per phase, in units of the sparse part's columns. Bounds the
    /// price war a structurally singular sparse part starts (prices fall
    /// without bound). 50 is 7x the busiest phase of test_auction_matching's
    /// KKT matrices (6.7 bids per column, opf_like_kkt 3000/1200); a
    /// singular sparse part costs 50 bids per column before the searches
    /// take over; provisional.
    double bid_cap = 50.0;
    /// The first phase's epsilon, in log units; <= 0 (the default): the old
    /// rule, max sparse cost / eps_factor. Probe option for
    /// SymmetricMatcher::probe_switch_to_auction (bench_matching_trial
    /// --hybrid-probe): from exact duals the old eps0 would wreck them.
    double eps_start = 0.0;
};
// SymmetricMatcher validates its AuctionOptions: a field that is not finite
// or outside its domain (eps_factor <= 1, eps_final <= 0, bid_cap <= 0)
// takes its default, and the valid ones are clamped to eps_factor >= 1.5,
// eps_final >= 1e-12 and bid_cap <= 1e6 (auction_matching.cpp), so the
// auction's phases and bids stay bounded.

class SymmetricMatcher {
public:
    /// alg: the sparse part's engine (MatchingAlgorithm, types.hpp); auction
    /// tunes MatchingAlgorithm::Auction (nullptr: the defaults) and is
    /// copied. The constructor's initial phase is the same for both.
    explicit SymmetricMatcher(const CscLower &A, MatchingSearchOrder order = MatchingSearchOrder::Column,
                              MatchingAlgorithm alg = MatchingAlgorithm::ShortestPath,
                              const AuctionOptions *auction = nullptr);
    ~SymmetricMatcher();
    SymmetricMatcher(const SymmetricMatcher &) = delete;
    SymmetricMatcher &operator=(const SymmetricMatcher &) = delete;

    /// Success, or why the matching failed (IllegalValue, SingularMatrix).
    ErrorCode status() const;
    /// Runs searches until none is left (true; status() says whether the
    /// matching is perfect) or, with work_budget >= 0, until the edges
    /// scanned so far (work()) reach work_budget (false). The budget
    /// is checked between searches (and under Auction between bids), so
    /// stopping never changes the result.
    bool advance(long long work_budget = -1);
    /// From then on advance() ends the searches once at most max_free
    /// columns are free (checked before each search over the sparse part;
    /// negative, the default: all of them run). current() then reports that
    /// partial matching and the scaling of its duals with status Success: a
    /// column left free may have no matching at all. MatchingAlgorithm::
    /// ShortestPath only: under Auction it asserts in debug builds and is
    /// ignored in release.
    void set_stop_free(int max_free);
    /// Edges scanned so far: by the searches, and under Auction also by the
    /// auction's bids and phase passes and by the repair's pass.
    long long work() const;
    /// Under Auction, the sparse columns the repair left free and handed to
    /// the searches (0 before the repair ran); 0 under ShortestPath.
    int repaired() const;
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

    // ---- Probe hooks: bench_matching_trial --hybrid-probe only. Nothing in
    // analyse() calls them; a matcher on which they are never called behaves
    // bitwise as before. ----
    /// ShortestPath, phase 1: the sparse columns free now; -1 otherwise.
    int probe_free_sparse() const;
    /// Shortest-path searches run so far (phases 1 and 3, failed ones too).
    long long probe_searches() const;
    /// From then on advance() also pauses (returns false) before a phase-1
    /// search once at most max_free sparse columns are free (negative: off).
    /// probe_switch_to_auction() clears it.
    void probe_pause_at_free(int max_free);
    /// On a ShortestPath matcher paused in phase 1 (advance() returned
    /// false): the next advance() first runs the auction (forced warm) on the
    /// free sparse columns not yet searched, from the current duals and
    /// matching (0-CS), then repair() (rows reachable from the columns left
    /// free get the duals of this call back, not the constructor's), drops the
    /// columns whose phase-1 search already failed from the relisted search
    /// order, and continues the normal phase-1 searches and phases 2-3. All of
    /// it counts in work(); repaired() reports the columns repair() left free.
    /// Ignored outside phase 1 or under MatchingAlgorithm::Auction.
    void probe_switch_to_auction(const AuctionOptions &opt);

private:
    struct State;
    std::unique_ptr<State> s_;
};

ErrorCode compute_matching_scale(const CscLower &A,
                                 std::vector<double> &scale,
                                 int print_level = 0);

} // namespace smf
