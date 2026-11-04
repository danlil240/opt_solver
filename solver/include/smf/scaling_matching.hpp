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

/// Tuning of the epsilon-scaling auction that MatchingAlgorithm::Hybrid runs
/// on the search tail, ahead of its exact shortest-path repair. All costs
/// and epsilons are in log units (log col_max - log|a|). The struct's own
/// defaults are the auction kernel's (the kernel tests use them); Hybrid's
/// are hybrid_auction_options(). The auction always starts warm, from the
/// searches' duals and matching (0-CS).
struct AuctionOptions {
    /// Epsilon divides by this per scaling phase. 5 is in the 4-10 range
    /// Bertsekas recommends for epsilon-scaling.
    double eps_factor = 5.0;
    /// The last phase's epsilon, in log units: the matching's product is
    /// within n * eps_final of the optimum (1e-3: a factor 1.001 per column)
    /// before the repair makes it exact.
    double eps_final = 1e-3;
    /// Max bids per phase, in units of the sparse part's columns. Bounds the
    /// price war a structurally singular sparse part starts (prices fall
    /// without bound). 50 is 7x the busiest phase of test_auction_matching's
    /// KKT matrices (6.7 bids per column, opf_like_kkt 3000/1200); a
    /// singular sparse part costs 50 bids per column before the searches
    /// take over; provisional.
    double bid_cap = 50.0;
    /// The first phase's epsilon, in log units; <= 0 (the struct's default,
    /// the kernel's own rule): max sparse cost / eps_factor. Hybrid starts
    /// warm from the searches' exact duals, which that eps0 would wreck: 0.1
    /// (hybrid_auction_options()) is the probe's best of {0.01, 0.1, 1}
    /// (docs/performance/2026-11-02-auction-matching.md, "The probe"; 0.01,
    /// a single phase at eps_final, scans 1.1x the plain searches' edges).
    double eps_start = 0.0;
    /// MatchingAlgorithm::Hybrid only: the phase-1 search work, in passes
    /// (1 pass = 2 x the lower triangle's entries), past which the auction
    /// takes the free columns left. Checked between searches on the work
    /// alone (work() > switch_passes x pass), so the switch point does not
    /// depend on how advance() is budgeted. 4 (2026-11-04 sweep of
    /// {40, 16, 8, 4} on the front tier, 12900K, 1T and 8T, both tracks,
    /// after the canonical duals): wall geomean 0.988 (1T) / 0.985 (8T),
    /// hard_indef 0.976-0.965, no case above 1.10 after rechecks, no
    /// outcome or inertia change; only cvxqp3 and ncvxqp1/3/7 change their
    /// ordering, ncvxqp5 switches with the same one (8T 0.75x). The
    /// earlier 40 came from the hybrid probe (rule w40, before the
    /// canonical duals; write-up "The probe"), which counted matcher edges
    /// alone: an early switch pays the auction's fixed passes on matrices
    /// the searches finish quickly (pglib_opf_case300_ieee: 13.4 passes
    /// plain, 17.8 at w5), which the front-tier wall geomean does not show.
    double switch_passes = 4.0;
};
/// The Hybrid defaults (SymmetricMatcher with MatchingAlgorithm::Hybrid and
/// auction == nullptr): eps_factor 10, eps_final 1e-2, eps_start 0.1,
/// switch_passes 4, bid_cap 50 (the auction settings of the probe's w40
/// run, docs/performance/2026-11-02-auction-matching.md, "The probe", with
/// the switch moved to 4 passes by the 2026-11-04 sweep).
AuctionOptions hybrid_auction_options();
// SymmetricMatcher validates its AuctionOptions under Hybrid: a field that
// is not finite or outside its domain (eps_factor <= 1, eps_final <= 0,
// bid_cap <= 0, eps_start <= 0, switch_passes < 0) takes
// hybrid_auction_options()'s value, and the valid ones are clamped to
// eps_factor >= 1.5, eps_final >= 1e-12, eps_start <= 1e3 and bid_cap <= 1e6
// (auction_matching.cpp), so the auction's phases and bids stay bounded.

class SymmetricMatcher {
public:
    /// alg: the sparse part's engine (MatchingAlgorithm, types.hpp); auction
    /// tunes MatchingAlgorithm::Hybrid (nullptr: hybrid_auction_options())
    /// and is copied; ShortestPath ignores it. The constructor's initial
    /// phase is the same for both.
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
    /// is checked between searches (and under Hybrid's auction between
    /// bids), so stopping never changes the result.
    bool advance(long long work_budget = -1);
    /// From then on advance() ends the searches once at most max_free
    /// columns are free (checked before each search over the sparse part;
    /// negative, the default: all of them run). current() then reports that
    /// partial matching and the scaling of its duals with status Success: a
    /// column left free may have no matching at all. MatchingAlgorithm::
    /// ShortestPath only: under Hybrid (whose auction matches the tail as a
    /// whole) a max_free >= 0 asserts in debug builds and is ignored in
    /// release; a negative one is a no-op under both.
    void set_stop_free(int max_free);
    /// Edges scanned so far: by the searches, and under Hybrid after a
    /// switch also by the auction's bids and phase passes, the repair's pass
    /// and the canonical duals' pass.
    long long work() const;
    /// Under Hybrid, the sparse columns the repair left free and handed to
    /// the searches (0 before a switch); 0 under ShortestPath.
    int repaired() const;
    /// Under Hybrid, whether the auction took over phase 1 (its searches
    /// passed switch_passes with free columns left); false otherwise. After a
    /// switch, phase 1 ends by moving the sparse duals to the largest ones
    /// below the duals at the switch that keep the matching optimal (one
    /// pass): the duals the ShortestPath searches end at, so the scale is
    /// ShortestPath's to rounding (docs/performance/
    /// 2026-11-02-auction-matching.md, "Re-gate with canonical duals";
    /// AuctionMatching.HybridCanonicalDualsMatchShortestPathScale).
    bool switched() const;
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
