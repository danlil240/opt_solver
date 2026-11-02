#pragma once

// Epsilon-scaling auction on the sparse part of the matching's cost graph:
// a fast approximate (eps-optimal) start for MatchingAlgorithm::Auction,
// whose exact shortest-path repair then finishes the matching.

#include "matching_internal.hpp"

#include "smf/scaling_matching.hpp"
#include "smf/types.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace smf::detail {

enum class AuctionStatus { Done, Paused, Capped };

struct AuctionState {        // resumable: everything advance needs to continue
    double eps = 0.0;
    int    phase = -1;       // -1: not started
    std::vector<int> queue;  // free sparse columns, FIFO
    std::size_t head = 0;
    long long bids_in_phase = 0;
    // Set when the auction starts, from one pass over the sparse part:
    double cost_range = 0.0;  // max - min sparse cost (single-entry columns' bid)
    long long bid_limit = 0;  // opt.bid_cap x sparse columns, bids per phase
    bool dropped = false;     // a column without sparse rows left the queue
};

// opt with every field valid for auction_advance: a non-finite or
// out-of-domain field takes def's value (AuctionOptions{} for Auction,
// hybrid_auction_options() for Hybrid), and the valid ones are clamped
// (scaling_matching.hpp states the rule). SymmetricMatcher applies it at the
// public boundary.
AuctionOptions validated(const AuctionOptions &opt, const AuctionOptions &def = AuctionOptions{});

// Runs bids until every sparse column is matched in the last phase (Done),
// work >= work_budget at a bid boundary (Paused; work_budget < 0: none),
// or a phase exceeds opt.bid_cap x sparse columns bids (Capped: the
// columns left free stay free). A column without a sparse row leaves the
// queue unmatched and the others go on; the last phase then ends Capped
// instead of Done. Once Done or Capped, later calls return the same status
// and change nothing. Dense rows/columns (dense[i] != 0) are invisible.
// u: row duals, v: column duals, row_match / col_to_row: the matching, all
// updated in place; work += every scanned edge.
//
// Preconditions (SymmetricMatcher's constructor and advance() uphold them):
// the incoming matching pairs only sparse columns with sparse rows,
// row_match and col_to_row agree, and first_queue lists exactly the free
// sparse columns (in the search order); it is read when st.phase == -1. The
// incoming duals are feasible on every sparse edge. opt.eps_factor > 1,
// opt.eps_final > 0 and opt.bid_cap > 0, all finite (validated()).
//
// On return the duals are feasible on every sparse edge (c - u - v >= 0)
// and each matched edge is within eps of its column's minimum of c - u
// (eps-CS), so the matching's cost is within (sparse columns) x eps of the
// optimum once Done. (A matched edge is eps above v right after its bid,
// but a warm-start edge is tight and an edge kept across phases keeps an
// earlier, larger eps, so no tighter bound on c - u - v holds.)
AuctionStatus auction_advance(const CostGraph &g, const std::uint8_t *dense,
                              std::vector<RowState> &rs, std::vector<double> &v,
                              std::vector<int> &row_match, std::vector<Int> &col_to_row,
                              const std::vector<int> &first_queue, const AuctionOptions &opt,
                              AuctionState &st, long long &work, long long work_budget);

} // namespace smf::detail
