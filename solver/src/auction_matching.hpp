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
};

// Runs bids until every sparse column is matched in the last phase (Done),
// work >= work_budget at a bid boundary (Paused; work_budget < 0: none),
// or a phase exceeds opt.bid_cap x sparse columns bids (Capped: the
// columns left free stay free). Dense rows/columns (dense[i] != 0) are
// invisible. u: row duals, v: column duals, row_match / col_to_row: the
// matching, all updated in place; work += every scanned edge.
//
// first_queue holds the free sparse columns in the trial's search order; it
// is read when st.phase == -1. Requires opt.eps_factor > 1 and
// opt.eps_final > 0. On return the duals are feasible on every sparse edge
// (c - u - v >= 0), each matched edge eps above its column's v and within
// eps of its column's minimum of c - u (eps-CS), so the matching's cost is
// within (sparse columns) x eps of the optimum once Done.
AuctionStatus auction_advance(const CostGraph &g, const std::uint8_t *dense,
                              std::vector<RowState> &rs, std::vector<double> &v,
                              std::vector<int> &row_match, std::vector<Int> &col_to_row,
                              const std::vector<int> &first_queue, const AuctionOptions &opt,
                              AuctionState &st, long long &work, long long work_budget);

} // namespace smf::detail
