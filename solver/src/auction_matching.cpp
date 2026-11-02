#include "auction_matching.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace smf::detail {

namespace {

// The FIFO queue drops its popped front once at least this many entries
// were popped and they are half of it: a column is queued only while free,
// so the live part never exceeds the sparse columns and the vector stays
// within about twice that. The value only bounds how often it compacts.
constexpr std::size_t kQueueCompactMin = 1024;

// validated()'s clamps on valid options, which bound the auction's work:
// the phases number log(eps0 / eps_final) / log(eps_factor), eps0 being the
// largest sparse cost (up to ~1500 log units on the test corpus). At 1.5
// (Bertsekas recommends 4-10) and 1e-12 that is at most 87 phases. 1e-12
// is about four ulps of a price near 1500: a smaller eps no longer moves
// prices. The bid cap times the sparse columns (< 2^31) must fit a long
// long; 1e6 is 2e4x the default 50. An explicit eps_start above 1e3 log
// units (e^1000, past double's whole range of ~1418) is no tighter a start
// than the max-cost rule's and only adds phases: at 1e3 and the clamps
// above, at most log(1e3 / 1e-12) / log(1.5) = 86 phases.
constexpr double kMinEpsFactor = 1.5;
constexpr double kMinEpsFinal = 1e-12;
constexpr double kMaxBidCap = 1e6;
constexpr double kMaxEpsStart = 1e3;

} // namespace

AuctionOptions validated(const AuctionOptions &opt, const AuctionOptions &def) {
    AuctionOptions o = opt;
    // NaN fails every comparison, so it takes the default too.
    if (!(std::isfinite(o.eps_factor) && o.eps_factor > 1.0)) o.eps_factor = def.eps_factor;
    if (!(std::isfinite(o.eps_final) && o.eps_final > 0.0)) o.eps_final = def.eps_final;
    if (!(std::isfinite(o.bid_cap) && o.bid_cap > 0.0)) o.bid_cap = def.bid_cap;
    o.eps_factor = std::max(o.eps_factor, kMinEpsFactor);
    o.eps_final = std::max(o.eps_final, kMinEpsFinal);
    o.bid_cap = std::min(o.bid_cap, kMaxBidCap);
    // Not finite or <= 0: def's (Hybrid: 0.1; AuctionOptions{}: 0, the
    // max-cost rule, kept for the kernel tests).
    if (!(std::isfinite(o.eps_start) && o.eps_start > 0.0)) o.eps_start = def.eps_start;
    o.eps_start = std::min(o.eps_start, kMaxEpsStart);
    if (!(std::isfinite(o.switch_passes) && o.switch_passes >= 0.0)) o.switch_passes = def.switch_passes;
    return o;
}

// A forward (Gauss-Seidel) auction in min-cost form (Bertsekas): columns bid
// for rows, and a row's price is -u_i. Column j's bid takes its cheapest row
// i1 (b1 = min_i c_ij - u_i, b2 the second smallest) and sets
//   v_j = b2,  u_i1 = c_i1j - b2 - eps,
// so row i1's price rises by (b2 - b1) + eps, and the column it displaces
// goes to the back of the queue. Bids keep eps-CS, and with it the duals
// stay feasible: u only falls, and every edge of j but the won one is at
// least b2 after the bid. Epsilon-scaling divides eps by eps_factor per
// phase; each phase unmatches the columns its smaller eps makes too loose.
AuctionStatus auction_advance(const CostGraph &g, const std::uint8_t *dense,
                              std::vector<RowState> &rs, std::vector<double> &v,
                              std::vector<int> &row_match, std::vector<Int> &col_to_row,
                              const std::vector<int> &first_queue, const AuctionOptions &opt,
                              AuctionState &st, long long &work, long long work_budget) {
    assert(opt.eps_factor > 1.0 && opt.eps_final > 0.0);
    const int n = static_cast<int>(col_to_row.size());
    const Int *const gptr = g.ptr.data();
    const int *const grow = g.row.data();
    const double *const gcost = g.cost.data();
    RowState *const R = rs.data();
    const auto is_dense = [dense](int i) { return dense[i] != 0; };

    if (st.phase == -1) {
        // The sparse part's cost bounds: eps0 = max cost / eps_factor (one
        // huge cost only lengthens the first phases), and the range is the
        // second-best value of a column with a single sparse row.
        double cmax = 0.0, cmin = kInf;
        long long sparse_cols = 0;
        for (int j = 0; j < n; ++j) {
            if (is_dense(j)) continue;
            ++sparse_cols;
            work += gptr[j + 1] - gptr[j];
            for (Int q = gptr[j]; q < gptr[j + 1]; ++q) {
                if (is_dense(grow[q])) continue;
                cmax = std::max(cmax, gcost[q]);
                cmin = std::min(cmin, gcost[q]);
            }
        }
        st.cost_range = cmin == kInf ? 0.0 : cmax - cmin;
        st.bid_limit = static_cast<long long>(std::ceil(opt.bid_cap * static_cast<double>(sparse_cols)));
        // eps_start > 0: an explicit first epsilon (Hybrid: warm from exact duals).
        st.eps = std::max(opt.eps_start > 0.0 ? opt.eps_start : cmax / opt.eps_factor, opt.eps_final);
        st.queue.assign(first_queue.begin(), first_queue.end());
        st.head = 0;
        st.bids_in_phase = 0;
        st.phase = 0;
    }

    for (;;) {
        if (st.head == st.queue.size()) {
            if (st.eps <= opt.eps_final) return st.dropped ? AuctionStatus::Capped : AuctionStatus::Done;
            // Next phase: the matched columns whose edge is more than the
            // new eps above their column's minimum become free again.
            st.eps = std::max(st.eps / opt.eps_factor, opt.eps_final);
            ++st.phase;
            st.queue.clear();
            st.head = 0;
            st.bids_in_phase = 0;
            for (int j = 0; j < n; ++j) {
                const Int r = col_to_row[static_cast<std::size_t>(j)];
                if (is_dense(j) || r < 0) continue;
                work += gptr[j + 1] - gptr[j];
                double col_min = kInf, matched = kInf;
                for (Int q = gptr[j]; q < gptr[j + 1]; ++q) {
                    const int i = grow[q];
                    if (is_dense(i)) continue;
                    const double d = gcost[q] - R[i].u;
                    col_min = std::min(col_min, d);
                    if (i == r) matched = d;
                }
                if (matched - col_min > st.eps) {
                    row_match[static_cast<std::size_t>(r)] = -1;
                    col_to_row[static_cast<std::size_t>(j)] = -1;
                    st.queue.push_back(j);
                }
            }
            continue;
        }
        // A sparse part without a perfect matching (structurally singular,
        // or needing the dense rows) never ends a phase: prices fall
        // without bound, so the cap ends it.
        if (st.bids_in_phase >= st.bid_limit) return AuctionStatus::Capped;
        if (work_budget >= 0 && work >= work_budget) return AuctionStatus::Paused;

        const int j = st.queue[st.head++];
        assert(!is_dense(j) && col_to_row[static_cast<std::size_t>(j)] < 0);
        work += gptr[j + 1] - gptr[j];
        // Ties in b1 go to the lowest position in the column.
        double b1 = kInf, b2 = kInf;
        int i1 = -1;
        Int q1 = -1;
        for (Int q = gptr[j]; q < gptr[j + 1]; ++q) {
            const int i = grow[q];
            if (is_dense(i)) continue;
            const double d = gcost[q] - R[i].u;
            if (d < b1) {
                b2 = b1;
                b1 = d;
                i1 = i;
                q1 = q;
            } else if (d < b2) {
                b2 = d;
            }
        }
        if (i1 < 0) {
            // No sparse row: the column can only be matched to a dense row,
            // so the sparse part has no perfect matching. It leaves the
            // queue free (no bid, never displaced, so never queued again)
            // for the searches, and the auction ends Capped.
            st.dropped = true;
            continue;
        }
        ++st.bids_in_phase;
        // A single sparse row has no second best: the bid raises its price
        // by the sparse part's cost range (at least eps), a finite step, so
        // single-entry columns competing for one row (no perfect matching)
        // bid until the cap, with prices falling by at least that per bid.
        if (b2 == kInf) b2 = b1 + std::max(st.eps, st.cost_range);
        v[static_cast<std::size_t>(j)] = b2;
        R[i1].u = gcost[q1] - b2 - st.eps;
        const int k = row_match[static_cast<std::size_t>(i1)];
        if (k >= 0) {
            col_to_row[static_cast<std::size_t>(k)] = -1;
            st.queue.push_back(k);
        }
        row_match[static_cast<std::size_t>(i1)] = j;
        col_to_row[static_cast<std::size_t>(j)] = i1;
        if (st.head >= kQueueCompactMin && 2 * st.head >= st.queue.size()) {
            st.queue.erase(st.queue.begin(), st.queue.begin() + static_cast<std::ptrdiff_t>(st.head));
            st.head = 0;
        }
    }
}

} // namespace smf::detail

namespace smf {

// The hybrid probe's auction settings (the w40 run; docs/performance/
// 2026-11-02-auction-matching.md, "The probe"), the Hybrid defaults;
// switch_passes keeps the struct's 40 and bid_cap its 50.
AuctionOptions hybrid_auction_options() {
    AuctionOptions o;
    o.eps_factor = 10.0;
    o.eps_final = 1e-2;
    o.eps_start = 0.1;
    return o;
}

} // namespace smf
