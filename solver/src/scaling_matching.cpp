#include "smf/scaling_matching.hpp"

#include "auction_matching.hpp"
#include "indexed_min_heap.hpp"
#include "matching_internal.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace smf {

namespace detail {

ErrorCode build_cost_graph(const CscLower &A, CostGraph &g) {
    const int n = static_cast<int>(A.n);
    std::vector<Int> count(static_cast<std::size_t>(n) + 1, 0);
    // Cleaned input (each column's rows strictly increasing, none above the
    // diagonal, as analyse() passes it) expands without duplicates: column
    // j gets rows jj < j from the earlier columns, then its own rows >= j.
    // Its nonzeros per column and largest |a_ij| are counted here as well.
    bool cleaned = true;
    g.ptr.assign(static_cast<std::size_t>(n) + 1, 0);
    g.col_max.assign(static_cast<std::size_t>(n), 0.0);
    Int *const nz_count = g.ptr.data() + 1;
    double *const cmax = g.col_max.data();
    for (int j = 0; j < n; ++j) {
        const Int p0 = A.col_ptr[static_cast<std::size_t>(j)];
        for (Int p = p0; p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
            const int i = static_cast<int>(A.row_idx[static_cast<std::size_t>(p)]);
            const double av = std::abs(A.values[static_cast<std::size_t>(p)]);
            if (!std::isfinite(av)) {
                return ErrorCode::IllegalValue;
            }
            if (i < j || (p > p0 && i <= A.row_idx[static_cast<std::size_t>(p) - 1])) cleaned = false;
            ++count[static_cast<std::size_t>(j) + 1];
            if (i != j) ++count[static_cast<std::size_t>(i) + 1];
            if (av > 0.0) {
                ++nz_count[j];
                cmax[j] = std::max(cmax[j], av);
                if (i != j) {
                    ++nz_count[i];
                    cmax[i] = std::max(cmax[i], av);
                }
            }
        }
    }
    if (cleaned) {
        // No duplicates to merge: the nonzeros go straight to their slots
        // with their costs, one log per stored entry, in the order the
        // expansion below would leave them (bitwise the same graph).
        for (int j = 0; j < n; ++j) {
            if (cmax[j] <= 0.0) return ErrorCode::SingularMatrix;
            g.ptr[static_cast<std::size_t>(j) + 1] += g.ptr[static_cast<std::size_t>(j)];
        }
        const std::size_t nnz = static_cast<std::size_t>(g.ptr[static_cast<std::size_t>(n)]);
        g.row.resize(nnz);
        g.cost.resize(nnz);
        std::vector<double> log_cmax(static_cast<std::size_t>(n));
        for (int j = 0; j < n; ++j) log_cmax[static_cast<std::size_t>(j)] = std::log(cmax[j]);
        std::vector<Int> next(g.ptr.begin(), g.ptr.end() - 1);
        int *const grow = g.row.data();
        double *const gcost = g.cost.data();
        for (int j = 0; j < n; ++j) {
            const double lj = log_cmax[static_cast<std::size_t>(j)];
            for (Int p = A.col_ptr[static_cast<std::size_t>(j)];
                 p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
                const double av = std::abs(A.values[static_cast<std::size_t>(p)]);
                if (!(av > 0.0)) continue;
                const int i = static_cast<int>(A.row_idx[static_cast<std::size_t>(p)]);
                const double la = std::log(av);
                const std::size_t qj = static_cast<std::size_t>(next[static_cast<std::size_t>(j)]++);
                grow[qj] = i;
                gcost[qj] = std::max(0.0, lj - la);
                if (i != j) {
                    const std::size_t qi = static_cast<std::size_t>(next[static_cast<std::size_t>(i)]++);
                    grow[qi] = j;
                    gcost[qi] = std::max(0.0, log_cmax[static_cast<std::size_t>(i)] - la);
                }
            }
        }
        return ErrorCode::Success;
    }
    for (int j = 0; j < n; ++j) {
        count[static_cast<std::size_t>(j) + 1] += count[static_cast<std::size_t>(j)];
    }
    const std::size_t nnz = static_cast<std::size_t>(count[static_cast<std::size_t>(n)]);
    g.ptr.assign(static_cast<std::size_t>(n) + 1, 0);
    g.row.resize(nnz);
    g.cost.resize(nnz);
    g.col_max.assign(static_cast<std::size_t>(n), 0.0);
    std::vector<int> raw_row_buf(nnz);
    std::vector<double> raw_val_buf(nnz);
    int *const raw_row = raw_row_buf.data();
    double *const raw_val = raw_val_buf.data();
    std::vector<Int> next(count.begin(), count.end() - 1);
    for (int j = 0; j < n; ++j) {
        for (Int p = A.col_ptr[static_cast<std::size_t>(j)];
             p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
            const int i = static_cast<int>(A.row_idx[static_cast<std::size_t>(p)]);
            const double v = A.values[static_cast<std::size_t>(p)];
            const std::size_t qj = static_cast<std::size_t>(next[static_cast<std::size_t>(j)]++);
            raw_row[qj] = i;
            raw_val[qj] = v;
            if (i != j) {
                const std::size_t qi = static_cast<std::size_t>(next[static_cast<std::size_t>(i)]++);
                raw_row[qi] = j;
                raw_val[qi] = v;
            }
        }
    }

    // Merge duplicates per column (summing them, like the factorization
    // does) with a last-seen-column marker, then drop zeros. The marker is
    // the column, not the output slot: dropping a column's zeros moves `out`
    // back, so slots left by the previous column can lie past the next
    // column's start and would swallow its entries as false duplicates.
    std::vector<int> seen_in(static_cast<std::size_t>(n), -1);
    std::vector<Int> slot_of(static_cast<std::size_t>(n), -1);
    Int out = 0;
    for (int j = 0; j < n; ++j) {
        const Int begin = out;
        for (Int q = count[static_cast<std::size_t>(j)]; q < count[static_cast<std::size_t>(j) + 1]; ++q) {
            const int i = raw_row[static_cast<std::size_t>(q)];
            if (seen_in[static_cast<std::size_t>(i)] == j) {
                g.cost[static_cast<std::size_t>(slot_of[static_cast<std::size_t>(i)])] +=
                    raw_val[static_cast<std::size_t>(q)];
            } else {
                seen_in[static_cast<std::size_t>(i)] = j;
                slot_of[static_cast<std::size_t>(i)] = out;
                g.row[static_cast<std::size_t>(out)] = i;
                g.cost[static_cast<std::size_t>(out)] = raw_val[static_cast<std::size_t>(q)];
                ++out;
            }
        }
        // g.cost holds the merged value for now; compact out zeros.
        Int w = begin;
        double col_max = 0.0;
        for (Int q = begin; q < out; ++q) {
            const double av = std::abs(g.cost[static_cast<std::size_t>(q)]);
            if (av > 0.0) {
                g.row[static_cast<std::size_t>(w)] = g.row[static_cast<std::size_t>(q)];
                g.cost[static_cast<std::size_t>(w)] = av;
                col_max = std::max(col_max, av);
                ++w;
            }
        }
        out = w;
        if (col_max <= 0.0) {
            return ErrorCode::SingularMatrix;
        }
        g.col_max[static_cast<std::size_t>(j)] = col_max;
        const double log_cmax = std::log(col_max);
        for (Int q = begin; q < out; ++q) {
            double &c = g.cost[static_cast<std::size_t>(q)];
            c = std::max(0.0, log_cmax - std::log(c));
        }
        g.ptr[static_cast<std::size_t>(j) + 1] = out;
    }
    g.row.resize(static_cast<std::size_t>(out));
    g.cost.resize(static_cast<std::size_t>(out));
    return ErrorCode::Success;
}

} // namespace detail

using detail::CostGraph;
using detail::RowState;
using detail::build_cost_graph;
using detail::kInf;

// Maximum-product perfect matching (MC64 job 5): minimum-cost assignment on
// the costs of CostGraph (matching_internal.hpp), solved by shortest
// augmenting paths with dual potentials u[row], v[col] (reduced cost
// c - u - v >= 0, zero on matched edges).
// As in MC64, a greedy matching on zero-reduced-cost edges (plus length-2
// augmentations) settles most columns, and each remaining column runs a
// heap-based Dijkstra that touches, and afterwards resets, only the rows it
// reaches. Under MatchingAlgorithm::Auction an epsilon-scaling auction
// (auction_matching.hpp) first matches the sparse part approximately, and
// repair() hands what it left loose to the same searches. Under Hybrid the
// searches run first; past AuctionOptions::switch_passes of their work the
// auction takes the free columns left (hybrid_step()).
struct SymmetricMatcher::State {
    int n = 0;
    bool shape_ok = true;
    ErrorCode status = ErrorCode::Success;
    CostGraph g;
    std::vector<RowState> rs;
    std::vector<int> row_match;           // matched column of each row, or -1
    std::vector<int> row_parent;          // column the search reached the row from
    std::vector<std::uint8_t> row_done;   // finalized by the current search
    std::vector<std::uint8_t> row_dense;  // dense row/column (see the constructor)
    std::vector<double> v;  // column duals
    std::vector<Int> col_to_row;
    int matched = 0;
    bool any_dense = false;
    // Searches left: phase 1 (sparse part) from search_order[next_col] on,
    // 3 (full graph) from column next_col on; 4 once none is left.
    int phase = 1;
    int next_col = 0;
    MatchingSearchOrder order = MatchingSearchOrder::Column;
    MatchingAlgorithm alg = MatchingAlgorithm::ShortestPath;
    AuctionOptions auction_opt;      // validated (Hybrid: warm)
    detail::AuctionState auction;    // the auction's resumable state
    bool repair_done = false;        // Auction: the auction and repair() ran
    int repaired = 0;                // Auction, Hybrid: columns repair() left free
    // Hybrid: phase 1 switches to the auction before the first search at
    // which work > switch_work (switch_passes x 2 x lower entries).
    double switch_work = 0.0;
    bool switched = false;           // Hybrid: the switch fired
    // Phase 1's free sparse columns in the search order, listed when
    // phase 1 starts (under Auction: the auction's first queue, then
    // relisted by repair()).
    std::vector<int> search_order;
    int stop_free = -1;  // set_stop_free()
    long long work = 0;  // edges scanned (searches; under Auction also the auction and repair())
    // The initial phase's matching and duals.
    std::vector<Int> init_col_to_row;
    std::vector<double> init_u, init_v;
    // The switch to the auction in phase 1 (Hybrid, or the probe's
    // probe_switch_to_auction): hybrid_step() runs it.
    bool hybrid_armed = false;           // the next advance() runs the auction first
    AuctionOptions hybrid_opt;           // validated, warm
    std::vector<int> hybrid_queue;       // the auction's first queue
    std::vector<int> hybrid_failed;      // phase-1 searches that failed before the switch
    std::vector<double> hybrid_reset_u;  // duals at the switch, for repair()'s reset
    // The auction ran (hybrid_step): at the end of phase 1,
    // canonical_duals() moves the sparse duals below hybrid_reset_u.
    bool canon_due = false;
    // Probe state (bench_matching_trial --hybrid-probe only). Unused unless
    // a probe_* hook is called.
    int n_sparse = 0;                 // sparse columns (n minus the dense ones)
    long long searches = 0;           // search() calls (probe_searches())
    int probe_pause_free = -1;        // probe_pause_at_free()
    // Search scratch.
    // Each holds at most n rows per search (a row is touched, finalized and
    // queued at the current distance at most once), so they are sized n
    // once and filled through counters.
    std::vector<int> touched, finalized, level;
    // Rows queued above the current distance, one entry per row, popped in
    // (distance, row) order: the order of the binary heap of (distance,
    // row) pairs with lazy duplicates it replaces (see IndexedMinHeap).
    IndexedMinHeap heap;

    bool dense(int i) const { return row_dense[static_cast<std::size_t>(i)] != 0; }
    bool search(int start_col);
    void list_free_sparse();
    void repair();
    void dense_duals();
    bool hybrid_step(long long work_budget);
    void canonical_duals();
    MatchingResult result(const std::vector<Int> &c2r, const std::vector<double> *u,
                          const std::vector<double> *vcol) const;
};

SymmetricMatcher::SymmetricMatcher(const CscLower &A, MatchingSearchOrder order, MatchingAlgorithm alg,
                                   const AuctionOptions *auction)
    : s_(std::make_unique<State>()) {
    State &s = *s_;
    s.order = order;
    s.alg = alg;
    if (alg == MatchingAlgorithm::Hybrid) {
        const AuctionOptions def = hybrid_auction_options();
        s.auction_opt = detail::validated(auction != nullptr ? *auction : def, def);
        // The auction starts from the searches' matching (0-CS).
        s.auction_opt.warm = true;
        s.switch_work = s.auction_opt.switch_passes * 2.0 * static_cast<double>(A.nnz());
    } else {
        s.auction_opt = detail::validated(auction != nullptr ? *auction : AuctionOptions{});
    }
    if (A.n < 0 || A.validate_shape() != ErrorCode::Success) {
        s.shape_ok = false;
        s.status = ErrorCode::IllegalValue;
        return;
    }
    const int n = static_cast<int>(A.n);
    s.n = n;
    s.col_to_row.assign(static_cast<std::size_t>(n), -1);
    if (n == 0) {
        s.phase = 4;
        return;
    }
    s.status = build_cost_graph(A, s.g);
    if (s.status != ErrorCode::Success) return;
    const CostGraph &g = s.g;
    auto &col_to_row = s.col_to_row;
    const Int *const gptr = g.ptr.data();
    const int *const grow = g.row.data();
    const double *const gcost = g.cost.data();

    s.rs.assign(static_cast<std::size_t>(n), RowState{kInf, kInf});
    s.heap.reset(n);
    s.touched.resize(static_cast<std::size_t>(n));
    s.finalized.resize(static_cast<std::size_t>(n));
    s.level.resize(static_cast<std::size_t>(n));
    s.row_match.assign(static_cast<std::size_t>(n), -1);
    s.row_parent.assign(static_cast<std::size_t>(n), -1);
    s.row_done.assign(static_cast<std::size_t>(n), 0);
    s.row_dense.assign(static_cast<std::size_t>(n), 0);
    RowState *const R = s.rs.data();
    int *const M = s.row_match.data();
    std::uint8_t *const D = s.row_dense.data();

    // Dense rows/columns (AMD's threshold, 10 sqrt(n) entries) are where
    // almost every column's cheapest entry lives in KKT systems with a few
    // dense constraints; matched greedily, they route every augmenting search
    // through an O(n) column. As MC64 keeps cheap assignments away from dense
    // columns, phase 1 solves the matching with them left out, phase 2 gives
    // them feasible duals, and phase 3 completes the matching on the full
    // graph. Duals stay feasible and matched edges tight throughout, so the
    // result is still the optimal matching.
    if (n > 50) {
        const double threshold = std::max(16.0, 10.0 * std::sqrt(static_cast<double>(n)));
        for (int j = 0; j < n; ++j) {
            if (static_cast<double>(gptr[j + 1] - gptr[j]) > threshold) {
                D[j] = 1;
                s.any_dense = true;
            }
        }
    }
    const auto is_dense = [D](int i) { return D[i] != 0; };
    for (int j = 0; j < n; ++j) s.n_sparse += D[j] ? 0 : 1;

    // Initial duals on the sparse part: u[i] = min_j cost(i, j);
    // v[j] = min_i cost(i, j) - u[i]. Rows or columns without sparse entries
    // start at 0 (phase 2 makes their edges feasible).
    std::vector<double> &v = s.v;
    v.assign(static_cast<std::size_t>(n), kInf);
    for (int j = 0; j < n; ++j) {
        if (is_dense(j)) continue;
        for (Int q = gptr[j]; q < gptr[j + 1]; ++q) {
            const int i = grow[q];
            if (is_dense(i)) continue;
            R[i].u = std::min(R[i].u, gcost[q]);
        }
    }
    for (int i = 0; i < n; ++i) {
        if (R[i].u == kInf) R[i].u = 0.0;
    }
    for (int j = 0; j < n; ++j) {
        double vj = kInf;
        if (!is_dense(j)) {
            for (Int q = gptr[j]; q < gptr[j + 1]; ++q) {
                const int i = grow[q];
                if (is_dense(i)) continue;
                vj = std::min(vj, gcost[q] - R[i].u);
            }
        }
        v[static_cast<std::size_t>(j)] = (vj == kInf) ? 0.0 : vj;
    }
    const auto reduced = [&](Int q, int j) {
        return gcost[q] - R[grow[q]].u - v[static_cast<std::size_t>(j)];
    };

    // Greedy: match each sparse column to a free sparse row along a tight edge.
    int matched = 0;
    for (int j = 0; j < n; ++j) {
        if (is_dense(j)) continue;
        for (Int q = gptr[j]; q < gptr[j + 1]; ++q) {
            const int i = grow[q];
            if (!is_dense(i) && M[i] < 0 && reduced(q, j) <= 0.0) {
                M[i] = j;
                col_to_row[static_cast<std::size_t>(j)] = i;
                ++matched;
                break;
            }
        }
    }
    // Length-2 augmentations on tight edges: free column j takes row i from
    // column jj, which moves to another free tight row. Rows never become
    // free again here, so each column's scan cursor only moves forward.
    std::vector<Int> cursor(g.ptr.begin(), g.ptr.end() - 1);
    for (int j = 0; j < n && matched < n; ++j) {
        if (is_dense(j) || col_to_row[static_cast<std::size_t>(j)] >= 0) continue;
        for (Int q = gptr[j]; q < gptr[j + 1]; ++q) {
            const int i = grow[q];
            if (is_dense(i) || reduced(q, j) > 0.0) continue;
            const int jj = M[i];
            if (jj < 0) continue;
            Int &c = cursor[static_cast<std::size_t>(jj)];
            int free_row = -1;
            for (; c < gptr[jj + 1]; ++c) {
                const int k = grow[c];
                if (!is_dense(k) && M[k] < 0 && reduced(c, jj) <= 0.0) {
                    free_row = k;
                    ++c;
                    break;
                }
            }
            if (free_row < 0) continue;
            M[free_row] = jj;
            col_to_row[static_cast<std::size_t>(jj)] = free_row;
            M[i] = j;
            col_to_row[static_cast<std::size_t>(j)] = i;
            ++matched;
            break;
        }
    }
    s.matched = matched;

    s.init_col_to_row = col_to_row;
    s.init_u.resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) s.init_u[static_cast<std::size_t>(i)] = R[i].u;
    s.init_v = v;

    // Phase 1 hides the dense rows from its searches by a distance of -inf,
    // which no candidate distance beats (dense_duals() restores inf).
    if (s.any_dense) {
        for (int i = 0; i < n; ++i) {
            if (is_dense(i)) R[i].dist = -kInf;
        }
    }
}

SymmetricMatcher::~SymmetricMatcher() = default;

ErrorCode SymmetricMatcher::status() const { return s_->status; }

void SymmetricMatcher::set_stop_free(int max_free) {
    // The auction matches the sparse part as a whole: no partial stop.
    assert(s_->alg == MatchingAlgorithm::ShortestPath);
    if (s_->alg == MatchingAlgorithm::ShortestPath) s_->stop_free = max_free;
}

long long SymmetricMatcher::work() const { return s_->work; }

int SymmetricMatcher::repaired() const { return s_->repaired; }

bool SymmetricMatcher::switched() const { return s_->switched; }

bool SymmetricMatcher::diagonal() const {
    const State &s = *s_;
    if (s.status != ErrorCode::Success || s.init_col_to_row.empty()) return false;
    for (int j = 0; j < s.n; ++j) {
        if (!s.dense(j) && s.init_col_to_row[static_cast<std::size_t>(j)] != j) return false;
    }
    return true;
}

MatchingResult SymmetricMatcher::diagonal_result() {
    assert(diagonal());
    State &s = *s_;
    // Phase 1 has no free sparse column to search: give the dense columns
    // their duals (phase 2) and stop short of phase 3.
    if (s.status == ErrorCode::Success && s.phase == 1) {
        if (s.any_dense) s.dense_duals();
        s.phase = 3;
        s.next_col = 0;
    }
    return current();
}

// Shortest augmenting path from a free column (Dijkstra on reduced costs); on
// success the duals are updated and the matching augmented. In phase 1 the
// dense rows are invisible (their distance is -inf, see the constructor).
bool SymmetricMatcher::State::search(int start_col) {
    ++searches;  // probe counter only; affects nothing else
    RowState *const R = rs.data();
    int *const M = row_match.data();
    int *const P = row_parent.data();
    std::uint8_t *const DN = row_done.data();
    const Int *const gptr = g.ptr.data();
    const int *const grow = g.row.data();
    const double *const gcost = g.cost.data();
    int *const T = touched.data();
    int *const F = finalized.data();
    int *const L = level.data();
    // The counters index fixed-size lists of n entries (see State): a row
    // is touched, finalized and queued at the current distance at most once
    // per search, which the debug asserts below check.
    std::size_t n_touched = 0, n_final = 0, n_level = 0;
    heap.clear();
    std::size_t level_head = 0;

    // Free rows are never expanded: the closest one seen so far ends the
    // search as soon as no matched row can beat it (MC64's early exit).
    // Rows reached over a zero reduced cost sit at the current distance,
    // the minimum, so they queue in `level` without heap traffic.
    //
    // A matched row reached at nd >= path_len is left untouched: path_len
    // never grows, so such a row could never be finalized (the loop below
    // stops at the first distance >= path_len), and only finalized rows
    // feed the dual update and the augmenting path. A later, shorter
    // distance below path_len still beats its (untouched) inf, exactly as
    // it would have beaten the skipped nd, so the search's every decision
    // is unchanged. This drops ~90% of the heap pushes on KKT systems
    // whose searches find a free row early.
    //
    // Finalized rows need no test either: a row is finalized at a distance
    // <= base (base never decreases), and every candidate distance is
    // base + a non-negative reduced cost, so it never beats theirs.
    int free_row = -1;
    double path_len = kInf;
    const auto relax = [&](int j, double base) {
        const double vj = v[static_cast<std::size_t>(j)];
        const Int qend = gptr[j + 1];
        work += qend - gptr[j];
        for (Int q = gptr[j]; q < qend; ++q) {
            const int i = grow[q];
            RowState &ri = R[i];
            const double nd = base + std::max(0.0, gcost[q] - ri.u - vj);
            if (nd < ri.dist) {
                if (M[i] < 0) {
                    if (ri.dist == kInf) {
                        assert(n_touched < static_cast<std::size_t>(n));
                        T[n_touched++] = i;
                    }
                    ri.dist = nd;
                    P[i] = j;
                    if (nd < path_len) {
                        free_row = i;
                        path_len = nd;
                    }
                    // Nothing beats a free row at the current distance;
                    // stop scanning (dense columns would cost O(n)).
                    if (nd == base) return;
                } else if (nd < path_len) {
                    if (ri.dist == kInf) {
                        assert(n_touched < static_cast<std::size_t>(n));
                        T[n_touched++] = i;
                    }
                    ri.dist = nd;
                    P[i] = j;
                    if (nd == base) {
                        assert(n_level < static_cast<std::size_t>(n));
                        L[n_level++] = i;
                    } else {
                        heap.push_or_decrease(i, nd);
                    }
                }
            }
        }
    };

    relax(start_col, 0.0);
    double base = 0.0;
    while (true) {
        int i;
        if (level_head < n_level) {
            if (base >= path_len) break;
            i = L[level_head++];
            if (DN[i]) continue;
        } else {
            n_level = 0;
            level_head = 0;
            if (heap.empty() || heap.top_key() >= path_len) break;
            const double d = heap.top_key();
            const int hi = heap.top_id();
            heap.pop();
            // A row moved to `level` keeps its older, larger entry.
            if (DN[hi] || d > R[hi].dist) continue;
            i = hi;
            base = d;
        }
        DN[i] = 1;
        assert(n_final < static_cast<std::size_t>(n));
        F[n_final++] = i;
        relax(M[i], base);
    }

    if (free_row >= 0) {
        // Dual update over the finalized rows (dist <= path_len) and
        // their matched columns keeps reduced costs >= 0 and makes the
        // path tight.
        v[static_cast<std::size_t>(start_col)] += path_len;
        for (std::size_t k = 0; k < n_final; ++k) {
            const int i = F[k];
            const double delta = path_len - R[i].dist;
            R[i].u -= delta;
            const int mc = M[i];
            if (mc >= 0) {
                v[static_cast<std::size_t>(mc)] += delta;
            }
        }

        // Augment along the path from free_row back to start_col.
        int r = free_row;
        while (r >= 0) {
            const int c = P[r];
            const int prev_r = (c == start_col) ? -1 : static_cast<int>(col_to_row[static_cast<std::size_t>(c)]);
            M[r] = c;
            col_to_row[static_cast<std::size_t>(c)] = r;
            r = prev_r;
        }
        ++matched;
    }

    for (std::size_t k = 0; k < n_touched; ++k) {
        const int i = T[k];
        R[i].dist = kInf;
        DN[i] = 0;
    }
    return free_row >= 0;
}

// Lists phase 1's free sparse columns in search_order: column order, or by
// decreasing degree (stable).
void SymmetricMatcher::State::list_free_sparse() {
    search_order.clear();
    for (int j = 0; j < n; ++j) {
        if (!dense(j) && col_to_row[static_cast<std::size_t>(j)] < 0) search_order.push_back(j);
    }
    const Int *const gptr = g.ptr.data();
    if (order == MatchingSearchOrder::Degree)
        std::stable_sort(search_order.begin(), search_order.end(),
                         [gptr](int a, int b) { return gptr[a + 1] - gptr[a] > gptr[b + 1] - gptr[b]; });
}

// After the auction: v_j = min over sparse rows of (c_ij - u_i) makes the
// duals exactly feasible on the sparse part (the auction keeps them so up
// to its v), every matched edge left loose (c - u - v > 0) is unmatched,
// and the free sparse columns are listed for phase 1's searches. The
// searches need nothing more: feasible duals and tight matched edges, for
// any u.
//
// So first, if the auction left columns free (capped, or a column without
// sparse rows), every sparse row reachable from them by alternating paths
// (any edge to a row, its matched edge back to a column) gets its
// constructor dual back. A sparse part without a perfect matching makes
// the auction's prices fall without bound on exactly those rows (a price
// war: ~1e5 log units at the bid cap), and the searches would carry them
// into the scaling: 6 of the 12 deficient KKTs of
// AuctionMatching.PriceWarsDoNotReachTheScaling ended with scales clamped
// at exp(+-340) without the reset. Rows outside keep the auction's duals.
void SymmetricMatcher::State::repair() {
    RowState *const R = rs.data();
    const Int *const gptr = g.ptr.data();
    const int *const grow = g.row.data();
    const double *const gcost = g.cost.data();
    {
        // Columns to expand in `level`, reached rows marked in row_done and
        // listed in `touched` (each holds at most n: a column is pushed
        // free or through its matched row, once), the marks cleared after.
        int *const stack = level.data();
        int *const reached = touched.data();
        std::uint8_t *const seen = row_done.data();
        std::size_t n_stack = 0, n_reached = 0;
        // The reset target: the constructor's duals, or after a switch to
        // the auction (hybrid_step) the exact duals at the switch.
        const std::vector<double> &reset_u = hybrid_reset_u.empty() ? init_u : hybrid_reset_u;
        for (int j = 0; j < n; ++j) {
            if (!dense(j) && col_to_row[static_cast<std::size_t>(j)] < 0) stack[n_stack++] = j;
        }
        while (n_stack > 0) {
            const int j = stack[--n_stack];
            work += gptr[j + 1] - gptr[j];
            for (Int q = gptr[j]; q < gptr[j + 1]; ++q) {
                const int i = grow[q];
                if (dense(i) || seen[i]) continue;
                seen[i] = 1;
                reached[n_reached++] = i;
                R[i].u = reset_u[static_cast<std::size_t>(i)];
                const int k = row_match[static_cast<std::size_t>(i)];
                if (k >= 0) stack[n_stack++] = k;
            }
        }
        for (std::size_t t = 0; t < n_reached; ++t) seen[reached[t]] = 0;
    }
    matched = 0;
    for (int j = 0; j < n; ++j) {
        if (dense(j)) continue;
        work += gptr[j + 1] - gptr[j];
        const int r = static_cast<int>(col_to_row[static_cast<std::size_t>(j)]);
        double vj = kInf, dr = kInf;
        for (Int q = gptr[j]; q < gptr[j + 1]; ++q) {
            const int i = grow[q];
            if (dense(i)) continue;
            const double d = gcost[q] - R[i].u;
            vj = std::min(vj, d);
            if (i == r) dr = d;
        }
        // No sparse row: 0, as the constructor gives it (phase 2 makes the
        // column's dense edges feasible).
        v[static_cast<std::size_t>(j)] = vj == kInf ? 0.0 : vj;
        if (r < 0) continue;
        // The searches' reduced cost, (c - u) - v, of the matched edge.
        if (dr - v[static_cast<std::size_t>(j)] > 0.0) {
            row_match[static_cast<std::size_t>(r)] = -1;
            col_to_row[static_cast<std::size_t>(j)] = -1;
        } else {
            ++matched;
        }
    }
    list_free_sparse();
    next_col = 0;
    repaired = static_cast<int>(search_order.size());
}

// Phase 2: dense rows take the largest u, then dense columns the largest v,
// that keep every reduced cost >= 0.
void SymmetricMatcher::State::dense_duals() {
    RowState *const R = rs.data();
    const Int *const gptr = g.ptr.data();
    const int *const grow = g.row.data();
    const double *const gcost = g.cost.data();
    for (int j = 0; j < n; ++j) {
        if (dense(j)) {
            R[j].u = kInf;
            R[j].dist = kInf;  // visible to the phase-3 searches
        }
    }
    for (int j = 0; j < n; ++j) {
        if (dense(j)) continue;
        for (Int q = gptr[j]; q < gptr[j + 1]; ++q) {
            const int i = grow[q];
            if (!dense(i)) continue;
            R[i].u = std::min(R[i].u, gcost[q] - v[static_cast<std::size_t>(j)]);
        }
    }
    for (int j = 0; j < n; ++j) {
        if (dense(j) && R[j].u == kInf) R[j].u = 0.0;
    }
    for (int j = 0; j < n; ++j) {
        if (!dense(j)) continue;
        double vj = kInf;
        for (Int q = gptr[j]; q < gptr[j + 1]; ++q) {
            vj = std::min(vj, gcost[q] - R[grow[q]].u);
        }
        v[static_cast<std::size_t>(j)] = vj;
    }
}

// The switch to the auction in phase 1 (Hybrid, or probe_switch_to_auction):
// the auction and repair(); false: paused by the budget inside the auction
// (the next call resumes it; the snapshot below is taken once, before the
// auction starts). The searched part of search_order (before next_col)
// holds matched columns and the failed searches' columns; the rest are the
// free, unsearched ones the auction bids for. Phase 1's
// duals are feasible and its matched edges tight (0-CS), as the warm
// auction requires; dense rows stay invisible to both.
bool SymmetricMatcher::State::hybrid_step(long long work_budget) {
    if (auction.phase == -1) {
        hybrid_queue.assign(search_order.begin() + next_col, search_order.end());
        hybrid_failed.clear();
        for (int k = 0; k < next_col; ++k) {
            const int j = search_order[static_cast<std::size_t>(k)];
            if (col_to_row[static_cast<std::size_t>(j)] < 0) hybrid_failed.push_back(j);
        }
        hybrid_reset_u.resize(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) hybrid_reset_u[static_cast<std::size_t>(i)] = rs[static_cast<std::size_t>(i)].u;
    }
    if (hybrid_queue.empty()) {  // nothing left to bid for: plain run
        hybrid_armed = false;
        return true;
    }
    const detail::AuctionStatus st = detail::auction_advance(g, row_dense.data(), rs, v, row_match, col_to_row,
                                                             hybrid_queue, hybrid_opt, auction, work, work_budget);
    if (st == detail::AuctionStatus::Paused) return false;
    repair();  // relists search_order (every free sparse column), next_col = 0
    canon_due = true;
    if (!hybrid_failed.empty()) {
        // Their phase-1 search already failed: phase 3 takes them, as in
        // the plain run (row_done is clear between searches).
        for (int j : hybrid_failed) row_done[static_cast<std::size_t>(j)] = 1;
        std::erase_if(search_order, [&](int j) { return row_done[static_cast<std::size_t>(j)] != 0; });
        for (int j : hybrid_failed) row_done[static_cast<std::size_t>(j)] = 0;
        repaired = static_cast<int>(search_order.size());
    }
    hybrid_armed = false;
    return true;
}

bool SymmetricMatcher::advance(long long work_budget) {
    State &s = *s_;
    if (s.status != ErrorCode::Success || s.phase == 4) return true;
    const int n = s.n;
    const auto out_of_budget = [&] { return work_budget >= 0 && s.work >= work_budget; };
    // Phase 1: the sparse part (a failed search just leaves its column for
    // phase 3).
    if (s.phase == 1) {
        if (s.alg == MatchingAlgorithm::Auction) {
            if (!s.repair_done) {
                // The auction's first queue: the free sparse columns in the
                // search order (none free: nothing to bid for or repair).
                if (s.auction.phase == -1) s.list_free_sparse();
                if (!s.search_order.empty()) {
                    const detail::AuctionStatus st =
                        detail::auction_advance(s.g, s.row_dense.data(), s.rs, s.v, s.row_match, s.col_to_row,
                                                s.search_order, s.auction_opt, s.auction, s.work, work_budget);
                    if (st == detail::AuctionStatus::Paused) return false;
                    s.repair();
                }
                s.repair_done = true;
            }
        } else if (s.next_col == 0 && s.search_order.empty()) {
            s.list_free_sparse();
        }
        // Again after a switch: the auction and repair() relist
        // search_order, and the searches start over on it.
        for (bool relisted = true; relisted;) {
            relisted = false;
            if (s.hybrid_armed && !s.hybrid_step(work_budget)) return false;
            const int n_free = static_cast<int>(s.search_order.size());
            for (; s.next_col < n_free && s.matched < n; ++s.next_col) {
                const int j = s.search_order[static_cast<std::size_t>(s.next_col)];
                if (out_of_budget()) return false;
                // Hybrid: the switch depends on the search work alone (not
                // on the budget), so every budgeting reaches it at the same
                // search boundary, with j and the columns after it free.
                if (s.alg == MatchingAlgorithm::Hybrid && !s.switched &&
                    static_cast<double>(s.work) > s.switch_work) {
                    s.switched = true;
                    s.hybrid_opt = s.auction_opt;
                    s.hybrid_armed = true;
                    relisted = true;
                    break;
                }
                // Probe (probe_pause_at_free): pause for the caller's switch.
                if (s.probe_pause_free >= 0 && s.n_sparse - s.matched <= s.probe_pause_free) return false;
                if (s.stop_free >= 0 && n - s.matched <= s.stop_free) {
                    // Partial (set_stop_free): the dense columns' duals made
                    // feasible, as before their searches; the rest stay free.
                    if (s.any_dense) s.dense_duals();
                    s.phase = 4;
                    return true;
                }
                s.search(j);
            }
        }
        // After a switch, before phase 2 derives the dense rows' duals from
        // the sparse columns' (canonical_duals()). Never paused (no budget
        // check), so every budgeting runs it whole on the same state.
        if (s.canon_due) s.canonical_duals();
        if (s.any_dense) s.dense_duals();
        s.phase = 3;
        s.next_col = 0;
    }
    // Phase 3: every column still free, on the full graph.
    for (; s.next_col < n && s.matched < n; ++s.next_col) {
        const int j = s.next_col;
        if (s.col_to_row[static_cast<std::size_t>(j)] < 0) {
            if (out_of_budget()) return false;
            if (!s.search(j)) {
                s.status = ErrorCode::SingularMatrix;
                s.phase = 4;
                return true;
            }
        }
    }
    s.phase = 4;
    return true;
}

// Canonical duals after a switch to the auction (ncvxqp1-probe-report.md,
// candidate d), at the end of phase 1. Every optimal dual scales the matrix
// validly, but the auction's bids lower u along their displacement chains,
// and repair() keeps those prices on every row no free column reaches: a
// different point of the optimal dual face than the searches end at. On
// ncvxqp1 rec (18 columns left at the switch, 68% handed back) the scale
// drifted by up to e^8.5 from ShortestPath's, delays went 24 -> 40 and omega
// 8.8e-12 -> 4.6e-8 (gate 1e-8).
//
// So the sparse rows' duals move to the pointwise largest u <= cap (the
// duals at the switch, hybrid_reset_u) that keeps every reduced cost >= 0
// and every matched edge tight: delta_i = u_new - u is the shortest
// distance to row i in the rows' graph k -> i (i on the column matched to
// row k, weight the reduced cost c_ij - u_i - v_j >= 0), every row starting
// at max(0, cap_i - u_i); then u_i += delta_i, and the column matched to
// row i takes v_j -= delta_i (delta_i <= delta_k + weight keeps each
// reduced cost >= 0, up to the rounding of the searches' own dual updates).
// The matching stays a maximum-product one of the sparse part. Up to the
// switch Hybrid is the Degree ShortestPath matcher bitwise, and the
// ShortestPath searches from the switch's duals end at this u (it does not
// depend on which optimal matching was found): the probe's scales equal
// ShortestPath's to dlog 4.4e-14 on the 4 switched corpus matrices
// (ncvxqp1 rec omega 5.7e-12), and AuctionMatching.
// HybridCanonicalDualsMatchShortestPathScale checks it on KKTs.
//
// Here and not after phase 3 (the probe's place): phase 2 derives the
// dense rows' duals from the sparse columns' v, so canonical sparse duals
// give it ShortestPath's input; after phase 3 the dense rows' duals would
// already differ (dense_row_kkt: dlog up to 1.6 with the probe's cap of
// today's u on dense rows). Dense rows are hidden (phase 1 never touches
// them). Free sparse columns (failed searches, left to phase 3) constrain
// no row and keep the constructor's v, as under ShortestPath (a column
// never matched keeps it; it stays feasible since u only fell since the
// constructor), so phase 2 sees ShortestPath's v there too.
//
// Cost: one pass over the matched columns plus n heap operations (+0.8-1.0
// passes on the probe's matrices), counted in work().
void SymmetricMatcher::State::canonical_duals() {
    canon_due = false;
    RowState *const R = rs.data();
    const Int *const gptr = g.ptr.data();
    const int *const grow = g.row.data();
    const double *const gcost = g.cost.data();
    std::vector<double> delta(static_cast<std::size_t>(n), 0.0);
    std::vector<std::uint8_t> done(static_cast<std::size_t>(n), 0);
    heap.clear();
    for (int i = 0; i < n; ++i) {
        if (dense(i)) {
            done[static_cast<std::size_t>(i)] = 1;
            continue;
        }
        delta[static_cast<std::size_t>(i)] = std::max(0.0, hybrid_reset_u[static_cast<std::size_t>(i)] - R[i].u);
        heap.push_or_decrease(i, delta[static_cast<std::size_t>(i)]);
    }
    while (!heap.empty()) {
        const int k = heap.top_id();
        const double dk = heap.top_key();
        heap.pop();
        done[static_cast<std::size_t>(k)] = 1;
        const int j = row_match[static_cast<std::size_t>(k)];
        if (j < 0) continue;
        const double vj = v[static_cast<std::size_t>(j)];
        work += gptr[j + 1] - gptr[j];
        for (Int q = gptr[j]; q < gptr[j + 1]; ++q) {
            const int i = grow[q];
            if (done[static_cast<std::size_t>(i)]) continue;
            const double nd = dk + std::max(0.0, gcost[q] - R[i].u - vj);
            if (nd < delta[static_cast<std::size_t>(i)]) {
                delta[static_cast<std::size_t>(i)] = nd;
                heap.push_or_decrease(i, nd);
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        const double d = delta[static_cast<std::size_t>(i)];
        R[i].u += d;
        const int j = row_match[static_cast<std::size_t>(i)];
        if (j >= 0) v[static_cast<std::size_t>(j)] -= d;
    }
    for (int j = 0; j < n; ++j) {
        if (dense(j) || col_to_row[static_cast<std::size_t>(j)] >= 0) continue;
        work += gptr[j + 1] - gptr[j];
        double vj = init_v[static_cast<std::size_t>(j)];
        for (Int q = gptr[j]; q < gptr[j + 1]; ++q) {
            const int i = grow[q];
            if (!dense(i)) vj = std::min(vj, gcost[q] - R[i].u);  // rounding guard
        }
        v[static_cast<std::size_t>(j)] = vj;
    }
}

// The matching given by c2r, scaled from the duals u (rows) and vcol
// (columns), or unscaled (unit scale) without them: v[j] is re-derived on
// the matched (tight) edge, then
// s_row[i] = exp(u[i]), s_col[i] = exp(v[i]) / col_max[i] and
// scale[i] = sqrt(s_row[i] * s_col[i]) = exp(0.5 * (u[i] + v[i])) / sqrt(col_max[i]).
MatchingResult SymmetricMatcher::State::result(const std::vector<Int> &c2r, const std::vector<double> *u_rows,
                                               const std::vector<double> *vcol) const {
    MatchingResult res;
    if (!shape_ok) {
        res.status = status;
        return res;
    }
    res.scale.assign(static_cast<std::size_t>(n), 1.0);
    res.row_to_col.assign(static_cast<std::size_t>(n), -1);
    res.col_to_row = c2r;
    if (n == 0) return res;
    for (int j = 0; j < n; ++j) {
        const Int i = c2r[static_cast<std::size_t>(j)];
        if (i >= 0) {
            res.row_to_col[static_cast<std::size_t>(i)] = j;
            ++res.matched_rank;
        }
    }
    if (u_rows == nullptr || vcol == nullptr) return res;
    const std::vector<double> &u = *u_rows;
    std::vector<double> vt = *vcol;
    for (int j = 0; j < n; ++j) {
        const int r = static_cast<int>(c2r[static_cast<std::size_t>(j)]);
        for (Int q = g.ptr[static_cast<std::size_t>(j)]; q < g.ptr[static_cast<std::size_t>(j) + 1]; ++q) {
            if (g.row[static_cast<std::size_t>(q)] == r) {
                vt[static_cast<std::size_t>(j)] = g.cost[static_cast<std::size_t>(q)] - u[static_cast<std::size_t>(r)];
                break;
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        const double log_s = 0.5 * (u[static_cast<std::size_t>(i)] + vt[static_cast<std::size_t>(i)] -
                                    std::log(g.col_max[static_cast<std::size_t>(i)]));
        const double clamped_log = std::clamp(log_s, -340.0, 340.0);
        res.scale[static_cast<std::size_t>(i)] = std::exp(clamped_log);
    }
    return res;
}

// ---- Probe hooks (bench_matching_trial --hybrid-probe only). ----

long long SymmetricMatcher::probe_searches() const { return s_->searches; }

int SymmetricMatcher::probe_free_sparse() const {
    const State &s = *s_;
    if (s.status != ErrorCode::Success || s.phase != 1 || s.alg != MatchingAlgorithm::ShortestPath) return -1;
    return s.n_sparse - s.matched;
}

void SymmetricMatcher::probe_pause_at_free(int max_free) { s_->probe_pause_free = max_free; }

void SymmetricMatcher::probe_switch_to_auction(const AuctionOptions &opt) {
    State &s = *s_;
    s.probe_pause_free = -1;
    if (s.status != ErrorCode::Success || s.phase != 1 || s.alg != MatchingAlgorithm::ShortestPath) return;
    s.hybrid_opt = detail::validated(opt);
    s.hybrid_opt.warm = true;
    s.hybrid_armed = true;
}

MatchingResult SymmetricMatcher::current() const {
    const State &s = *s_;
    if (s.status != ErrorCode::Success) {
        // A failed matching keeps its matched part and the unit scaling.
        MatchingResult res = s.result(s.col_to_row, nullptr, nullptr);
        res.status = s.status;
        return res;
    }
    std::vector<double> u(static_cast<std::size_t>(s.n));
    for (int i = 0; i < s.n; ++i) u[static_cast<std::size_t>(i)] = s.rs[static_cast<std::size_t>(i)].u;
    return s.result(s.col_to_row, &u, &s.v);
}

MatchingResult SymmetricMatcher::initial() const {
    const State &s = *s_;
    if (s.init_col_to_row.empty() && s.n > 0) return current();  // construction failed
    MatchingResult res = s.result(s.init_col_to_row, &s.init_u, &s.init_v);
    res.status = s.shape_ok ? ErrorCode::Success : s.status;
    return res;
}

MatchingResult compute_symmetric_matching(const CscLower &A) {
    SymmetricMatcher m(A);
    m.advance(-1);
    return m.current();
}

ErrorCode compute_matching_scale(const CscLower &A,
                                 std::vector<double> &scale,
                                 int /*print_level*/) {
    MatchingResult mr = compute_symmetric_matching(A);
    if (mr.status != ErrorCode::Success) {
        return mr.status;
    }
    scale = std::move(mr.scale);
    return ErrorCode::Success;
}

} // namespace smf
