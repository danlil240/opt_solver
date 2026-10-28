#include "smf/scaling_matching.hpp"

#include "indexed_min_heap.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace smf {

namespace {

// Full symmetric pattern (both triangles) in CSC form, duplicates merged and
// explicit zeros dropped, with the matching cost of every entry:
//   cost(i, j) = log(col_max[j]) - log|a_ij| >= 0.
struct CostGraph {
    std::vector<Int> ptr;
    std::vector<int> row;
    std::vector<double> cost;
    std::vector<double> col_max;
};

ErrorCode build_cost_graph(const CscLower &A, CostGraph &g) {
    const int n = static_cast<int>(A.n);
    std::vector<Int> count(static_cast<std::size_t>(n) + 1, 0);
    // Cleaned input (each column's rows strictly increasing, none above the
    // diagonal, as analyse() passes it) expands without duplicates: column
    // j gets rows jj < j from the earlier columns, then its own rows >= j.
    bool cleaned = true;
    for (int j = 0; j < n; ++j) {
        const Int p0 = A.col_ptr[static_cast<std::size_t>(j)];
        for (Int p = p0; p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
            const int i = static_cast<int>(A.row_idx[static_cast<std::size_t>(p)]);
            if (!std::isfinite(A.values[static_cast<std::size_t>(p)])) {
                return ErrorCode::IllegalValue;
            }
            if (i < j || (p > p0 && i <= A.row_idx[static_cast<std::size_t>(p) - 1])) cleaned = false;
            ++count[static_cast<std::size_t>(j) + 1];
            if (i != j) ++count[static_cast<std::size_t>(i) + 1];
        }
    }
    for (int j = 0; j < n; ++j) {
        count[static_cast<std::size_t>(j) + 1] += count[static_cast<std::size_t>(j)];
    }
    const std::size_t nnz = static_cast<std::size_t>(count[static_cast<std::size_t>(n)]);
    g.ptr.assign(static_cast<std::size_t>(n) + 1, 0);
    g.row.resize(nnz);
    g.cost.resize(nnz);
    g.col_max.assign(static_cast<std::size_t>(n), 0.0);
    // Without duplicates the expanded entries go straight into g, where the
    // merge below would have copied each one unchanged.
    std::vector<int> raw_row_buf;
    std::vector<double> raw_val_buf;
    if (!cleaned) {
        raw_row_buf.resize(nnz);
        raw_val_buf.resize(nnz);
    }
    int *const raw_row = cleaned ? g.row.data() : raw_row_buf.data();
    double *const raw_val = cleaned ? g.cost.data() : raw_val_buf.data();
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
    // Cleaned input skips the merge: column j already sits at
    // [count[j], count[j+1]) of g, and is compacted from there to `out`
    // (never past the read position).
    std::vector<int> seen_in(cleaned ? 0 : static_cast<std::size_t>(n), -1);
    std::vector<Int> slot_of(cleaned ? 0 : static_cast<std::size_t>(n), -1);
    Int out = 0;
    for (int j = 0; j < n; ++j) {
        const Int begin = out;
        Int read = begin;  // the column's merged entries: [read, out)
        if (cleaned) {
            read = count[static_cast<std::size_t>(j)];
            out = count[static_cast<std::size_t>(j) + 1];
        } else {
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
        }
        // g.cost holds the merged value for now; compact out zeros.
        Int w = begin;
        double cmax = 0.0;
        for (Int q = read; q < out; ++q) {
            const double av = std::abs(g.cost[static_cast<std::size_t>(q)]);
            if (av > 0.0) {
                g.row[static_cast<std::size_t>(w)] = g.row[static_cast<std::size_t>(q)];
                g.cost[static_cast<std::size_t>(w)] = av;
                cmax = std::max(cmax, av);
                ++w;
            }
        }
        out = w;
        if (cmax <= 0.0) {
            return ErrorCode::SingularMatrix;
        }
        g.col_max[static_cast<std::size_t>(j)] = cmax;
        const double log_cmax = std::log(cmax);
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

// The per-row state every scanned edge reads: the row dual and the search
// distance, 16 bytes, so four rows share a cache line (the searches are
// bound by these random row accesses). The row's match, parent and flags,
// needed only once a row's distance improves or it is finalized, live in
// separate arrays of the matcher.
struct RowState {
    double u;      // row dual
    double dist;   // tentative search distance (inf when untouched)
};

constexpr double kInf = std::numeric_limits<double>::infinity();

} // namespace

// Maximum-product perfect matching (MC64 job 5): minimum-cost assignment on
// the costs above, solved by shortest augmenting paths with dual potentials
// u[row], v[col] (reduced cost c - u - v >= 0, zero on matched edges).
// As in MC64, a greedy matching on zero-reduced-cost edges (plus length-2
// augmentations) settles most columns, and each remaining column runs a
// heap-based Dijkstra that touches, and afterwards resets, only the rows it
// reaches.
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
    // Searches left: phase 1 (sparse part) or 3 (full graph) from column
    // next_col on; 4 once none is left.
    int phase = 1;
    int next_col = 0;
    long long work = 0;  // edges scanned by the searches
    // The initial phase's matching and duals.
    std::vector<Int> init_col_to_row;
    std::vector<double> init_u, init_v;
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
    void dense_duals();
    MatchingResult result(const std::vector<Int> &c2r, const std::vector<double> *u,
                          const std::vector<double> *vcol) const;
};

SymmetricMatcher::SymmetricMatcher(const CscLower &A) : s_(std::make_unique<State>()) {
    State &s = *s_;
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

long long SymmetricMatcher::work() const { return s_->work; }

// Shortest augmenting path from a free column (Dijkstra on reduced costs); on
// success the duals are updated and the matching augmented. In phase 1 the
// dense rows are invisible (their distance is -inf, see the constructor).
bool SymmetricMatcher::State::search(int start_col) {
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
                    if (ri.dist == kInf) T[n_touched++] = i;
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
                    if (ri.dist == kInf) T[n_touched++] = i;
                    ri.dist = nd;
                    P[i] = j;
                    if (nd == base) {
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
            i = level[level_head++];
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

bool SymmetricMatcher::advance(long long work_budget) {
    State &s = *s_;
    if (s.status != ErrorCode::Success || s.phase == 4) return true;
    const int n = s.n;
    const auto out_of_budget = [&] { return work_budget >= 0 && s.work >= work_budget; };
    // Phase 1: the sparse part (a failed search just leaves its column for
    // phase 3).
    if (s.phase == 1) {
        for (; s.next_col < n && s.matched < n; ++s.next_col) {
            const int j = s.next_col;
            if (!s.dense(j) && s.col_to_row[static_cast<std::size_t>(j)] < 0) {
                if (out_of_budget()) return false;
                s.search(j);
            }
        }
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
