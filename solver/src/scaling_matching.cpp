#include "smf/scaling_matching.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
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
    for (int j = 0; j < n; ++j) {
        for (Int p = A.col_ptr[static_cast<std::size_t>(j)];
             p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
            const int i = static_cast<int>(A.row_idx[static_cast<std::size_t>(p)]);
            if (!std::isfinite(A.values[static_cast<std::size_t>(p)])) {
                return ErrorCode::IllegalValue;
            }
            ++count[static_cast<std::size_t>(j) + 1];
            if (i != j) ++count[static_cast<std::size_t>(i) + 1];
        }
    }
    for (int j = 0; j < n; ++j) {
        count[static_cast<std::size_t>(j) + 1] += count[static_cast<std::size_t>(j)];
    }
    const std::size_t nnz = static_cast<std::size_t>(count[static_cast<std::size_t>(n)]);
    std::vector<int> raw_row(nnz);
    std::vector<double> raw_val(nnz);
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
    // does) with a last-seen marker, then drop zeros.
    g.ptr.assign(static_cast<std::size_t>(n) + 1, 0);
    g.row.resize(nnz);
    g.cost.resize(nnz);
    g.col_max.assign(static_cast<std::size_t>(n), 0.0);
    std::vector<Int> slot_of(static_cast<std::size_t>(n), -1);
    Int out = 0;
    for (int j = 0; j < n; ++j) {
        const Int begin = out;
        for (Int q = count[static_cast<std::size_t>(j)]; q < count[static_cast<std::size_t>(j) + 1]; ++q) {
            const int i = raw_row[static_cast<std::size_t>(q)];
            const Int s = slot_of[static_cast<std::size_t>(i)];
            if (s >= begin) {
                g.cost[static_cast<std::size_t>(s)] += raw_val[static_cast<std::size_t>(q)];
            } else {
                slot_of[static_cast<std::size_t>(i)] = out;
                g.row[static_cast<std::size_t>(out)] = i;
                g.cost[static_cast<std::size_t>(out)] = raw_val[static_cast<std::size_t>(q)];
                ++out;
            }
        }
        // g.cost holds the merged value for now; compact out zeros.
        Int w = begin;
        double cmax = 0.0;
        for (Int q = begin; q < out; ++q) {
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

} // namespace

// Maximum-product perfect matching (MC64 job 5): minimum-cost assignment on
// the costs above, solved by shortest augmenting paths with dual potentials
// u[row], v[col] (reduced cost c - u - v >= 0, zero on matched edges).
// As in MC64, a greedy matching on zero-reduced-cost edges (plus length-2
// augmentations) settles most columns, and each remaining column runs a
// heap-based Dijkstra that touches, and afterwards resets, only the rows it
// reaches.
MatchingResult compute_symmetric_matching(const CscLower &A) {
    MatchingResult res;
    if (A.n < 0 || A.validate_shape() != ErrorCode::Success) {
        res.status = ErrorCode::IllegalValue;
        return res;
    }
    const int n = static_cast<int>(A.n);
    res.scale.assign(static_cast<std::size_t>(n), 1.0);
    res.row_to_col.assign(static_cast<std::size_t>(n), -1);
    res.col_to_row.assign(static_cast<std::size_t>(n), -1);
    if (n == 0) {
        res.status = ErrorCode::Success;
        return res;
    }

    CostGraph g;
    const ErrorCode gerr = build_cost_graph(A, g);
    if (gerr != ErrorCode::Success) {
        res.status = gerr;
        return res;
    }
    auto &row_to_col = res.row_to_col;
    auto &col_to_row = res.col_to_row;
    const auto col_begin = [&](int j) { return g.ptr[static_cast<std::size_t>(j)]; };
    const auto col_end = [&](int j) { return g.ptr[static_cast<std::size_t>(j) + 1]; };

    // Dense rows/columns (AMD's threshold, 10 sqrt(n) entries) are where
    // almost every column's cheapest entry lives in KKT systems with a few
    // dense constraints; matched greedily, they route every augmenting search
    // through an O(n) column. As MC64 keeps cheap assignments away from dense
    // columns, phase 1 solves the matching with them left out, phase 2 gives
    // them feasible duals, and phase 3 completes the matching on the full
    // graph. Duals stay feasible and matched edges tight throughout, so the
    // result is still the optimal matching.
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<std::uint8_t> dense(static_cast<std::size_t>(n), 0);
    bool any_dense = false;
    if (n > 50) {
        const double threshold = std::max(16.0, 10.0 * std::sqrt(static_cast<double>(n)));
        for (int j = 0; j < n; ++j) {
            if (static_cast<double>(col_end(j) - col_begin(j)) > threshold) {
                dense[static_cast<std::size_t>(j)] = 1;
                any_dense = true;
            }
        }
    }
    const auto is_dense = [&](int i) { return dense[static_cast<std::size_t>(i)] != 0; };

    // Initial duals on the sparse part: u[i] = min_j cost(i, j);
    // v[j] = min_i cost(i, j) - u[i]. Rows or columns without sparse entries
    // start at 0 (phase 2 makes their edges feasible).
    std::vector<double> u(static_cast<std::size_t>(n), inf);
    std::vector<double> v(static_cast<std::size_t>(n), inf);
    for (int j = 0; j < n; ++j) {
        if (is_dense(j)) continue;
        for (Int q = col_begin(j); q < col_end(j); ++q) {
            const int i = g.row[static_cast<std::size_t>(q)];
            if (is_dense(i)) continue;
            double &ui = u[static_cast<std::size_t>(i)];
            ui = std::min(ui, g.cost[static_cast<std::size_t>(q)]);
        }
    }
    for (double &ui : u) {
        if (ui == inf) ui = 0.0;
    }
    for (int j = 0; j < n; ++j) {
        double vj = inf;
        if (!is_dense(j)) {
            for (Int q = col_begin(j); q < col_end(j); ++q) {
                const int i = g.row[static_cast<std::size_t>(q)];
                if (is_dense(i)) continue;
                vj = std::min(vj, g.cost[static_cast<std::size_t>(q)] - u[static_cast<std::size_t>(i)]);
            }
        }
        v[static_cast<std::size_t>(j)] = (vj == inf) ? 0.0 : vj;
    }
    const auto reduced = [&](Int q, int j) {
        return g.cost[static_cast<std::size_t>(q)] -
               u[static_cast<std::size_t>(g.row[static_cast<std::size_t>(q)])] -
               v[static_cast<std::size_t>(j)];
    };

    // Greedy: match each sparse column to a free sparse row along a tight edge.
    int matched = 0;
    for (int j = 0; j < n; ++j) {
        if (is_dense(j)) continue;
        for (Int q = col_begin(j); q < col_end(j); ++q) {
            const int i = g.row[static_cast<std::size_t>(q)];
            if (!is_dense(i) && row_to_col[static_cast<std::size_t>(i)] < 0 && reduced(q, j) <= 0.0) {
                row_to_col[static_cast<std::size_t>(i)] = j;
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
        for (Int q = col_begin(j); q < col_end(j); ++q) {
            const int i = g.row[static_cast<std::size_t>(q)];
            if (is_dense(i) || reduced(q, j) > 0.0) continue;
            const int jj = static_cast<int>(row_to_col[static_cast<std::size_t>(i)]);
            if (jj < 0) continue;
            Int &c = cursor[static_cast<std::size_t>(jj)];
            int free_row = -1;
            for (; c < col_end(jj); ++c) {
                const int k = g.row[static_cast<std::size_t>(c)];
                if (!is_dense(k) && row_to_col[static_cast<std::size_t>(k)] < 0 && reduced(c, jj) <= 0.0) {
                    free_row = k;
                    ++c;
                    break;
                }
            }
            if (free_row < 0) continue;
            row_to_col[static_cast<std::size_t>(free_row)] = jj;
            col_to_row[static_cast<std::size_t>(jj)] = free_row;
            row_to_col[static_cast<std::size_t>(i)] = j;
            col_to_row[static_cast<std::size_t>(j)] = i;
            ++matched;
            break;
        }
    }

    // Shortest augmenting path from a free column (Dijkstra on reduced
    // costs); on success the duals are updated and the matching augmented.
    // With sparse_only, dense rows are invisible.
    std::vector<double> dist(static_cast<std::size_t>(n), inf);
    std::vector<int> parent_col(static_cast<std::size_t>(n), -1);
    std::vector<std::uint8_t> done(static_cast<std::size_t>(n), 0);
    std::vector<int> touched;
    std::vector<int> finalized;
    using HeapItem = std::pair<double, int>;
    std::vector<HeapItem> heap;
    const auto heap_cmp = std::greater<HeapItem>();
    std::vector<int> level;  // matched rows at the current distance (FIFO)

    const auto search = [&](int start_col, bool sparse_only) {
        touched.clear();
        finalized.clear();
        heap.clear();
        level.clear();
        std::size_t level_head = 0;

        // Free rows are never expanded: the closest one seen so far ends the
        // search as soon as no matched row can beat it (MC64's early exit).
        // Rows reached over a zero reduced cost sit at the current distance,
        // the minimum, so they queue in `level` without heap traffic.
        int free_row = -1;
        double path_len = inf;
        const auto relax = [&](int j, double base) {
            for (Int q = col_begin(j); q < col_end(j); ++q) {
                const int i = g.row[static_cast<std::size_t>(q)];
                if (done[static_cast<std::size_t>(i)] || (sparse_only && is_dense(i))) continue;
                const double nd = base + std::max(0.0, reduced(q, j));
                double &di = dist[static_cast<std::size_t>(i)];
                if (nd < di) {
                    if (di == inf) touched.push_back(i);
                    di = nd;
                    parent_col[static_cast<std::size_t>(i)] = j;
                    if (row_to_col[static_cast<std::size_t>(i)] < 0) {
                        if (nd < path_len) {
                            free_row = i;
                            path_len = nd;
                        }
                        // Nothing beats a free row at the current distance;
                        // stop scanning (dense columns would cost O(n)).
                        if (nd == base) return;
                    } else if (nd == base) {
                        level.push_back(i);
                    } else {
                        heap.emplace_back(nd, i);
                        std::push_heap(heap.begin(), heap.end(), heap_cmp);
                    }
                }
            }
        };

        relax(start_col, 0.0);
        double base = 0.0;
        while (true) {
            int i;
            if (level_head < level.size()) {
                if (base >= path_len) break;
                i = level[level_head++];
                if (done[static_cast<std::size_t>(i)]) continue;
            } else {
                level.clear();
                level_head = 0;
                if (heap.empty() || heap.front().first >= path_len) break;
                std::pop_heap(heap.begin(), heap.end(), heap_cmp);
                const auto [d, hi] = heap.back();
                heap.pop_back();
                if (done[static_cast<std::size_t>(hi)] || d > dist[static_cast<std::size_t>(hi)]) continue;
                i = hi;
                base = d;
            }
            done[static_cast<std::size_t>(i)] = 1;
            finalized.push_back(i);
            relax(static_cast<int>(row_to_col[static_cast<std::size_t>(i)]), base);
        }

        if (free_row >= 0) {
            // Dual update over the finalized rows (dist <= path_len) and
            // their matched columns keeps reduced costs >= 0 and makes the
            // path tight.
            v[static_cast<std::size_t>(start_col)] += path_len;
            for (int i : finalized) {
                const double delta = path_len - dist[static_cast<std::size_t>(i)];
                u[static_cast<std::size_t>(i)] -= delta;
                const int mc = static_cast<int>(row_to_col[static_cast<std::size_t>(i)]);
                if (mc >= 0) {
                    v[static_cast<std::size_t>(mc)] += delta;
                }
            }

            // Augment along the path from free_row back to start_col.
            int r = free_row;
            while (r >= 0) {
                const int c = parent_col[static_cast<std::size_t>(r)];
                const int prev_r = (c == start_col) ? -1 : static_cast<int>(col_to_row[static_cast<std::size_t>(c)]);
                row_to_col[static_cast<std::size_t>(r)] = c;
                col_to_row[static_cast<std::size_t>(c)] = r;
                r = prev_r;
            }
            ++matched;
        }

        for (int i : touched) {
            dist[static_cast<std::size_t>(i)] = inf;
            done[static_cast<std::size_t>(i)] = 0;
        }
        return free_row >= 0;
    };

    // Phase 1: the sparse part (a failed search just leaves its column for
    // phase 3).
    for (int j = 0; j < n && matched < n; ++j) {
        if (!is_dense(j) && col_to_row[static_cast<std::size_t>(j)] < 0) {
            search(j, any_dense);
        }
    }
    // Phase 2: dense rows take the largest u, then dense columns the largest
    // v, that keep every reduced cost >= 0.
    if (any_dense) {
        for (int j = 0; j < n; ++j) {
            if (is_dense(j)) u[static_cast<std::size_t>(j)] = inf;
        }
        for (int j = 0; j < n; ++j) {
            if (is_dense(j)) continue;
            for (Int q = col_begin(j); q < col_end(j); ++q) {
                const int i = g.row[static_cast<std::size_t>(q)];
                if (!is_dense(i)) continue;
                double &ui = u[static_cast<std::size_t>(i)];
                ui = std::min(ui, g.cost[static_cast<std::size_t>(q)] - v[static_cast<std::size_t>(j)]);
            }
        }
        for (int j = 0; j < n; ++j) {
            if (is_dense(j) && u[static_cast<std::size_t>(j)] == inf) u[static_cast<std::size_t>(j)] = 0.0;
        }
        for (int j = 0; j < n; ++j) {
            if (!is_dense(j)) continue;
            double vj = inf;
            for (Int q = col_begin(j); q < col_end(j); ++q) {
                vj = std::min(vj, g.cost[static_cast<std::size_t>(q)] -
                                      u[static_cast<std::size_t>(g.row[static_cast<std::size_t>(q)])]);
            }
            v[static_cast<std::size_t>(j)] = vj;
        }
    }
    // Phase 3: every column still free, on the full graph.
    for (int j = 0; j < n && matched < n; ++j) {
        if (col_to_row[static_cast<std::size_t>(j)] < 0 && !search(j, false)) {
            res.matched_rank = matched;
            res.status = ErrorCode::SingularMatrix;
            return res;
        }
    }
    res.matched_rank = matched;

    // Recompute exact dual v[j] = cost(match[j], j) - u[match[j]] for tight equality
    for (int j = 0; j < n; ++j) {
        const int r = static_cast<int>(col_to_row[static_cast<std::size_t>(j)]);
        for (Int q = col_begin(j); q < col_end(j); ++q) {
            if (g.row[static_cast<std::size_t>(q)] == r) {
                v[static_cast<std::size_t>(j)] =
                    g.cost[static_cast<std::size_t>(q)] - u[static_cast<std::size_t>(r)];
                break;
            }
        }
    }

    // Symmetrized positive diagonal scaling:
    // s_row[i] = exp(u[i]), s_col[i] = exp(v[i]) / col_max[i],
    // scale[i] = sqrt(s_row[i] * s_col[i]) = exp(0.5 * (u[i] + v[i])) / sqrt(col_max[i]).
    for (int i = 0; i < n; ++i) {
        const double log_s = 0.5 * (u[static_cast<std::size_t>(i)] + v[static_cast<std::size_t>(i)] -
                                    std::log(g.col_max[static_cast<std::size_t>(i)]));
        const double clamped_log = std::clamp(log_s, -340.0, 340.0);
        res.scale[static_cast<std::size_t>(i)] = std::exp(clamped_log);
    }

    res.status = ErrorCode::Success;
    return res;
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
