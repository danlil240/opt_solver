#include "smf/scaling_matching.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace smf {

namespace {

struct WeightedEdge {
    int to_row = 0;
    double abs_val = 0.0;
    double cost = 0.0;
};

} // namespace

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

    // Build full symmetric adjacency list per column (merging duplicates)
    std::vector<std::vector< std::pair<int, double> >> raw_cols(static_cast<std::size_t>(n));
    for (int j = 0; j < n; ++j) {
        for (Int p = A.col_ptr[static_cast<std::size_t>(j)];
             p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
            const int i = static_cast<int>(A.row_idx[static_cast<std::size_t>(p)]);
            const double v = A.values[static_cast<std::size_t>(p)];
            if (!std::isfinite(v)) {
                res.status = ErrorCode::IllegalValue;
                return res;
            }
            raw_cols[static_cast<std::size_t>(j)].push_back({i, v});
            if (i != j) {
                raw_cols[static_cast<std::size_t>(i)].push_back({j, v});
            }
        }
    }

    std::vector<std::vector<WeightedEdge>> col_edges(static_cast<std::size_t>(n));
    std::vector<double> col_max(static_cast<std::size_t>(n), 0.0);

    for (int j = 0; j < n; ++j) {
        auto &vec = raw_cols[static_cast<std::size_t>(j)];
        std::sort(vec.begin(), vec.end(), [](const auto &a, const auto &b) {
            return a.first < b.first;
        });
        auto &edges = col_edges[static_cast<std::size_t>(j)];
        std::size_t pos = 0;
        while (pos < vec.size()) {
            const int row = vec[pos].first;
            double sum = vec[pos].second;
            ++pos;
            while (pos < vec.size() && vec[pos].first == row) {
                sum += vec[pos].second;
                ++pos;
            }
            const double av = std::abs(sum);
            if (av > 0.0) {
                edges.push_back(WeightedEdge{row, av, 0.0});
                if (av > col_max[static_cast<std::size_t>(j)]) {
                    col_max[static_cast<std::size_t>(j)] = av;
                }
            }
        }
        if (col_max[static_cast<std::size_t>(j)] <= 0.0) {
            res.status = ErrorCode::SingularMatrix;
            return res;
        }
        const double log_cmax = std::log(col_max[static_cast<std::size_t>(j)]);
        for (auto &e : edges) {
            e.cost = std::max(0.0, log_cmax - std::log(e.abs_val));
        }
    }

    // Shortest augmenting path (Dijkstra with dual variables u[row], v[col])
    // Dual feasibility: u[i] + v[j] <= cost(i, j), with equality on matched edges.
    std::vector<double> u(static_cast<std::size_t>(n), 0.0);
    std::vector<double> v(static_cast<std::size_t>(n), 0.0);

    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> dist(static_cast<std::size_t>(n), inf);
    std::vector<int> parent_col(static_cast<std::size_t>(n), -1);
    std::vector<uint8_t> visited_row(static_cast<std::size_t>(n), 0);

    for (int start_col = 0; start_col < n; ++start_col) {
        std::fill(dist.begin(), dist.end(), inf);
        std::fill(parent_col.begin(), parent_col.end(), -1);
        std::fill(visited_row.begin(), visited_row.end(), 0);

        int cur_col = start_col;
        int matched_free_row = -1;
        double shortest_path_len = 0.0;

        while (true) {
            const double v_cur = v[static_cast<std::size_t>(cur_col)];
            for (const auto &e : col_edges[static_cast<std::size_t>(cur_col)]) {
                const int r = e.to_row;
                if (visited_row[static_cast<std::size_t>(r)]) continue;
                const double reduced = e.cost - u[static_cast<std::size_t>(r)] - v_cur;
                const double cand_dist = shortest_path_len + std::max(0.0, reduced);
                if (cand_dist < dist[static_cast<std::size_t>(r)]) {
                    dist[static_cast<std::size_t>(r)] = cand_dist;
                    parent_col[static_cast<std::size_t>(r)] = cur_col;
                }
            }

            // Pick unvisited row with minimum dist
            int best_row = -1;
            double best_dist = inf;
            for (int r = 0; r < n; ++r) {
                if (!visited_row[static_cast<std::size_t>(r)] &&
                    dist[static_cast<std::size_t>(r)] < best_dist) {
                    best_dist = dist[static_cast<std::size_t>(r)];
                    best_row = r;
                }
            }

            if (best_row < 0 || !std::isfinite(best_dist)) {
                break;
            }

            visited_row[static_cast<std::size_t>(best_row)] = 1;
            shortest_path_len = best_dist;

            const int next_col = res.row_to_col[static_cast<std::size_t>(best_row)];
            if (next_col < 0) {
                matched_free_row = best_row;
                break;
            }
            cur_col = next_col;
        }

        if (matched_free_row < 0) {
            res.status = ErrorCode::SingularMatrix;
            return res;
        }

        // Update dual potentials u and v along visited alternating tree
        v[static_cast<std::size_t>(start_col)] += shortest_path_len;
        for (int r = 0; r < n; ++r) {
            if (!visited_row[static_cast<std::size_t>(r)]) continue;
            const double delta = shortest_path_len - dist[static_cast<std::size_t>(r)];
            u[static_cast<std::size_t>(r)] -= delta;
            const int mc = res.row_to_col[static_cast<std::size_t>(r)];
            if (mc >= 0) {
                v[static_cast<std::size_t>(mc)] += delta;
            }
        }

        // Augment matching along path from matched_free_row back to start_col
        int r = matched_free_row;
        while (r >= 0) {
            const int c = parent_col[static_cast<std::size_t>(r)];
            const int prev_r = (c == start_col) ? -1 : res.col_to_row[static_cast<std::size_t>(c)];
            res.row_to_col[static_cast<std::size_t>(r)] = c;
            res.col_to_row[static_cast<std::size_t>(c)] = r;
            r = prev_r;
        }
        ++res.matched_rank;
    }

    // Recompute exact dual v[j] = cost(match[j], j) - u[match[j]] for tight equality
    for (int j = 0; j < n; ++j) {
        const int r = res.col_to_row[static_cast<std::size_t>(j)];
        if (r >= 0) {
            for (const auto &e : col_edges[static_cast<std::size_t>(j)]) {
                if (e.to_row == r) {
                    v[static_cast<std::size_t>(j)] = e.cost - u[static_cast<std::size_t>(r)];
                    break;
                }
            }
        }
    }

    // Symmetrized positive diagonal scaling:
    // s_row[i] = exp(u[i]), s_col[i] = exp(v[i]) / col_max[i],
    // scale[i] = sqrt(s_row[i] * s_col[i]) = exp(0.5 * (u[i] + v[i])) / sqrt(col_max[i]).
    for (int i = 0; i < n; ++i) {
        const double log_s = 0.5 * (u[static_cast<std::size_t>(i)] + v[static_cast<std::size_t>(i)] -
                                    std::log(col_max[static_cast<std::size_t>(i)]));
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
