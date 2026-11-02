#pragma once
// Synthetic KKT matrices shared by the analysis tests.

#include "smf/csc_matrix.hpp"
#include "smf/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace smf {

// OPF-like saddle-point KKT: n_x primal variables (5% with tiny curvature),
// m constraints with zero diagonal and 2-4 random Jacobian entries each.
inline CscLower opf_like_kkt(int n_x, int m, unsigned seed) {
    std::uint64_t state = seed;
    auto next = [&]() { state = state * 6364136223846793005ULL + 1442695040888963407ULL; return state >> 11; };
    auto unit = [&]() { return static_cast<double>(next()) / 9007199254740992.0; };
    const int n = n_x + m;
    std::vector<std::vector<std::pair<Int, double>>> cols(static_cast<std::size_t>(n));
    for (int j = 0; j < n_x; ++j)
        cols[static_cast<std::size_t>(j)].push_back({j, unit() < 0.05 ? 1e-5 : 1.0 + unit()});
    for (int r = 0; r < m; ++r) {
        const int row = n_x + r;
        const int k = 2 + static_cast<int>(next() % 3);
        std::vector<int> used;
        for (int t = 0; t < k; ++t) {
            const int j = static_cast<int>(next() % static_cast<std::uint64_t>(n_x));
            if (std::find(used.begin(), used.end(), j) != used.end()) continue;
            used.push_back(j);
            cols[static_cast<std::size_t>(j)].push_back({row, 0.5 + unit()});
        }
        cols[static_cast<std::size_t>(row)].push_back({row, 0.0});
    }
    CscLower A;
    A.n = n;
    A.col_ptr.assign(static_cast<std::size_t>(n) + 1, 0);
    for (int j = 0; j < n; ++j) {
        auto &c = cols[static_cast<std::size_t>(j)];
        std::sort(c.begin(), c.end());
        A.col_ptr[static_cast<std::size_t>(j) + 1] = A.col_ptr[static_cast<std::size_t>(j)] + static_cast<Int>(c.size());
        for (const auto &e : c) {
            A.row_idx.push_back(e.first);
            A.values.push_back(e.second);
        }
    }
    return A;
}

// opf_like_kkt with every Jacobian entry (an off-diagonal) scaled by
// 10^(spread * (u - 1/2)), u uniform in [0, 1) from a seeded LCG. The spread
// of magnitudes makes the matching's greedy initial phase pair columns
// worse than the full searches do.
inline CscLower scaled_opf_kkt(int n_x, int m, unsigned seed, double spread) {
    CscLower A = opf_like_kkt(n_x, m, seed);
    std::uint64_t st = seed * 7919u + 13u;
    for (Int j = 0; j < A.n; ++j) {
        for (Int q = A.col_ptr[j]; q < A.col_ptr[j + 1]; ++q) {
            if (A.row_idx[q] == j) continue;
            st = st * 6364136223846793005ULL + 1442695040888963407ULL;
            const double u = static_cast<double>(st >> 11) / 9007199254740992.0;
            A.values[q] *= std::pow(10.0, spread * (u - 0.5));
        }
    }
    return A;
}

} // namespace smf
