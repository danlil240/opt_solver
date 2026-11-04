#pragma once
// Synthetic KKT matrices shared by the analysis tests.

#include "smf/csc_matrix.hpp"
#include "smf/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
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

// Regularized KKT [H B^T; B creg I] (H: 5-point Laplacian of a g x g grid,
// diagonal hdiag; m constraint rows, row k coupling three pseudo-random
// variables; the pattern is fixed) as D A D, D_i = 10^U(-spread, spread)
// drawn from `seed` (spread 0: A itself). With creg < 0 its inertia is exact
// whatever D when H is positive definite (hdiag > 4): g^2 positive, m
// negative. With a large spread unscaled threshold pivoting delays most
// columns, many repeatedly (a cascade, as on pde_control). The generator of
// test_scale_on_delay.cpp's kkt(), but drawn from raw std::mt19937 output
// (fully specified) instead of the implementation-defined std::uniform_*
// distributions: the same matrix with MSVC, libstdc++ and libc++.
inline CscLower cascade_kkt(int g, int m, double hdiag, double spread, double creg = -0.1,
                           unsigned seed = 7) {
    const int nx = g * g, n = nx + m;
    std::mt19937 rng(7);
    std::mt19937 drng(seed);
    const auto var = [&rng, nx] { return static_cast<int>(rng() % static_cast<std::uint32_t>(nx)); };
    // u in [0, 1) from the top 27 bits of one draw: exact in binary64.
    const auto expo = [&drng, spread] {
        const double u = std::ldexp(static_cast<double>(drng() >> 5), -27);
        return spread * (2.0 * u - 1.0);
    };
    std::vector<double> d(static_cast<std::size_t>(n));
    for (double &v : d) v = std::pow(10.0, expo());
    std::vector<std::vector<std::pair<int, double>>> cols(static_cast<std::size_t>(n));
    for (int y = 0; y < g; ++y)
        for (int x = 0; x < g; ++x) {
            const int j = y * g + x;
            auto &c = cols[static_cast<std::size_t>(j)];
            c.push_back({j, hdiag});
            if (x + 1 < g) c.push_back({j + 1, -1.0});
            if (y + 1 < g) c.push_back({j + g, -1.0});
        }
    const double coef[3] = {1.0, -0.5, 0.25};
    for (int k = 0; k < m; ++k) {
        const int row = nx + k;
        int v[3];
        v[0] = var();
        v[1] = var();
        v[2] = var();
        while (v[1] == v[0]) v[1] = var();
        while (v[2] == v[0] || v[2] == v[1]) v[2] = var();
        for (int t = 0; t < 3; ++t) cols[static_cast<std::size_t>(v[t])].push_back({row, coef[t]});
        cols[static_cast<std::size_t>(row)].push_back({row, creg});
    }
    CscLower A;
    A.n = n;
    A.col_ptr.assign(1, 0);
    for (int j = 0; j < n; ++j) {
        auto &c = cols[static_cast<std::size_t>(j)];
        std::sort(c.begin(), c.end());
        for (auto [i, v] : c) {
            A.row_idx.push_back(i);
            A.values.push_back(d[static_cast<std::size_t>(i)] * v * d[static_cast<std::size_t>(j)]);
        }
        A.col_ptr.push_back(static_cast<Int>(A.row_idx.size()));
    }
    return A;
}

} // namespace smf
