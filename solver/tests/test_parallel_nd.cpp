// Parallel nested dissection (ordering_parallel_nd.cpp): validity, fill
// against serial METIS, components, compression, determinism across thread
// counts, and the METIS thread-safety gate it relies on.
#include <gtest/gtest.h>

#include "smf/control.hpp"
#include "smf/info.hpp"
#include "smf/matrix_market.hpp"
#include "smf/ordering.hpp"
#include "smf/ordering_metis.hpp"
#include "smf/ordering_parallel_nd.hpp"
#include "smf/solver.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

using smf::Int;

namespace {

struct Graph {
    Int n = 0;
    std::vector<Int> xadj, adj;
};

Graph from_edges(Int n, const std::vector<std::pair<Int, Int>>& edges)
{
    std::vector<std::vector<Int>> nb(static_cast<std::size_t>(n));
    for (auto [i, j] : edges) {
        nb[static_cast<std::size_t>(i)].push_back(j);
        nb[static_cast<std::size_t>(j)].push_back(i);
    }
    Graph g;
    g.n = n;
    g.xadj.push_back(0);
    for (auto& v : nb) {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        g.adj.insert(g.adj.end(), v.begin(), v.end());
        g.xadj.push_back(static_cast<Int>(g.adj.size()));
    }
    return g;
}

// 7-point 3D grid (nz = 1: 5-point 2D), vertices offset by `base`.
void grid_edges(Int nx, Int ny, Int nz, Int base, std::vector<std::pair<Int, Int>>& e)
{
    auto id = [&](Int i, Int j, Int k) { return base + i + nx * (j + ny * k); };
    for (Int k = 0; k < nz; ++k)
        for (Int j = 0; j < ny; ++j)
            for (Int i = 0; i < nx; ++i) {
                if (i + 1 < nx) e.push_back({id(i, j, k), id(i + 1, j, k)});
                if (j + 1 < ny) e.push_back({id(i, j, k), id(i, j + 1, k)});
                if (k + 1 < nz) e.push_back({id(i, j, k), id(i, j, k + 1)});
            }
}

Graph grid(Int nx, Int ny, Int nz)
{
    std::vector<std::pair<Int, Int>> e;
    grid_edges(nx, ny, nz, 0, e);
    return from_edges(nx * ny * nz, e);
}

// `dof` copies of every vertex, strided: dof d of node v is d * n + v. Copies
// of a node are coupled with each other and with every copy of its neighbours.
Graph multi_dof(const Graph& g, Int dof)
{
    std::vector<std::pair<Int, Int>> e;
    for (Int v = 0; v < g.n; ++v) {
        for (Int a = 0; a < dof; ++a)
            for (Int b = a + 1; b < dof; ++b)
                e.push_back({a * g.n + v, b * g.n + v});
        for (Int p = g.xadj[v]; p < g.xadj[v + 1]; ++p) {
            const Int u = g.adj[p];
            if (u < v) continue;
            for (Int a = 0; a < dof; ++a)
                for (Int b = 0; b < dof; ++b)
                    e.push_back({a * g.n + v, b * g.n + u});
        }
    }
    return from_edges(dof * g.n, e);
}

bool is_permutation(const std::vector<Int>& p, Int n)
{
    if (static_cast<Int>(p.size()) != n) return false;
    std::vector<char> seen(static_cast<std::size_t>(n), 0);
    for (Int v : p) {
        if (v < 0 || v >= n || seen[v]) return false;
        seen[v] = 1;
    }
    return true;
}

// SPD lower CSC with the pattern of g (diagonally dominant Laplacian).
smf::CscLower spd_from_graph(const Graph& g)
{
    smf::CscLower A;
    A.n = g.n;
    A.col_ptr.push_back(0);
    for (Int j = 0; j < g.n; ++j) {
        A.row_idx.push_back(j);
        A.values.push_back(static_cast<double>(g.xadj[j + 1] - g.xadj[j]) + 1.0);
        for (Int p = g.xadj[j]; p < g.xadj[j + 1]; ++p)
            if (g.adj[p] > j) {
                A.row_idx.push_back(g.adj[p]);
                A.values.push_back(-1.0);
            }
        A.col_ptr.push_back(static_cast<Int>(A.row_idx.size()));
    }
    return A;
}

double predicted_flops(const smf::CscLower& A, smf::OrderingMethod method, const Int* perm)
{
    smf::Control ctrl;
    ctrl.matrix_type = smf::MatrixType::RealSymmetricPositiveDefinite;
    ctrl.ordering = method;
    ctrl.user_perm = perm;
    ctrl.num_threads = 1;
    smf::Info info;
    smf::Solver solver;
    auto ak = solver.analyse(A, ctrl, info);
    EXPECT_NE(ak, nullptr);
    return info.predicted_flops;
}

void expect_close_to_metis(const smf::CscLower& A, const Graph& g, int threads, const char* what)
{
    std::vector<Int> perm;
    ASSERT_TRUE(smf::parallel_nd_order(g.n, g.xadj, g.adj, nullptr, threads, perm)) << what;
    ASSERT_TRUE(is_permutation(perm, g.n)) << what;
    const double nd = predicted_flops(A, smf::OrderingMethod::UserSupplied, perm.data());
    const double metis = predicted_flops(A, smf::OrderingMethod::METIS, nullptr);
    std::printf("[ParallelNd] %s: flops ratio parallel ND / serial METIS = %.4f (%.4g / %.4g)\n",
                what, nd / metis, nd, metis);
    EXPECT_LE(nd, 1.15 * metis) << what;
}

// bmwcra_1 from the large fixtures (the bench .smfbin cache when present).
bool load_large(const char* name, smf::CscLower& A)
{
    namespace fs = std::filesystem;
    const fs::path mtx = fs::path(__FILE__).parent_path() / ".." / ".." / "benchmarks" /
                         "fixtures" / "large" / name / (std::string(name) + ".mtx");
    std::error_code ec;
    if (!fs::exists(mtx, ec)) return false;
    const fs::path bin = fs::path(mtx.string() + ".smfbin");
    if (fs::exists(bin, ec)) {
        std::ifstream in(bin, std::ios::binary);
        char magic[8];
        std::int64_t n = 0, nnz = 0;
        in.read(magic, 8);
        in.read(reinterpret_cast<char*>(&n), sizeof n);
        in.read(reinterpret_cast<char*>(&nnz), sizeof nnz);
        if (in && std::memcmp(magic, "SMFBIN01", 8) == 0 && n > 0 && nnz > 0) {
            A.n = static_cast<Int>(n);
            A.col_ptr.resize(static_cast<std::size_t>(n) + 1);
            A.row_idx.resize(static_cast<std::size_t>(nnz));
            A.values.resize(static_cast<std::size_t>(nnz));
            in.read(reinterpret_cast<char*>(A.col_ptr.data()),
                    static_cast<std::streamsize>(A.col_ptr.size() * sizeof(Int)));
            in.read(reinterpret_cast<char*>(A.row_idx.data()),
                    static_cast<std::streamsize>(A.row_idx.size() * sizeof(Int)));
            in.read(reinterpret_cast<char*>(A.values.data()),
                    static_cast<std::streamsize>(A.values.size() * sizeof(double)));
            if (in) return true;
        }
    }
    auto mm = smf::read_matrix_market(mtx.string());
    if (!mm.ok) return false;
    A = std::move(mm.matrix);
    return true;
}

} // namespace

TEST(ParallelNd, ValidPermutation)
{
    const Graph g = grid(30, 30, 30);
    expect_close_to_metis(spd_from_graph(g), g, 2, "poisson3d_30");

    smf::CscLower A;
    if (!load_large("bmwcra_1", A)) {
        std::printf("[ParallelNd] bmwcra_1 fixture not present; skipped that part\n");
        return;
    }
    Graph gb;
    gb.n = A.n;
    std::tie(gb.xadj, gb.adj) = smf::build_symmetric_csr(A.n, A.col_ptr, A.row_idx);
    expect_close_to_metis(A, gb, 2, "bmwcra_1");
}

TEST(ParallelNd, DisconnectedAndTiny)
{
    // Two disjoint 20x20 grids: the top split is along the components, so
    // one grid is numbered entirely before the other.
    std::vector<std::pair<Int, Int>> e;
    grid_edges(20, 20, 1, 0, e);
    grid_edges(20, 20, 1, 400, e);
    const Graph g = from_edges(800, e);
    std::vector<Int> perm;
    ASSERT_TRUE(smf::parallel_nd_order(g.n, g.xadj, g.adj, nullptr, 2, perm));
    ASSERT_TRUE(is_permutation(perm, g.n));
    const bool first = perm[0] < 400;
    for (Int k = 0; k < 400; ++k)
        ASSERT_EQ(perm[k] < 400, first) << "separator vertices at position " << k;

    for (Int n = 0; n <= 2; ++n) {
        const Graph t = n == 2 ? from_edges(2, {{0, 1}}) : from_edges(n, {});
        std::vector<Int> p;
        ASSERT_TRUE(smf::parallel_nd_order(n, t.xadj, t.adj, nullptr, 4, p));
        ASSERT_EQ(static_cast<Int>(p.size()), n);
        for (Int k = 0; k < n; ++k) EXPECT_EQ(p[k], k);
    }
    // Two isolated vertices (no edges at all).
    const Graph t2 = from_edges(2, {});
    std::vector<Int> p2;
    ASSERT_TRUE(smf::parallel_nd_order(2, t2.xadj, t2.adj, nullptr, 1, p2));
    EXPECT_EQ(p2, (std::vector<Int>{0, 1}));
}

TEST(ParallelNd, CompressionExpands)
{
    const Graph base = grid(24, 24, 1);
    const Graph g = multi_dof(base, 3);
    std::vector<Int> super_of;
    const Int ns = smf::compress_closed_adjacency(g.n, g.xadj.data(), g.adj.data(), super_of);
    EXPECT_EQ(ns, base.n);
    for (Int v = 0; v < base.n; ++v) {
        EXPECT_EQ(super_of[v], super_of[base.n + v]);
        EXPECT_EQ(super_of[v], super_of[2 * base.n + v]);
    }
    // The base grid itself has no indistinguishable vertices.
    std::vector<Int> s2;
    EXPECT_EQ(smf::compress_closed_adjacency(base.n, base.xadj.data(), base.adj.data(), s2), base.n);

    // Default, AMD leaves throughout and METIS_NodeND leaves throughout.
    for (Int amd_leaf_max : {smf::kParallelNdAmdLeafMax, Int{1 << 30}, Int{0}}) {
        smf::ParallelNdOptions opt;
        opt.amd_leaf_max = amd_leaf_max;
        std::vector<Int> perm(static_cast<std::size_t>(g.n));
        ASSERT_TRUE(smf::parallel_nd_order(g.n, g.xadj.data(), g.adj.data(), nullptr, 2,
                                           perm.data(), opt));
        ASSERT_TRUE(is_permutation(perm, g.n));
        std::vector<Int> pos(static_cast<std::size_t>(g.n));
        for (Int k = 0; k < g.n; ++k) pos[perm[k]] = k;
        for (Int v = 0; v < base.n; ++v) {
            const Int lo = std::min({pos[v], pos[base.n + v], pos[2 * base.n + v]});
            const Int hi = std::max({pos[v], pos[base.n + v], pos[2 * base.n + v]});
            ASSERT_EQ(hi - lo, 2) << "dofs of node " << v << " not adjacent (leaf max "
                                  << amd_leaf_max << ")";
        }
    }
}

TEST(ParallelNd, DeterministicAcrossThreads)
{
    const Graph g = multi_dof(grid(14, 14, 14), 2);
    std::vector<Int> ref;
    ASSERT_TRUE(smf::parallel_nd_order(g.n, g.xadj, g.adj, nullptr, 1, ref));
    ASSERT_TRUE(is_permutation(ref, g.n));
    for (int rep = 0; rep < 20; ++rep)
        for (int threads : {1, 2, 8}) {
            std::vector<Int> p;
            ASSERT_TRUE(smf::parallel_nd_order(g.n, g.xadj, g.adj, nullptr, threads, p));
            ASSERT_EQ(p, ref) << "rep " << rep << " threads " << threads;
        }
}

#if defined(SMF_HAS_METIS) && SMF_HAS_METIS
// Gate for running METIS from several std::threads. The raw conda-forge
// METIS 5.2.1 aborts the process when called concurrently (GKlib's memory
// core is process-global; see task-6-report.md), so smf only calls METIS
// through the serialised wrappers, and this checks those: 8 threads x 50
// METIS_NodeND (+ vertex separator) calls equal the serial results.
TEST(ParallelNd, ConcurrentMetisIsSafe)
{
    constexpr int kGraphs = 8, kThreads = 8, kCalls = 50;
    std::vector<Graph> graphs;
    for (int k = 0; k < kGraphs; ++k) graphs.push_back(grid(10 + k, 11, 9 + k % 3));
    struct Result {
        std::vector<Int> perm, part;
        Int sepsize = 0;
        bool ok = false;
    };
    auto run = [](const Graph& g) {
        Result r;
        std::vector<Int> xadj = g.xadj, adj = g.adj;
        std::vector<Int> iperm(static_cast<std::size_t>(g.n));
        r.perm.resize(static_cast<std::size_t>(g.n));
        r.part.resize(static_cast<std::size_t>(g.n));
        const bool a = smf::metis_node_nd(g.n, xadj.data(), adj.data(), nullptr, r.perm.data(),
                                          iperm.data());
        const bool b = smf::metis_vertex_separator(g.n, xadj.data(), adj.data(), nullptr,
                                                   r.part.data(), &r.sepsize);
        r.ok = a && b;
        return r;
    };
    std::vector<Result> serial;
    for (const Graph& g : graphs) serial.push_back(run(g));
    for (const Result& r : serial) ASSERT_TRUE(r.ok);

    std::vector<int> mismatches(kThreads, 0);
    std::vector<std::thread> pool;
    for (int t = 0; t < kThreads; ++t)
        pool.emplace_back([&, t] {
            for (int c = 0; c < kCalls; ++c) {
                const int k = (t + c) % kGraphs;
                const Result r = run(graphs[static_cast<std::size_t>(k)]);
                const Result& s = serial[static_cast<std::size_t>(k)];
                if (!r.ok || r.perm != s.perm || r.part != s.part || r.sepsize != s.sepsize)
                    ++mismatches[static_cast<std::size_t>(t)];
            }
        });
    for (auto& th : pool) th.join();
    for (int t = 0; t < kThreads; ++t)
        EXPECT_EQ(mismatches[static_cast<std::size_t>(t)], 0) << "thread " << t;
}
#endif
