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
    ctrl.auto_threads = false;  // small test trees
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
// through its wrappers, which serialise the calls unless the vendored,
// thread-safe METIS is linked; this checks those: 8 threads x 50
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

#if defined(SMF_METIS_THREADSAFE) && SMF_METIS_THREADSAFE
#include <metis.h>
#ifndef SMF_VENDORED_METIS_H
#error "these tests need the vendored metis.h (third_party/metis/include) first on the include path"
#endif

namespace {

// 27-point 3D grid (every vertex coupled with all of its neighbours).
Graph grid27(Int nx, Int ny, Int nz)
{
    std::vector<std::pair<Int, Int>> e;
    auto id = [&](Int i, Int j, Int k) { return i + nx * (j + ny * k); };
    for (Int k = 0; k < nz; ++k)
        for (Int j = 0; j < ny; ++j)
            for (Int i = 0; i < nx; ++i)
                for (Int dk = 0; dk <= 1; ++dk)
                    for (Int dj = -1; dj <= 1; ++dj)
                        for (Int di = -1; di <= 1; ++di) {
                            if (dk == 0 && (dj < 0 || (dj == 0 && di <= 0))) continue;
                            const Int a = i + di, b = j + dj, c = k + dk;
                            if (a < 0 || b < 0 || a >= nx || b >= ny || c >= nz) continue;
                            e.push_back({id(i, j, k), id(a, b, c)});
                        }
    return from_edges(nx * ny * nz, e);
}

struct RawMetis {
    int rc = 0;
    std::vector<idx_t> v;  // NodeND: perm; separator: part
    idx_t sepsize = 0;
};

// METIS called directly (no smf wrapper, no lock), with smf's options.
RawMetis raw_metis(const Graph& g, bool separator)
{
    std::vector<idx_t> xadj(g.xadj.begin(), g.xadj.end()), adj(g.adj.begin(), g.adj.end());
    idx_t n = g.n, opts[METIS_NOPTIONS];
    METIS_SetDefaultOptions(opts);
    opts[METIS_OPTION_SEED] = 0;
    opts[METIS_OPTION_NUMBERING] = 0;
    RawMetis r;
    r.v.assign(static_cast<std::size_t>(n), -1);
    if (separator) {
        r.rc = METIS_ComputeVertexSeparator(&n, xadj.data(), adj.data(), nullptr, opts, &r.sepsize,
                                            r.v.data());
    } else {
        std::vector<idx_t> iperm(static_cast<std::size_t>(n));
        r.rc = METIS_NodeND(&n, xadj.data(), adj.data(), nullptr, opts, r.v.data(), iperm.data());
    }
    return r;
}

std::uint64_t fnv1a(const std::vector<idx_t>& v)
{
    std::uint64_t h = 1469598103934665603ull;
    for (idx_t x : v) {
        h ^= static_cast<std::uint32_t>(x);
        h *= 1099511628211ull;
    }
    return h;
}

} // namespace

// The vendored METIS (SMF_VENDORED_METIS) keeps GKlib's state thread-local
// and its random stream per thread (third_party/README.md), so METIS itself
// may be called concurrently: 16 threads x 200 direct calls, alternating
// METIS_NodeND and METIS_ComputeVertexSeparator over 8 grids (7- and
// 27-point), every result bitwise equal to the serial one. The conda-forge
// DLL crashes this within 4 threads x 20 calls.
TEST(ParallelNd, RawConcurrentMetisIsSafe)
{
    ASSERT_TRUE(smf::metis_is_thread_safe());
    constexpr int kThreads = 16, kCalls = 200;
    std::vector<Graph> graphs;
    std::vector<bool> separator;
    for (int k = 0; k < 4; ++k) {
        graphs.push_back(grid(9 + k, 10, 8 + k % 2));
        separator.push_back(k % 2 == 1);
        graphs.push_back(grid27(7 + k, 8, 6 + k % 2));
        separator.push_back(k % 2 == 0);
    }
    std::vector<RawMetis> serial;
    for (std::size_t k = 0; k < graphs.size(); ++k) {
        serial.push_back(raw_metis(graphs[k], separator[k]));
        ASSERT_EQ(serial.back().rc, METIS_OK) << "graph " << k;
    }
    std::vector<int> mismatches(kThreads, 0);
    std::vector<std::thread> pool;
    for (int t = 0; t < kThreads; ++t)
        pool.emplace_back([&, t] {
            for (int c = 0; c < kCalls; ++c) {
                const std::size_t k = static_cast<std::size_t>(t * 7 + c) % graphs.size();
                const RawMetis r = raw_metis(graphs[k], separator[k]);
                if (r.rc != METIS_OK || r.v != serial[k].v || r.sepsize != serial[k].sepsize)
                    ++mismatches[static_cast<std::size_t>(t)];
            }
        });
    for (auto& th : pool) th.join();
    for (int t = 0; t < kThreads; ++t)
        EXPECT_EQ(mismatches[static_cast<std::size_t>(t)], 0) << "thread " << t;
}

// The vendored METIS gives the orderings of the METIS smf linked before
// (conda-forge metis 5.2.1 on Windows): its random stream is a thread-local
// copy of MSVC's rand() (third_party/README.md, patch 2), so the order is
// also the same on Linux. Reference hashes recorded from conda-forge's
// metis.dll: 20^3 7-point NodeND perm, 30x12x10 7-point vertex separator.
TEST(ParallelNd, VendoredMetisMatchesCondaOrdering)
{
    const RawMetis nd = raw_metis(grid(20, 20, 20), false);
    ASSERT_EQ(nd.rc, METIS_OK);
    EXPECT_EQ(fnv1a(nd.v), 0xb1fef4826789ffbbull);
    const RawMetis sep = raw_metis(grid(30, 12, 10), true);
    ASSERT_EQ(sep.rc, METIS_OK);
    EXPECT_EQ(sep.sepsize, 120);
    EXPECT_EQ(fnv1a(sep.v), 0xf97248da81e2b681ull);
}
#endif

#if defined(SMF_METIS_THREADSAFE) && SMF_METIS_THREADSAFE
#include <cstddef>
#include "../third_party/gklib/gk_mksort.h"

namespace {
struct SortKv {
    int key;
    int tag;
};
#define SMF_SORTKV_LT(a, b) ((a)->key < (b)->key)
#define SMF_SORTKV_GT(a, b) ((a)->key > (b)->key)
void gk_sort_lt(SortKv* a, std::size_t n) { GK_MKQSORT(SortKv, a, n, SMF_SORTKV_LT); }
void gk_sort_gt(SortKv* a, std::size_t n) { GK_MKQSORT(SortKv, a, n, SMF_SORTKV_GT); }
} // namespace

// GK_MKQSORT (third_party/gklib/gk_mksort.h, smf patch 4) replaces GKlib's
// glibc-derived (LGPL) sort. METIS sorts key-value pairs by key only
// (ikvsorti in compress.c and coarsen.c), so the order of equal keys feeds
// the orderings: it must be the original's exactly. Pinned: FNV-1a of the
// tag sequence the ORIGINAL macro produced on random keys with many
// duplicates (one xorshift stream across the cases), ascending and
// descending. A scratch harness matched the two on 38,928 arrays.
TEST(ParallelNd, GkSortPinnedOrder)
{
    struct Case {
        std::size_t n;
        int range;
        int desc;
        std::uint64_t hash;
    };
    const Case cases[] = {
        {0, 1, 0, 0x14650fb0739d0383ull},      {1, 1, 0, 0x44bd2bd473ccf799ull},
        {9, 3, 0, 0x1fe5c0b800afea73ull},      {10, 3, 1, 0xf265370049768c60ull},
        {37, 3, 0, 0x3e54681c76a2e03bull},     {200, 2, 1, 0x88c4e38d6c38f923ull},
        {1000, 5, 0, 0xe09b80adbe1b22c1ull},   {1000, 5, 1, 0x8462e491aac55507ull},
        {4099, 1000, 0, 0x42355cf94d973058ull}, {100000, 50, 1, 0x98de7e13c43d83c5ull},
    };
    std::uint64_t s = 88172645463325252ull;
    for (const Case& c : cases) {
        std::vector<SortKv> a(c.n + 1);
        for (std::size_t i = 0; i < c.n; ++i) {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            a[i].key = static_cast<int>(s % static_cast<std::uint64_t>(c.range));
            a[i].tag = static_cast<int>(i);
        }
        if (c.desc) gk_sort_gt(a.data(), c.n);
        else gk_sort_lt(a.data(), c.n);
        std::uint64_t h = 1469598103934665603ull;
        for (std::size_t i = 0; i < c.n; ++i) {
            h ^= static_cast<std::uint32_t>(a[i].tag);
            h *= 1099511628211ull;
        }
        EXPECT_EQ(h, c.hash) << "n " << c.n << " range " << c.range << " desc " << c.desc;
        for (std::size_t i = 1; i < c.n; ++i)
            ASSERT_FALSE(c.desc ? SMF_SORTKV_GT(&a[i], &a[i - 1]) : SMF_SORTKV_LT(&a[i], &a[i - 1]))
                << "not sorted at " << i;
    }
}
#endif
