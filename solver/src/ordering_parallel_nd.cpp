// Parallel nested dissection with AMD leaves and exact compression.
//
// 1. Vertices with identical closed adjacency are merged into weighted
//    supervariables (bmwcra_1 148770 -> 53877, pwtk 217918 -> 41531).
// 2. The compressed graph is dissected to a fixed depth (kParallelNdLevels):
//    a part with several connected components is split along them with an
//    empty separator, otherwise METIS_ComputeVertexSeparator splits it. The
//    parts of one level are processed concurrently.
// 3. Leaves are ordered concurrently: below kParallelNdAmdLeafMax
//    supervariables AMD on the uncompressed leaf (AMD has no vertex weights;
//    its own supervariable detection sees the true degrees), members then
//    gathered at their supervariable's first position; otherwise
//    METIS_NodeND on the compressed leaf with weights.
// 4. Separators are numbered after both halves at every level.
//
// METIS calls go through the serialised wrappers of ordering_metis.cpp (the
// linked METIS is not thread-safe); everything else runs on std::threads.
// The tree depends on the graph only, and each METIS call reseeds, so the
// order is the same for every thread count.
#include "smf/ordering_parallel_nd.hpp"
#include "smf/ordering.hpp"
#include "smf/ordering_metis.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <numeric>
#include <thread>

namespace smf {

namespace {

// Parts with fewer supervariables than this are not split further.
constexpr std::size_t kMinSplit = 64;

// Runs f(0..count-1) on up to `threads` std::threads (the caller's included).
// Returns false when a task threw.
template <class F>
bool run_tasks(std::size_t count, int threads, F&& f)
{
    std::atomic<bool> ok{true};
    std::atomic<std::size_t> next{0};
    auto body = [&] {
        for (std::size_t i; (i = next.fetch_add(1)) < count;) {
            try {
                if (!f(i)) ok = false;
            } catch (...) {
                ok = false;
            }
        }
    };
    const std::size_t t = std::min<std::size_t>(static_cast<std::size_t>(std::max(threads, 1)), count);
    std::vector<std::thread> pool;
    try {
        for (std::size_t k = 1; k < t; ++k) pool.emplace_back(body);
    } catch (...) {
        // Thread creation failed: the caller's thread finishes the work.
    }
    body();
    for (auto& th : pool) th.join();
    return ok;
}

struct Node {
    std::vector<Int> verts;   // supervariables, ascending
    std::vector<Int> sep;     // separator supervariables, ascending
    std::vector<Int> order;   // leaf: original vertices in elimination order
    int child[2] = {-1, -1};
    int depth = 0;
    bool leaf = true;
};

struct Compressed {
    Int n = 0;                       // supervariables
    std::vector<Int> xadj, adj, wgt; // compressed graph, weights
    std::vector<Int> mptr, members;  // members of s: members[mptr[s] .. mptr[s+1])
    std::vector<Int> super_of;       // original vertex -> supervariable
};

Compressed compress(Int n, const Int* xadj, const Int* adj, const Int* vwgt)
{
    Compressed c;
    c.n = compress_closed_adjacency(n, xadj, adj, c.super_of);
    const auto sn = static_cast<std::size_t>(c.n);
    c.mptr.assign(sn + 1, 0);
    for (Int v = 0; v < n; ++v) ++c.mptr[static_cast<std::size_t>(c.super_of[v]) + 1];
    for (std::size_t s = 0; s < sn; ++s) c.mptr[s + 1] += c.mptr[s];
    c.members.resize(static_cast<std::size_t>(n));
    c.wgt.assign(sn, 0);
    {
        std::vector<Int> fill(c.mptr.begin(), c.mptr.end() - 1);
        for (Int v = 0; v < n; ++v) {
            const Int s = c.super_of[v];
            c.members[static_cast<std::size_t>(fill[s]++)] = v;
            c.wgt[s] += vwgt ? vwgt[v] : 1;
        }
    }
    // Neighbours of a supervariable: those of its first member, mapped.
    std::vector<Int> mark(sn, -1);
    c.xadj.assign(sn + 1, 0);
    for (Int s = 0; s < c.n; ++s) {
        const Int r = c.members[static_cast<std::size_t>(c.mptr[s])];
        mark[s] = s;
        const std::size_t start = c.adj.size();
        for (Int p = xadj[r]; p < xadj[r + 1]; ++p) {
            const Int t = c.super_of[adj[p]];
            if (mark[t] != s) {
                mark[t] = s;
                c.adj.push_back(t);
            }
        }
        std::sort(c.adj.begin() + static_cast<std::ptrdiff_t>(start), c.adj.end());
        c.xadj[static_cast<std::size_t>(s) + 1] = static_cast<Int>(c.adj.size());
    }
    return c;
}

// Induced subgraph of the compressed graph on node `id`'s vertices; owner[]
// marks membership and loc[] receives local indices (written for this node's
// vertices only, so nodes of one level can build theirs concurrently).
void induced(const Compressed& c, const std::vector<Int>& verts, int id, const std::vector<int>& owner,
             std::vector<Int>& loc, std::vector<Int>& sx, std::vector<Int>& sa, std::vector<Int>& sw)
{
    const std::size_t m = verts.size();
    for (std::size_t k = 0; k < m; ++k) loc[verts[k]] = static_cast<Int>(k);
    sx.assign(m + 1, 0);
    sa.clear();
    sw.resize(m);
    for (std::size_t k = 0; k < m; ++k) {
        const Int v = verts[k];
        sw[k] = c.wgt[v];
        for (Int p = c.xadj[v]; p < c.xadj[v + 1]; ++p) {
            const Int u = c.adj[p];
            if (owner[u] == id) sa.push_back(loc[u]);
        }
        sx[k + 1] = static_cast<Int>(sa.size());
    }
}

// Splits node `id` (children not yet allocated): side[k] in {0, 1, 2} per
// local vertex. Returns false on a METIS failure; `split` false when the node
// stays a leaf.
bool bisect(const Compressed& c, Node& node, int id, const std::vector<int>& owner,
            std::vector<Int>& loc, std::vector<Int>& side, bool& split)
{
    split = false;
    std::vector<Int> sx, sa, sw;
    induced(c, node.verts, id, owner, loc, sx, sa, sw);
    const auto m = static_cast<Int>(node.verts.size());
    side.assign(static_cast<std::size_t>(m), 0);

    // Connected components (labels in order of their smallest vertex).
    std::vector<Int> comp(static_cast<std::size_t>(m), -1), stack;
    std::vector<Int> cw;
    Int ncomp = 0;
    for (Int s = 0; s < m; ++s) {
        if (comp[s] >= 0) continue;
        comp[s] = ncomp;
        Int w = 0;
        stack.assign(1, s);
        while (!stack.empty()) {
            const Int v = stack.back();
            stack.pop_back();
            w += sw[v];
            for (Int p = sx[v]; p < sx[v + 1]; ++p)
                if (comp[sa[p]] < 0) {
                    comp[sa[p]] = ncomp;
                    stack.push_back(sa[p]);
                }
        }
        cw.push_back(w);
        ++ncomp;
    }
    if (ncomp > 1) {
        // Heaviest components first, each to the lighter side.
        std::vector<Int> idx(static_cast<std::size_t>(ncomp));
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(), [&](Int a, Int b) { return cw[a] > cw[b]; });
        std::vector<Int> to(static_cast<std::size_t>(ncomp));
        std::int64_t w0 = 0, w1 = 0;
        for (Int k : idx) {
            if (w0 <= w1) { to[k] = 0; w0 += cw[k]; }
            else { to[k] = 1; w1 += cw[k]; }
        }
        for (Int v = 0; v < m; ++v) side[v] = to[comp[v]];
        split = true;
        return true;
    }
    Int sepsize = 0;
    if (!metis_vertex_separator(m, sx.data(), sa.data(), sw.data(), side.data(), &sepsize))
        return false;
    Int count[3] = {0, 0, 0};
    for (Int v = 0; v < m; ++v) ++count[side[v]];
    split = count[0] > 0 && count[1] > 0;
    return true;
}

// Orders a leaf into node.order (original vertices).
bool order_leaf(const Compressed& c, Node& node, int id, const std::vector<int>& owner,
                std::vector<Int>& loc, std::vector<Int>& oloc, Int amd_leaf_max,
                const Int* xadj, const Int* adj)
{
    const std::size_t m = node.verts.size();
    node.order.clear();
    if (static_cast<Int>(m) < amd_leaf_max) {
        // AMD on the uncompressed leaf.
        std::vector<Int> orig;
        for (Int s : node.verts)
            for (Int p = c.mptr[s]; p < c.mptr[s + 1]; ++p) orig.push_back(c.members[static_cast<std::size_t>(p)]);
        const auto mo = static_cast<Int>(orig.size());
        for (Int k = 0; k < mo; ++k) oloc[orig[k]] = k;
        std::vector<Int> ox(static_cast<std::size_t>(mo) + 1, 0), oa;
        for (Int k = 0; k < mo; ++k) {
            const Int o = orig[k];
            for (Int p = xadj[o]; p < xadj[o + 1]; ++p) {
                const Int u = adj[p];
                if (owner[c.super_of[u]] == id) oa.push_back(oloc[u]);
            }
            ox[static_cast<std::size_t>(k) + 1] = static_cast<Int>(oa.size());
        }
        std::vector<Int> lp(static_cast<std::size_t>(mo));
        std::vector<Int> work(amd_adjacency_work_size(mo, oa.size()));
        if (!amd_order_adjacency(mo, ox.data(), oa.data(), lp.data(), work.data()))
            return false;
        // Gather each supervariable at its first member's position (members
        // are indistinguishable, so this adds no fill).
        std::vector<char> done(m, 0);
        for (std::size_t k = 0; k < m; ++k) loc[node.verts[k]] = static_cast<Int>(k);
        for (Int k = 0; k < mo; ++k) {
            const Int s = c.super_of[orig[lp[k]]];
            char& d = done[static_cast<std::size_t>(loc[s])];
            if (d) continue;
            d = 1;
            for (Int p = c.mptr[s]; p < c.mptr[s + 1]; ++p) node.order.push_back(c.members[static_cast<std::size_t>(p)]);
        }
        return true;
    }
    std::vector<Int> sx, sa, sw;
    induced(c, node.verts, id, owner, loc, sx, sa, sw);
    std::vector<Int> lp(m), lip(m);
    if (!metis_node_nd(static_cast<Int>(m), sx.data(), sa.data(), sw.data(), lp.data(), lip.data()))
        return false;
    for (std::size_t k = 0; k < m; ++k) {
        const Int s = node.verts[static_cast<std::size_t>(lp[k])];
        for (Int p = c.mptr[s]; p < c.mptr[s + 1]; ++p) node.order.push_back(c.members[static_cast<std::size_t>(p)]);
    }
    return true;
}

} // namespace

Int compress_closed_adjacency(Int n, const Int* xadj, const Int* adjncy, std::vector<Int>& super_of)
{
    const auto sn = static_cast<std::size_t>(std::max<Int>(n, 0));
    super_of.assign(sn, -1);
    if (n <= 0) return 0;
    // Key: the sum of the closed neighbourhood; equal sets have equal keys.
    std::vector<std::uint64_t> key(sn);
    for (Int v = 0; v < n; ++v) {
        std::uint64_t k = static_cast<std::uint64_t>(v);
        for (Int p = xadj[v]; p < xadj[v + 1]; ++p) k += static_cast<std::uint64_t>(adjncy[p]);
        key[static_cast<std::size_t>(v)] = k;
    }
    std::vector<Int> rep(sn, -1), mark(sn, -1);
    for (Int i = 0; i < n; ++i) {
        if (rep[i] >= 0) continue;
        rep[i] = i;
        const Int deg = xadj[i + 1] - xadj[i];
        bool marked = false;
        // Indistinguishable vertices are adjacent, so candidates are neighbours.
        for (Int p = xadj[i]; p < xadj[i + 1]; ++p) {
            const Int j = adjncy[p];
            if (j <= i || rep[j] >= 0 || key[j] != key[i] || xadj[j + 1] - xadj[j] != deg) continue;
            if (!marked) {
                mark[i] = i;
                for (Int q = xadj[i]; q < xadj[i + 1]; ++q) mark[adjncy[q]] = i;
                marked = true;
            }
            // Same size and N[j] within N[i] (j's own mark comes from N[i]).
            bool same = true;
            for (Int q = xadj[j]; q < xadj[j + 1] && same; ++q) same = mark[adjncy[q]] == i;
            if (same) rep[j] = i;
        }
    }
    Int ns = 0;
    for (Int v = 0; v < n; ++v)
        super_of[v] = rep[v] == v ? ns++ : super_of[rep[v]];
    return ns;
}

bool parallel_nd_order(Int n, const Int* xadj, const Int* adjncy, const Int* vwgt, int num_threads,
                       Int* perm, const ParallelNdOptions& options)
{
    if (n <= 2) {
        std::iota(perm, perm + std::max<Int>(n, 0), 0);
        return true;
    }
#if !(defined(SMF_HAS_METIS) && SMF_HAS_METIS)
    (void)xadj; (void)adjncy; (void)vwgt; (void)num_threads; (void)options;
    return false;
#else
    const Compressed c = compress(n, xadj, adjncy, vwgt);
    const auto sc = static_cast<std::size_t>(c.n);
    std::vector<Node> nodes(1);
    nodes[0].verts.resize(sc);
    std::iota(nodes[0].verts.begin(), nodes[0].verts.end(), 0);
    std::vector<int> owner(sc, 0);
    std::vector<Int> loc(sc), oloc(static_cast<std::size_t>(n));

    // Level by level: split every part of the level concurrently.
    std::vector<int> level{0};
    for (int depth = 0; depth < options.levels && !level.empty(); ++depth) {
        std::vector<std::vector<Int>> sides(level.size());
        std::vector<char> split(level.size(), 0);
        const bool ok = run_tasks(level.size(), num_threads, [&](std::size_t k) {
            Node& nd = nodes[static_cast<std::size_t>(level[k])];
            if (nd.verts.size() < kMinSplit) return true;
            bool s = false;
            if (!bisect(c, nd, level[k], owner, loc, sides[k], s)) return false;
            split[k] = s ? 1 : 0;
            return true;
        });
        if (!ok) return false;
        std::vector<int> next;
        for (std::size_t k = 0; k < level.size(); ++k) {
            if (!split[k]) continue;
            const int id = level[k];
            Node half[2];
            std::vector<Int> sep;
            const std::vector<Int>& verts = nodes[static_cast<std::size_t>(id)].verts;
            for (std::size_t v = 0; v < verts.size(); ++v) {
                const Int s = sides[k][v];
                (s == 2 ? sep : half[s].verts).push_back(verts[v]);
            }
            for (int h = 0; h < 2; ++h) {
                half[h].depth = depth + 1;
                const int cid = static_cast<int>(nodes.size());
                for (Int v : half[h].verts) owner[v] = cid;
                nodes.push_back(std::move(half[h]));
                nodes[static_cast<std::size_t>(id)].child[h] = cid;
                next.push_back(cid);
            }
            Node& parent = nodes[static_cast<std::size_t>(id)];
            for (Int v : sep) owner[v] = -1;
            parent.sep = std::move(sep);
            parent.leaf = false;
            parent.verts.clear();
            parent.verts.shrink_to_fit();
        }
        level = std::move(next);
    }

    // Leaves, largest first (scheduling only; the result does not depend on it).
    std::vector<int> leaves;
    for (std::size_t i = 0; i < nodes.size(); ++i)
        if (nodes[i].leaf) leaves.push_back(static_cast<int>(i));
    std::stable_sort(leaves.begin(), leaves.end(), [&](int a, int b) {
        return nodes[static_cast<std::size_t>(a)].verts.size() > nodes[static_cast<std::size_t>(b)].verts.size();
    });
    const bool ok = run_tasks(leaves.size(), num_threads, [&](std::size_t k) {
        return order_leaf(c, nodes[static_cast<std::size_t>(leaves[k])], leaves[k], owner, loc, oloc,
                          options.amd_leaf_max, xadj, adjncy);
    });
    if (!ok) return false;

    // Children before their separator, recursively (depth <= levels).
    Int pos = 0;
    auto emit = [&](auto&& self, int id) -> void {
        const Node& nd = nodes[static_cast<std::size_t>(id)];
        if (nd.leaf) {
            for (Int v : nd.order) perm[pos++] = v;
            return;
        }
        self(self, nd.child[0]);
        self(self, nd.child[1]);
        for (Int s : nd.sep)
            for (Int p = c.mptr[s]; p < c.mptr[s + 1]; ++p) perm[pos++] = c.members[static_cast<std::size_t>(p)];
    };
    emit(emit, 0);
    return pos == n;
#endif
}

bool parallel_nd_order(Int n, const std::vector<Int>& xadj, const std::vector<Int>& adjncy,
                       const std::vector<Int>* vwgt, int num_threads, std::vector<Int>& perm)
{
    perm.resize(static_cast<std::size_t>(std::max<Int>(n, 0)));
    return parallel_nd_order(n, xadj.data(), adjncy.data(), vwgt ? vwgt->data() : nullptr,
                             num_threads, perm.data());
}

} // namespace smf
