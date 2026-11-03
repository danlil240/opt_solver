// Parallel nested dissection with AMD leaves and exact compression.
//
// 1. Vertices with identical closed adjacency are merged into weighted
//    supervariables (bmwcra_1 148770 -> 53877, pwtk 217918 -> 41531).
// 2. The compressed graph is dissected to a fixed depth (kParallelNdLevels):
//    a part with several connected components is split along them with an
//    empty separator, otherwise METIS_ComputeVertexSeparator splits it.
// 3. Leaves are ordered: below kParallelNdAmdLeafMax supervariables AMD on
//    the uncompressed leaf (AMD has no vertex weights; its own supervariable
//    detection sees the true degrees), members then gathered at their
//    supervariable's first position; otherwise METIS_NodeND on the
//    compressed leaf with weights.
// 4. Separators are numbered after both halves at every level.
//
// Splits and leaves are tasks on one queue served by std::threads, largest
// part first: a part's children are queued as soon as it is split, with no
// barrier between levels. METIS calls go through the wrappers of
// ordering_metis.cpp, which run them concurrently with the vendored,
// thread-safe METIS and serialise them otherwise. The tree depends on the
// graph only, and each METIS call reseeds, so the order is the same for
// every thread count and schedule.
#include "smf/ordering_parallel_nd.hpp"
#include "smf/ordering.hpp"
#include "smf/ordering_metis.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <thread>

namespace smf {

namespace {

// Parts with fewer supervariables than this are not split further.
constexpr std::size_t kMinSplit = 64;

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
// vertices only, so several nodes can build theirs concurrently).
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
    // Nodes live behind unique_ptrs: a task keeps its Node* while others
    // append children. owner[] (supervariable -> node id, -1 = separator),
    // loc[] and oloc[] are written by a node's task for its own vertices only;
    // a part's vertices have neighbours only in the part itself and in the
    // separators above it, so concurrent tasks never touch the same entries.
    std::vector<std::unique_ptr<Node>> nodes;
    nodes.push_back(std::make_unique<Node>());
    nodes[0]->verts.resize(sc);
    std::iota(nodes[0]->verts.begin(), nodes[0]->verts.end(), 0);
    std::vector<int> owner(sc, 0);
    std::vector<Int> loc(sc), oloc(static_cast<std::size_t>(n));

    // Pending tasks, largest part first (ties: lower id). The order only
    // changes the schedule, never the result.
    struct Task {
        std::size_t size;
        int id;
        bool operator<(const Task& o) const { return size != o.size ? size < o.size : id > o.id; }
    };
    std::mutex mu;  // guards queue, pending, failed and nodes.push_back
    std::condition_variable cv;
    std::priority_queue<Task> queue;
    std::size_t pending = 1;  // queued or running
    bool failed = false;
    queue.push({sc, 0});

    // Splits node `id` (or orders it, when it stays a leaf); queues its children.
    auto process = [&](int id, Node& nd) -> bool {
        if (nd.depth < options.levels && nd.verts.size() >= kMinSplit) {
            std::vector<Int> side;
            bool split = false;
            if (!bisect(c, nd, id, owner, loc, side, split)) return false;
            if (split) {
                Node half[2];
                std::vector<Int> sep;
                for (std::size_t v = 0; v < nd.verts.size(); ++v) {
                    const Int s2 = side[v];
                    (s2 == 2 ? sep : half[s2].verts).push_back(nd.verts[v]);
                }
                Node* child[2];
                int cid[2];
                {
                    std::lock_guard<std::mutex> lk(mu);
                    for (int h = 0; h < 2; ++h) {
                        half[h].depth = nd.depth + 1;
                        cid[h] = static_cast<int>(nodes.size());
                        nodes.push_back(std::make_unique<Node>(std::move(half[h])));
                        child[h] = nodes.back().get();
                    }
                }
                for (int h = 0; h < 2; ++h) {
                    for (Int v : child[h]->verts) owner[v] = cid[h];
                    nd.child[h] = cid[h];
                }
                for (Int v : sep) owner[v] = -1;
                nd.sep = std::move(sep);
                nd.leaf = false;
                nd.verts.clear();
                nd.verts.shrink_to_fit();
                std::lock_guard<std::mutex> lk(mu);
                for (int h = 0; h < 2; ++h) queue.push({child[h]->verts.size(), cid[h]});
                pending += 2;
                return true;
            }
        }
        return order_leaf(c, nd, id, owner, loc, oloc, options.amd_leaf_max, xadj, adjncy);
    };
    auto worker = [&] {
        std::unique_lock<std::mutex> lk(mu);
        for (;;) {
            cv.wait(lk, [&] { return !queue.empty() || pending == 0 || failed; });
            if (failed || queue.empty()) return;
            const int id = queue.top().id;
            queue.pop();
            Node* nd = nodes[static_cast<std::size_t>(id)].get();
            lk.unlock();
            bool ok = false;
            try {
                ok = process(id, *nd);
            } catch (...) {
                ok = false;
            }
            lk.lock();
            if (!ok) failed = true;
            --pending;
            cv.notify_all();
        }
    };
    // At most 2^levels tasks are ever ready at once (the leaves).
    const int nthreads = std::min(num_threads, 1 << std::clamp(options.levels, 0, 16));
    std::vector<std::thread> pool;
    try {
        for (int k = 1; k < nthreads; ++k) pool.emplace_back(worker);
    } catch (...) {
        // Thread creation failed: the threads already running finish the work.
    }
    worker();
    for (auto& th : pool) th.join();
    if (failed) return false;

    // Children before their separator, recursively (depth <= levels).
    Int pos = 0;
    auto emit = [&](auto&& self, int id) -> void {
        const Node& nd = *nodes[static_cast<std::size_t>(id)];
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
