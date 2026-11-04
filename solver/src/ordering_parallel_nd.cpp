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
//
// With the vendored METIS, the separators of the top kSplitSepDepth levels
// are split into tasks too (smf_sep.h): the separator's coarsening, then its
// SMF_SEP_NRUNS independent initial-separator runs as separate tasks, then
// (in the task that completes the last run) the choice and refinement. These
// separators are a third to two thirds METIS's runs, and at the top the
// queue has little else to run. Each run is seeded by its index, so the
// result still depends on the graph only.
#include "smf/ordering_parallel_nd.hpp"
#include "smf/ordering.hpp"
#include "smf/ordering_metis.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <thread>
#if defined(SMF_HAS_METIS) && SMF_HAS_METIS && defined(SMF_METIS_THREADSAFE) && SMF_METIS_THREADSAFE
#include "../third_party/metis/libmetis/smf_sep.h"
#define SMF_ND_SPLIT_SEPARATOR 1
#else
#define SMF_ND_SPLIT_SEPARATOR 0
#endif

namespace smf {

namespace {

// Parts with fewer supervariables than this are not split further.
constexpr std::size_t kMinSplit = 64;

// Separators of nodes above this depth run as split tasks (smf_sep.h):
// depths 0-2, where a few large separators leave most threads idle.
constexpr int kSplitSepDepth = 3;

#if SMF_ND_SPLIT_SEPARATOR
// A separator in flight: its graph (smf_sep_begin() keeps pointers into
// it), the METIS handle, and the runs still to finish.
struct SepJob {
    std::vector<Int> sx, sa, sw;
    smf_sep_t* h = nullptr;
    std::atomic<int> left{0};
    ~SepJob() { smf_sep_free(h); }
};
#endif

struct Node {
    std::vector<Int> verts;   // supervariables, ascending
    std::vector<Int> sep;     // separator supervariables, ascending
    std::vector<Int> order;   // leaf: original vertices in elimination order
    int child[2] = {-1, -1};
    int depth = 0;
    bool leaf = true;
#if SMF_ND_SPLIT_SEPARATOR
    std::unique_ptr<SepJob> job;  // a split separator in flight
#endif
};

struct Compressed {
    Int n = 0;                       // supervariables
    const Int* xadj = nullptr;       // compressed graph: own_xadj / own_adj, or
    const Int* adj = nullptr;        // the input's when nothing merged (no copy)
    std::vector<Int> own_xadj, own_adj;
    std::vector<Int> wgt;            // weights
    std::vector<Int> mptr, members;  // members of s: members[mptr[s] .. mptr[s+1])
    std::vector<Int> super_of;       // original vertex -> supervariable
};

// Whether every adjacency list is strictly increasing without self loops:
// what compress() would build from it when no vertices merge.
bool adjacency_is_clean(Int n, const Int* xadj, const Int* adj)
{
    for (Int v = 0; v < n; ++v)
        for (Int p = xadj[v]; p < xadj[v + 1]; ++p)
            if (adj[p] == v || (p > xadj[v] && adj[p] <= adj[p - 1])) return false;
    return true;
}

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
    // Nothing merged (super_of is the identity) and the input already in
    // the form built below: the input's adjacency itself.
    if (c.n == n && adjacency_is_clean(n, xadj, adj)) {
        c.xadj = xadj;
        c.adj = adj;
        return c;
    }
    // Neighbours of a supervariable: those of its first member, mapped.
    std::vector<Int> mark(sn, -1);
    c.own_xadj.assign(sn + 1, 0);
    for (Int s = 0; s < c.n; ++s) {
        const Int r = c.members[static_cast<std::size_t>(c.mptr[s])];
        mark[s] = s;
        const std::size_t start = c.own_adj.size();
        for (Int p = xadj[r]; p < xadj[r + 1]; ++p) {
            const Int t = c.super_of[adj[p]];
            if (mark[t] != s) {
                mark[t] = s;
                c.own_adj.push_back(t);
            }
        }
        std::sort(c.own_adj.begin() + static_cast<std::ptrdiff_t>(start), c.own_adj.end());
        c.own_xadj[static_cast<std::size_t>(s) + 1] = static_cast<Int>(c.own_adj.size());
    }
    c.xadj = c.own_xadj.data();
    c.adj = c.own_adj.data();
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

// side[] of a finished separator (0, 1, 2 per local vertex): whether it
// splits the node (both halves non-empty).
bool splits(const std::vector<Int>& side)
{
    Int count[3] = {0, 0, 0};
    for (Int s2 : side) ++count[s2];
    return count[0] > 0 && count[1] > 0;
}

// Splits node `id` (children not yet allocated): side[k] in {0, 1, 2} per
// local vertex. Returns false on a METIS failure; `split` false when the node
// stays a leaf. With `nruns` > 0 the separator is in flight instead
// (node.job, split_sep): its runs are to be queued, and side / split come
// from the task that completes them.
bool bisect(const Compressed& c, Node& node, int id, const std::vector<int>& owner,
            std::vector<Int>& loc, std::vector<Int>& side, bool& split, int& nruns)
{
    split = false;
    nruns = 0;
    const auto m = static_cast<Int>(node.verts.size());
    // The root holds every supervariable in order: its induced graph is the
    // compressed graph itself.
    const bool whole = id == 0 && m == c.n;
    std::vector<Int> sx, sa, sw;
    if (!whole) induced(c, node.verts, id, owner, loc, sx, sa, sw);
    const Int* gx = whole ? c.xadj : sx.data();
    const Int* ga = whole ? c.adj : sa.data();
    const Int* gw = whole ? c.wgt.data() : sw.data();
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
            w += gw[v];
            for (Int p = gx[v]; p < gx[v + 1]; ++p)
                if (comp[ga[p]] < 0) {
                    comp[ga[p]] = ncomp;
                    stack.push_back(ga[p]);
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
    // METIS reads the graph only (the root's is the compressed graph's).
    Int* mx = const_cast<Int*>(gx);
    Int* ma = const_cast<Int*>(ga);
    Int* mw = const_cast<Int*>(gw);
    Int sepsize = 0;
#if SMF_ND_SPLIT_SEPARATOR
    if (node.depth < kSplitSepDepth) {
        auto job = std::make_unique<SepJob>();
        if (!whole) {
            // The handle keeps pointers into the graph: move it into the job.
            job->sx = std::move(sx);
            job->sa = std::move(sa);
            job->sw = std::move(sw);
            mx = job->sx.data();
            ma = job->sa.data();
            mw = job->sw.data();
        }
        if (smf_sep_begin(m, mx, ma, mw, 0, &job->h, &nruns) != METIS_OK) return false;
        if (nruns > 0) {
            job->left.store(nruns);
            node.job = std::move(job);
            return true;
        }
        smf_sep_t* h = job->h;
        job->h = nullptr;
        if (smf_sep_finish(h, side.data(), &sepsize) != METIS_OK) return false;
        split = splits(side);
        return true;
    }
#endif
    if (!metis_vertex_separator(m, mx, ma, mw, side.data(), &sepsize))
        return false;
    split = splits(side);
    return true;
}

#if SMF_ND_SPLIT_SEPARATOR
// Run `i` of node's separator in flight. The task that completes the last
// run finishes the separator (side, split) and returns `last`.
bool split_sep(Node& node, int i, std::vector<Int>& side, bool& split, bool& last)
{
    split = false;
    last = false;
    SepJob& job = *node.job;
    if (smf_sep_run(job.h, i) != METIS_OK) return false;
    if (job.left.fetch_sub(1, std::memory_order_acq_rel) != 1) return true;
    last = true;
    side.assign(node.verts.size(), 0);
    Int sepsize = 0;
    smf_sep_t* h = job.h;
    job.h = nullptr;
    const bool ok = smf_sep_finish(h, side.data(), &sepsize) == METIS_OK;
    node.job.reset();
    if (!ok) return false;
    split = splits(side);
    return true;
}
#endif

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

    // Pending tasks, largest part first (ties: lower id, then a separator's
    // runs, lowest first, before other tasks). `run` >= 0: run `run` of node
    // id's separator in flight. The order only changes the schedule, never
    // the result.
    struct Task {
        std::size_t size;
        int id;
        int run = -1;
        bool operator<(const Task& o) const
        {
            if (size != o.size) return size < o.size;
            if (id != o.id) return id > o.id;
            return run < o.run;
        }
    };
    std::mutex mu;  // guards queue, pending, failed and nodes.push_back
    std::condition_variable cv;
    std::priority_queue<Task> queue;
    std::size_t pending = 1;  // queued or running
    bool failed = false;
    queue.push({sc, 0});

    // Splits node `id` along side[] (split) or orders it as a leaf; queues
    // its children.
    auto apply = [&](int id, Node& nd, const std::vector<Int>& side, bool split) -> bool {
        if (!split)
            return order_leaf(c, nd, id, owner, loc, oloc, options.amd_leaf_max, xadj, adjncy);
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
    };
    // Task (id, run): run < 0 splits node id (or orders it, when it stays a
    // leaf); a split separator queues its runs instead, and the run that
    // completes it applies the split.
    auto process = [&](int id, Node& nd, int run) -> bool {
        std::vector<Int> side;
        bool split = false;
#if SMF_ND_SPLIT_SEPARATOR
        if (run >= 0) {
            bool last = false;
            if (!split_sep(nd, run, side, split, last)) return false;
            return !last || apply(id, nd, side, split);
        }
#endif
        (void)run;
        if (nd.depth < options.levels && nd.verts.size() >= kMinSplit) {
            int nruns = 0;
            if (!bisect(c, nd, id, owner, loc, side, split, nruns)) return false;
            if (nruns > 0) {
                std::lock_guard<std::mutex> lk(mu);
                for (int i = 0; i < nruns; ++i) queue.push({nd.verts.size(), id, i});
                pending += static_cast<std::size_t>(nruns);
                return true;
            }
        }
        return apply(id, nd, side, split);
    };
    auto worker = [&] {
        std::unique_lock<std::mutex> lk(mu);
        for (;;) {
            cv.wait(lk, [&] { return !queue.empty() || pending == 0 || failed; });
            if (failed || queue.empty()) return;
            const int id = queue.top().id;
            const int run = queue.top().run;
            queue.pop();
            Node* nd = nodes[static_cast<std::size_t>(id)].get();
            lk.unlock();
            bool ok = false;
            try {
                ok = process(id, *nd, run);
            } catch (...) {
                ok = false;
            }
            lk.lock();
            if (!ok) failed = true;
            --pending;
            cv.notify_all();
        }
    };
    // At most 2^levels tasks are ever ready at once (the leaves; a split
    // separator's runs come before them).
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
