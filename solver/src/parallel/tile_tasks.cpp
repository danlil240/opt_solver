// 2D tile tasks for the fronts at the top of the assembly tree (see
// tile_tasks.hpp for the task graph).

#include "tile_tasks.hpp"

#include "../aptp_internal.hpp"
#include "smf/blas_thread_guard.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <thread>

namespace smf {

// ---------------------------------------------------------------------------
// TileRunner
// ---------------------------------------------------------------------------

void TileRunner::run(TileDag &dag, int *pending, const int *roots, int nroots, int total,
                     int *ready_hi, int *ready_lo, int nthreads, TileHelpers *helpers) {
    if (total <= 0) return;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        dag_ = &dag;
        pending_ = pending;
        hi_ = ready_hi;
        lo_ = ready_lo;
        nhi_ = 0;
        lo_head_ = 0;
        lo_tail_ = 0;
        done_ = 0;
        total_ = total;
        for (int r = 0; r < nroots; ++r) {
            if (dag.urgent(roots[r])) hi_[nhi_++] = roots[r];
            else lo_[lo_tail_++] = roots[r];
        }
    }
    // The caller's share of BLAS threads is spread over the helpers: it runs
    // serial BLAS from before the first helper joins until the last has left.
    BlasSerialGuard serial;
    // Helpers: idle threads of the caller's pool, or threads of our own
    // (standalone use; the vector stays empty and unallocated otherwise).
    std::vector<std::thread> own;
    if (nthreads > 1) {
        if (helpers != nullptr) {
            helpers->offer(*this, nthreads - 1);
        } else {
            {
                std::lock_guard<std::mutex> lk(mtx_);
                helpers_ += nthreads - 1;
            }
            own.reserve(static_cast<std::size_t>(nthreads) - 1);
            for (int t = 1; t < nthreads; ++t) own.emplace_back([this] { help(); });
        }
    }
    work();
    if (nthreads > 1 && helpers != nullptr) helpers->withdraw(*this);
    for (auto &th : own) th.join();
    std::unique_lock<std::mutex> lk(mtx_);
    cv_.wait(lk, [&] { return helpers_ == 0; });
    dag_ = nullptr;
}

void TileRunner::release(int task) {
    if (--pending_[task] != 0) return;
    if (dag_->urgent(task)) hi_[nhi_++] = task;
    else lo_[lo_tail_++] = task;
    ++pushed_;
}

void TileRunner::add_helper() {
    std::lock_guard<std::mutex> lk(mtx_);
    ++helpers_;
}

void TileRunner::help() {
    {
        BlasSerialGuard serial;
        work();
    }
    std::lock_guard<std::mutex> lk(mtx_);
    --helpers_;
    cv_.notify_all();
}

void TileRunner::work() {
    std::unique_lock<std::mutex> lk(mtx_);
    for (;;) {
        cv_.wait(lk, [&] { return nhi_ > 0 || lo_head_ < lo_tail_ || done_ == total_; });
        if (nhi_ == 0 && lo_head_ == lo_tail_) return; // all done
        const int t = (nhi_ > 0) ? hi_[--nhi_] : lo_[lo_head_++];
        lk.unlock();
        dag_->execute(t);
        lk.lock();
        pushed_ = 0;
        dag_->release(t, *this);
        ++done_;
        // This thread takes one of the new tasks itself.
        if (done_ == total_ || pushed_ > 1) cv_.notify_all();
    }
}

// ---------------------------------------------------------------------------
// APTP front as tile tasks
// ---------------------------------------------------------------------------

namespace {

using namespace aptp_detail;

// Grid of a sweep over the m x m front with ne eligible columns: K block
// columns of width nb over [0, ne), NR row blocks: the K eligible ones, then
// blocks of tb rows over [ne, m). nb is APTP's effective block width (at
// most ne), tb the tile size (AptpParams::nb), so a front with few eligible
// columns still gets tiles of tb rows.
struct Grid {
    int nb = 1, tb = 1, ne = 0, m = 0, K = 0, NR = 0;
    int r0(int i) const { return i < K ? i * nb : ne + (i - K) * tb; }
    int r1(int i) const {
        return i < K ? std::min((i + 1) * nb, ne) : std::min(ne + (i - K + 1) * tb, m);
    }
};

// Task ids: Factor(k) = k, Finish(k) = K + k, ApplyN(i, k) = 2K + k NR + i,
// Update(i, j, k) = baseU + k TU + tri(j) + i - j, where tri(j) = sum over
// j' < j of (NR - j') numbers the lower-trapezoid tiles (i >= j) of the grid.
struct AptpTiles final : TileDag {
    Sweep *sw = nullptr;
    Ctx *cx = nullptr;
    Grid g;
    int *ne_t = nullptr;  // K: pivots of each diagonal tile
    int *nel = nullptr;   // K: accepted pivots of each block column
    int *tfail = nullptr; // NR: ApplyN result per row block (current k)
    int *ucol = nullptr;  // TU: block column j of each lower-trapezoid tile
    int baseA = 0, baseU = 0, TU = 0;

    int tri(int j) const { return j * g.NR - j * (j - 1) / 2; }
    int idF(int k) const { return k; }
    int idR(int k) const { return g.K + k; }
    int idA(int i, int k) const { return baseA + k * g.NR + i; }
    int idU(int i, int j, int k) const { return baseU + k * TU + tri(j) + (i - j); }
    void decode_u(int t, int &i, int &j, int &k) const {
        const int r = t - baseU;
        k = r / TU;
        const int q = r - k * TU;
        j = ucol[q];
        i = j + (q - tri(j));
    }
    int width(int k) const { return std::min(g.nb, g.ne - k * g.nb); }

    void execute(int t) override {
        const int K = g.K;
        if (t < K) {
            ne_t[t] = block_factor(*sw, t * g.nb, width(t), *cx);
        } else if (t < 2 * K) {
            const int k = t - K;
            int nelim = ne_t[k];
            for (int i = k + 1; i < g.NR; ++i) nelim = std::min(nelim, tfail[i]);
            nel[k] = block_finish(*sw, k * g.nb, width(k), ne_t[k], nelim, UpdateNN::OwnBlock, *cx);
        } else if (t < baseU) {
            const int r = t - baseA;
            const int k = r / g.NR;
            const int i = r - k * g.NR;
            tfail[i] = apply_n(*sw, k * g.nb, ne_t[k], g.r0(i), g.r1(i), *cx);
        } else {
            int i, j, k;
            decode_u(t, i, j, k);
            update_tile(*sw, k * g.nb, nel[k], g.r0(j), g.r1(j), g.r0(i), g.r1(i));
        }
    }

    void release(int t, TileRunner &run) override {
        const int K = g.K;
        if (t < K) {
            for (int i = t + 1; i < g.NR; ++i) run.release(idA(i, t));
            run.release(idR(t));
        } else if (t < 2 * K) {
            const int k = t - K;
            for (int j = k + 1; j < K; ++j)
                for (int i = j; i < g.NR; ++i) run.release(idU(i, j, k));
            if (k + 1 < K) run.release(idF(k + 1));
        } else if (t < baseU) {
            run.release(idR((t - baseA) / g.NR));
        } else {
            int i, j, k;
            decode_u(t, i, j, k);
            if (j == k + 1) run.release(idF(k + 1));
            else run.release(idU(i, j, k + 1));
        }
    }

    // Critical path: the block-column chain and the updates of the next
    // block column (lookahead).
    bool urgent(int t) const override {
        if (t < baseU) return true;
        int i, j, k;
        decode_u(t, i, j, k);
        return j == k + 1;
    }
};

// Deferred update of the non-eligible trailing triangle: FormW(j) = j for
// its block columns, then Tile(i, j) = KT + j KT + i, i >= j.
struct TrailingTiles final : TileDag {
    const Sweep *sw = nullptr;
    int nb = 1, KT = 0;
    int c0(int j) const { return sw->ne + j * nb; }
    int c1(int j) const { return std::min(sw->ne + (j + 1) * nb, sw->m); }
    void execute(int t) override {
        if (t < KT) {
            trailing_form_w(*sw, c0(t), c1(t));
        } else {
            const int r = t - KT;
            const int j = r / KT;
            const int i = r - j * KT;
            trailing_tile(*sw, c0(j), c1(j), c0(i), c1(i));
        }
    }
    void release(int t, TileRunner &run) override {
        if (t >= KT) return;
        for (int i = t; i < KT; ++i) run.release(KT + t * KT + i);
    }
    bool urgent(int t) const override { return t < KT; }
};

// Task-id spaces beyond this fall back to the serial kernel: some 50000
// eligible columns at nb = 256 (a 20 GB dense front), ~50 MB of scratch.
constexpr std::size_t kMaxTileIds = std::size_t{1} << 22;

int *grow_ints(TileWorkspace &tws, std::size_t need) {
    if (tws.ints.size() < need) {
        if (tws.ints.capacity() < need) {
            ++tws.growths;
            tws.ints.reserve(std::max(need, tws.ints.capacity() + tws.ints.capacity() / 2));
        }
        tws.ints.resize(need);
    }
    return tws.ints.data();
}

} // namespace

AptpResult ldlt_aptp_front_tiled(double *F, int f, int num_elig, Int *rows, const AptpParams &prm,
                                 int8_t *tags, double *work, TileWorkspace &tws, int nthreads,
                                 TileHelpers *helpers) {
    if (f <= 0 || num_elig <= 0) {
        AptpResult res;
        res.num_failed = std::max(0, num_elig);
        return res;
    }
    Ctx cx = make_ctx(f, num_elig, prm, work);
    Grid g;
    g.nb = cx.nb;
    g.tb = std::max(cx.nb, prm.nb);
    g.ne = num_elig;
    g.m = f;
    g.K = (num_elig + g.nb - 1) / g.nb;
    g.NR = g.K + (f - num_elig + g.tb - 1) / g.tb;
    const std::size_t K = static_cast<std::size_t>(g.K);
    const std::size_t NR = static_cast<std::size_t>(g.NR);
    const std::size_t TU = K * NR - K * (K - 1) / 2; // tiles (i >= j) over j < K
    const std::size_t KT = NR - K;
    const std::size_t ids1 = 2 * K + K * NR + K * TU;
    const std::size_t ids2 = KT + KT * KT;
    const std::size_t ids = std::max(ids1, ids2);
    if (ids > kMaxTileIds) return ldlt_aptp_front(F, f, num_elig, rows, prm, tags, work);
    // Tasks of each DAG (the ready queues' room).
    std::size_t tasks1 = 2 * K;
    for (std::size_t k = 0; k < K; ++k) {
        const std::size_t tri_k1 = (k + 1) * NR - (k + 1) * k / 2;
        tasks1 += (NR - 1 - k) + (TU - tri_k1);
    }
    const std::size_t tasks = std::max(tasks1, KT + KT * (KT + 1) / 2);

    // ints: pending (ids), ready hi / lo (tasks each), ne_t, nel (K), tfail
    // (NR), ucol (TU).
    int *ip = grow_ints(tws, ids + 2 * tasks + 2 * K + NR + TU);
    int *pending = ip;
    int *hi = pending + ids;
    int *lo = hi + tasks;

    Sweep sw = sweep_begin(F, f, f, num_elig, 0, tags, rows, cx.top_lp, cx);
    AptpTiles dag;
    dag.sw = &sw;
    dag.cx = &cx;
    dag.g = g;
    dag.ne_t = lo + tasks;
    dag.nel = dag.ne_t + K;
    dag.tfail = dag.nel + K;
    dag.ucol = dag.tfail + NR;
    for (int j = 0; j < g.K; ++j)
        for (int i = j; i < g.NR; ++i) dag.ucol[dag.tri(j) + (i - j)] = j;
    dag.baseA = static_cast<int>(2 * K);
    dag.baseU = static_cast<int>(2 * K + K * NR);
    dag.TU = static_cast<int>(TU);

    int total = 0;
    for (int k = 0; k < g.K; ++k) {
        pending[dag.idF(k)] = (k == 0) ? 0 : 1 + (g.NR - k);
        pending[dag.idR(k)] = 1 + (g.NR - 1 - k);
        total += 2;
        for (int i = k + 1; i < g.NR; ++i) {
            pending[dag.idA(i, k)] = 1;
            ++total;
        }
        for (int j = k + 1; j < g.K; ++j)
            for (int i = j; i < g.NR; ++i) {
                pending[dag.idU(i, j, k)] = (k == 0) ? 1 : 2;
                ++total;
            }
    }
    const int root = dag.idF(0);
    TileRunner runner;
    assert(static_cast<std::size_t>(total) == tasks1);
    runner.run(dag, pending, &root, 1, total, hi, lo, nthreads, helpers);

    sweep_compact(sw);

    if (f > num_elig && sw.num_elim > 0) {
        TrailingTiles tr;
        tr.sw = &sw;
        tr.nb = g.tb;
        tr.KT = static_cast<int>(KT);
        int t2 = 0;
        int *roots = dag.ne_t; // 2K + NR >= KT ints, free after the main DAG
        for (int j = 0; j < tr.KT; ++j) {
            pending[j] = 0;
            roots[j] = j;
            ++t2;
            for (int i = j; i < tr.KT; ++i) {
                pending[tr.KT + j * tr.KT + i] = 1;
                ++t2;
            }
        }
        runner.run(tr, pending, roots, tr.KT, t2, hi, lo, nthreads, helpers);
    }
    return front_result(F, f, num_elig, sw.num_elim, prm, tags);
}

} // namespace smf
