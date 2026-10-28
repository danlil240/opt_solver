#include "smf/workspace.hpp"
#include "smf/solve_pool.hpp"

#include <algorithm>
#include <cmath>

namespace smf {

void SolveWorkspace::ensure_capacity(int n, int max_p, int max_q, int nrhs) {
    const std::size_t n_sz = static_cast<std::size_t>(std::max(0, n));
    const std::size_t p_sz = static_cast<std::size_t>(std::max(0, max_p));
    const std::size_t q_sz = static_cast<std::size_t>(std::max(0, max_q));
    const std::size_t nrhs_sz = static_cast<std::size_t>(std::max(1, nrhs));

    if (tmp.size() < n_sz) tmp.resize(n_sz);
    if (loc.size() < p_sz) loc.resize(p_sz);
    if (ext.size() < q_sz) ext.resize(q_sz);
    if (rhs_perm.size() < n_sz * nrhs_sz) rhs_perm.resize(n_sz * nrhs_sz);
}

SolvePool &SolveWorkspace::ensure_pool(int nthreads) {
    if (!pool || pool->size() != nthreads) {
        pool.reset();
        pool = std::make_shared<SolvePool>(nthreads);
    }
    if (threads.size() < static_cast<std::size_t>(nthreads))
        threads.resize(static_cast<std::size_t>(nthreads));
    return *pool;
}

void NumericWorkspace::prepare_for_analysis(const AnalysisKeep &ak) {
    const std::size_t ns = ak.supernodes.size();
    const std::size_t clean_nnz = ak.cleaned.values.size();
    const std::size_t perm_nnz = ak.perm_col_ptr.empty()
                                     ? clean_nnz
                                     : static_cast<std::size_t>(ak.perm_col_ptr.back());

    if (clean_values.capacity() < clean_nnz) {
        clean_values.reserve(clean_nnz);
    }
    if (perm_values.capacity() < perm_nnz) {
        perm_values.reserve(perm_nnz);
    }
    if (node_results.size() < ns) {
        node_results.resize(ns);
    }

    std::size_t max_front = 1;
    for (std::size_t s = 0; s < ns; ++s) {
        const std::size_t f = static_cast<std::size_t>(ak.front_size(static_cast<Int>(s)));
        const std::size_t p = static_cast<std::size_t>(ak.supernodes[s].width());
        if (f > max_front) max_front = f;
        auto &fr = node_results[s];
        if (fr.accepted_row_ids.capacity() < f) fr.accepted_row_ids.reserve(f);
        if (fr.l_row_ids.capacity() < f) fr.l_row_ids.reserve(f);
        if (fr.pivot_tags.capacity() < f) fr.pivot_tags.reserve(f);
        if (fr.d_blocks.capacity() < f) fr.d_blocks.reserve(f);
        if (fr.delayed_col_ids.capacity() < f) fr.delayed_col_ids.reserve(f);
        if (fr.contrib_row_ids.capacity() < f) fr.contrib_row_ids.reserve(f);
        if (fr.l_panel.capacity() < f * std::max<std::size_t>(1, p)) {
            fr.l_panel.reserve(f * std::max<std::size_t>(1, p));
        }
    }
    if (front_scratch.capacity() < max_front * max_front) {
        front_scratch.reserve(max_front * max_front);
    }
}

std::vector<double> NumericWorkspace::acquire_contrib(std::size_t elems, bool &allocated_new) {
    allocated_new = false;
    for (std::size_t i = 0; i < contrib_pool.size(); ++i) {
        if (contrib_pool[i].capacity() >= elems) {
            std::vector<double> buf = std::move(contrib_pool[i]);
            contrib_pool[i] = std::move(contrib_pool.back());
            contrib_pool.pop_back();
            contrib_pool_bytes -= buf.capacity() * sizeof(double);
            buf.assign(elems, 0.0);
            return buf;
        }
    }
    if (!contrib_pool.empty()) {
        std::vector<double> buf = std::move(contrib_pool.back());
        contrib_pool.pop_back();
        contrib_pool_bytes -= buf.capacity() * sizeof(double);
        allocated_new = true;
        buf.assign(elems, 0.0);
        return buf;
    }
    allocated_new = (elems > 0);
    return std::vector<double>(elems, 0.0);
}

void NumericWorkspace::release_contrib(std::vector<double> &&buf) {
    // Bounded pool with a running byte count: nothing on the serial path
    // re-acquires these buffers, so an unbounded pool grew by one entry per
    // front per factorization and made every release O(pool size).
    constexpr std::size_t kMaxPooledBuffers = 64;
    if (buf.capacity() == 0 || contrib_pool.size() >= kMaxPooledBuffers) return;
    contrib_pool_bytes += buf.capacity() * sizeof(double);
    if (contrib_pool_bytes > peak_pool_bytes) {
        peak_pool_bytes = contrib_pool_bytes;
    }
    buf.clear();
    contrib_pool.push_back(std::move(buf));
}

bool resolve_factor_values(const AnalysisKeep &ak, const CscLower &updated,
                           FactorValues &out) {
    out = FactorValues{updated.values.data(), updated.values.size(), true};
    if (updated.n != ak.n) return false;
    if (!ak.orig_col_ptr.empty() && updated.col_ptr == ak.orig_col_ptr &&
        updated.row_idx == ak.orig_row_idx &&
        updated.values.size() == ak.orig_row_idx.size())
        return true;
    out.original_layout = false;
    return updated.col_ptr == ak.cleaned.col_ptr && updated.row_idx == ak.cleaned.row_idx &&
           updated.values.size() == ak.cleaned.row_idx.size();
}

ErrorCode map_values_to_permuted(const AnalysisKeep &ak, const FactorValues &values,
                                 NumericWorkspace &ws, bool need_clean, double *max_abs) {
    if (max_abs != nullptr) *max_abs = 0.0;
    if (ak.n < 0) return ErrorCode::IllegalValue;
    if (ak.n == 0) {
        ws.clean_values.clear();
        ws.perm_values.clear();
        return ErrorCode::Success;
    }

    const std::size_t clean_nnz = ak.cleaned.row_idx.size();
    const std::size_t perm_nnz = ak.perm_col_ptr.empty()
                                     ? clean_nnz
                                     : static_cast<std::size_t>(ak.perm_col_ptr.back());
    // Original-layout values go through the cleaning map unless the
    // original needed none (then it is the cleaned layout).
    const bool via_orig = values.original_layout && !ak.orig_to_clean_idx.empty();
    const std::size_t expect = via_orig ? ak.orig_row_idx.size() : clean_nnz;
    if (values.data == nullptr || values.size != expect) {
        return ErrorCode::IllegalValue;
    }

    // Finiteness and max |a| in one pass. Entries dropped by cleaning (out of
    // range) are included, as before.
    const std::size_t nv = values.size;
    const double *uv = values.data;
    double vmax = 0.0;
    bool all_finite = true;
    for (std::size_t k = 0; k < nv; ++k) {
        all_finite = all_finite && is_finite_fast(uv[k]);
        vmax = std::max(vmax, std::abs(uv[k]));
    }
    if (!all_finite) {
        return ErrorCode::IllegalValue;
    }
    if (max_abs != nullptr) *max_abs = vmax;

    if (need_clean) {
        ws.clean_values.assign(clean_nnz, 0.0);
    }
    ws.perm_values.assign(perm_nnz, 0.0);
    double *pv = ws.perm_values.data();

    if (via_orig) {
        const Int *to_perm = ak.orig_to_perm_direct.data();
        for (std::size_t k = 0; k < nv; ++k) {
            const Int p_slot = to_perm[k];
            if (p_slot >= 0) pv[p_slot] += uv[k];
        }
        if (need_clean) {
            const Int *to_clean = ak.orig_to_clean_idx.data();
            double *cv = ws.clean_values.data();
            for (std::size_t k = 0; k < nv; ++k) {
                const Int c_slot = to_clean[k];
                if (c_slot >= 0) cv[c_slot] += uv[k];
            }
        }
    } else {
        if (need_clean) ws.clean_values.assign(uv, uv + nv);
        if (ak.orig_to_perm_idx.empty()) {
            std::copy(uv, uv + nv, pv);
        } else {
            const Int *to_perm = ak.orig_to_perm_idx.data();
            for (std::size_t k = 0; k < nv; ++k) pv[to_perm[k]] += uv[k];
        }
    }

    return ErrorCode::Success;
}

ErrorCode map_updated_values_to_permuted(const AnalysisKeep &ak,
                                         const CscLower &updated,
                                         NumericWorkspace &ws,
                                         bool need_clean,
                                         double *max_abs) {
    if (max_abs != nullptr) *max_abs = 0.0;
    FactorValues values;
    if (!resolve_factor_values(ak, updated, values)) return ErrorCode::IllegalValue;
    return map_values_to_permuted(ak, values, ws, need_clean, max_abs);
}

} // namespace smf
