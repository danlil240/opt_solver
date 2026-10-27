// determinism.cpp — Deterministic parallel execution helpers.
#include "smf/determinism.hpp"

#include <atomic>
#include <mutex>

#if defined(SMF_USE_MKL) && SMF_USE_MKL
#  include <mkl.h>
#endif

namespace smf {

namespace {
std::atomic<int> g_cbwr_status{1};
std::atomic<int> g_cbwr_branch{0};
} // namespace

void enable_blas_reproducibility() noexcept {
    static std::once_flag once;
    std::call_once(once, [] {
#if defined(SMF_USE_MKL) && SMF_USE_MKL
        // MKL_CBWR_AVX2: the same results on every AVX2-capable Intel CPU.
        // MKL rejects that branch on other vendors' CPUs (AMD:
        // MKL_CBWR_ERR_UNSUPPORTED_BRANCH, only COMPATIBLE / AUTO are
        // accepted there); AUTO then keeps the results reproducible on this
        // CPU (what deterministic mode promises), without the SSE2-level
        // COMPATIBLE branch's slowdown.
        int branch = MKL_CBWR_AVX2;
        int st = mkl_cbwr_set(branch);
        if (st == MKL_CBWR_ERR_UNSUPPORTED_BRANCH) {
            branch = MKL_CBWR_AUTO;
            st = mkl_cbwr_set(branch);
        }
        g_cbwr_branch.store(st == MKL_CBWR_SUCCESS ? branch : 0, std::memory_order_relaxed);
        g_cbwr_status.store(st, std::memory_order_relaxed);
#endif
    });
}

int blas_reproducibility_status() noexcept {
    return g_cbwr_status.load(std::memory_order_relaxed);
}

int blas_reproducibility_branch() noexcept {
    return g_cbwr_branch.load(std::memory_order_relaxed);
}

void sort_children_deterministic(std::vector<int>& children) {
    std::sort(children.begin(), children.end());
}

void deterministic_reduce(std::vector<double>& accumulator,
                           const std::vector<std::vector<double>>& contributions,
                           int size) {
    accumulator.assign(static_cast<std::size_t>(size), 0.0);
    for (const auto& contrib : contributions) {
        for (int i = 0; i < size; ++i) {
            accumulator[static_cast<std::size_t>(i)] +=
                contrib[static_cast<std::size_t>(i)];
        }
    }
}

} // namespace smf
