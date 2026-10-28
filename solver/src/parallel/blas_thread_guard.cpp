#include "smf/blas_thread_guard.hpp"

#include <atomic>
#include <mutex>

#if defined(SMF_USE_MKL) && SMF_USE_MKL
#  include <mkl.h>
static int  blas_get_num_threads()        { return MKL_Get_Max_Threads(); }
static void blas_set_num_threads(int n)   { MKL_Set_Num_Threads_Local(n); }
#  define SMF_BLAS_THREAD_API 1

#elif defined(SMF_HAS_OPENBLAS) && SMF_HAS_OPENBLAS
#  include <cblas.h>
static int  blas_get_num_threads()        { return openblas_get_num_threads(); }
static void blas_set_num_threads(int n)   { openblas_set_num_threads(n); }
#  define SMF_BLAS_THREAD_API 1

#else
static int  blas_get_num_threads()        { return 1; }
static void blas_set_num_threads(int /*n*/) {}
#  define SMF_BLAS_THREAD_API 0
#endif

namespace smf {

namespace {
std::mutex &blas_guard_mutex() {
    static std::mutex mtx;
    return mtx;
}

// BLAS-thread accounting (blas_threads_active): this thread's count, the
// sum over threads, and its peak.
thread_local int tl_counted = 0;
std::atomic<int> g_active{0};
std::atomic<int> g_peak{0};

void count_change(int delta) noexcept {
    const int now = g_active.fetch_add(delta, std::memory_order_relaxed) + delta;
    int peak = g_peak.load(std::memory_order_relaxed);
    while (now > peak && !g_peak.compare_exchange_weak(peak, now, std::memory_order_relaxed)) {
    }
}
} // namespace

BlasThreadGuard::BlasThreadGuard(int n_threads) noexcept
    : saved_threads_(1), prev_counted_(tl_counted)
{
    const int requested = (n_threads > 0) ? n_threads : 1;
    {
        std::lock_guard<std::mutex> lock(blas_guard_mutex());
        saved_threads_ = blas_get_num_threads();
        if (saved_threads_ <= 0) saved_threads_ = 1;
        blas_set_num_threads(requested);
    }
    tl_counted = requested;
    count_change(requested - prev_counted_);
}

BlasThreadGuard::~BlasThreadGuard() noexcept
{
    {
        std::lock_guard<std::mutex> lock(blas_guard_mutex());
        blas_set_num_threads(saved_threads_);
    }
    count_change(prev_counted_ - tl_counted);
    tl_counted = prev_counted_;
}

BlasIdleScope::BlasIdleScope() noexcept : prev_counted_(tl_counted)
{
    tl_counted = 0;
    count_change(-prev_counted_);
}

BlasIdleScope::~BlasIdleScope() noexcept
{
    count_change(prev_counted_ - tl_counted);
    tl_counted = prev_counted_;
}

int blas_threads_active() noexcept { return g_active.load(std::memory_order_relaxed); }
int blas_threads_peak() noexcept { return g_peak.load(std::memory_order_relaxed); }
void blas_threads_reset_peak() noexcept {
    g_peak.store(g_active.load(std::memory_order_relaxed), std::memory_order_relaxed);
}

} // namespace smf
