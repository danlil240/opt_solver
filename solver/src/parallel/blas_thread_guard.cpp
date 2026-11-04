#include "smf/blas_thread_guard.hpp"

#include <atomic>
#include <mutex>

#if defined(SMF_USE_MKL) && SMF_USE_MKL
#  include <mkl.h>
static int  blas_get_num_threads()        { return MKL_Get_Max_Threads(); }
// This thread's local setting: returns the previous local one (0: none,
// the global mkl_set_num_threads value applies), which blas_restore puts
// back, so a guard never turns the global value into a local override.
static int  blas_set_num_threads(int n)   { return MKL_Set_Num_Threads_Local(n); }
static void blas_restore(int prev)        { MKL_Set_Num_Threads_Local(prev); }
#  define SMF_BLAS_THREAD_API 1

#elif defined(SMF_HAS_OPENBLAS) && SMF_HAS_OPENBLAS
#  include <cblas.h>
static int  blas_get_num_threads()        { return openblas_get_num_threads(); }
// Process-wide (OpenBLAS has no per-thread setting): restores the count.
static int  blas_set_num_threads(int n)   { const int prev = openblas_get_num_threads(); openblas_set_num_threads(n); return prev; }
static void blas_restore(int prev)        { openblas_set_num_threads(prev > 0 ? prev : 1); }
#  define SMF_BLAS_THREAD_API 1

#else
static int  blas_get_num_threads()        { return 1; }
static int  blas_set_num_threads(int /*n*/) { return 0; }
static void blas_restore(int /*prev*/)    {}
#  define SMF_BLAS_THREAD_API 0
#endif

#if defined(SMF_USE_MKL) && SMF_USE_MKL
#  if defined(_WIN32)
#    ifndef NOMINMAX
#      define NOMINMAX
#    endif
#    include <windows.h>
#  else
#    include <dlfcn.h>
#  endif
#endif

namespace smf {

namespace {
// ---- Intel OpenMP runtime blocktime (see blas_thread_guard.hpp) ----------
struct BlocktimeApi {
    void (*set_fn)(int) = nullptr;
    int (*get_fn)() = nullptr;
    explicit operator bool() const { return set_fn && get_fn; }
};

#if defined(SMF_USE_MKL) && SMF_USE_MKL
BlocktimeApi resolve_blocktime_api() noexcept {
    BlocktimeApi api;
#  if defined(_WIN32)
    // Only a module that is already loaded: never LoadLibrary it.
    if (HMODULE h = GetModuleHandleA("libiomp5md.dll")) {
        api.set_fn = reinterpret_cast<void (*)(int)>(
            reinterpret_cast<void *>(GetProcAddress(h, "kmp_set_blocktime")));
        api.get_fn = reinterpret_cast<int (*)()>(
            reinterpret_cast<void *>(GetProcAddress(h, "kmp_get_blocktime")));
    }
#  else
    // RTLD_NOLOAD: succeeds only when libiomp5 is already mapped. Not
    // dlsym(RTLD_DEFAULT): LLVM libomp exports the same kmp_* names.
    if (void *h = dlopen("libiomp5.so", RTLD_LAZY | RTLD_NOLOAD)) {
        api.set_fn = reinterpret_cast<void (*)(int)>(dlsym(h, "kmp_set_blocktime"));
        api.get_fn = reinterpret_cast<int (*)()>(dlsym(h, "kmp_get_blocktime"));
    }
#  endif
    if (!api) api = BlocktimeApi{};
    return api;
}
#endif

std::atomic<void (*)(int)> g_bt_set{nullptr};   // the real runtime, once found
std::atomic<int (*)()> g_bt_get{nullptr};
std::atomic<bool> g_bt_final{false};            // lookup found, or gave up
std::atomic<bool> g_bt_hooked{false};           // test seam
std::atomic<void (*)(int)> g_hook_set{nullptr};
std::atomic<int (*)()> g_hook_get{nullptr};
std::atomic<long long> g_bt_applied{0};
#if defined(SMF_USE_MKL) && SMF_USE_MKL
// A miss can mean "MKL has not loaded its threading layer yet", so it is
// retried for a while (the guard runs once per front) before it is cached as
// final. A hit is cached at once.
constexpr int kBlocktimeMaxMisses = 256;
std::atomic<int> g_bt_misses{0};
#endif

BlocktimeApi blocktime_api() noexcept {
    if (g_bt_hooked.load(std::memory_order_acquire))
        return {g_hook_set.load(std::memory_order_relaxed), g_hook_get.load(std::memory_order_relaxed)};
    if (g_bt_final.load(std::memory_order_acquire))
        return {g_bt_set.load(std::memory_order_relaxed), g_bt_get.load(std::memory_order_relaxed)};
#if defined(SMF_USE_MKL) && SMF_USE_MKL
    const BlocktimeApi api = resolve_blocktime_api();
    if (api) {
        g_bt_set.store(api.set_fn, std::memory_order_relaxed);
        g_bt_get.store(api.get_fn, std::memory_order_relaxed);
        g_bt_final.store(true, std::memory_order_release);
        return api;
    }
    if (g_bt_misses.fetch_add(1, std::memory_order_relaxed) + 1 >= kBlocktimeMaxMisses)
        g_bt_final.store(true, std::memory_order_release);   // final: absent
    return {};
#else
    g_bt_final.store(true, std::memory_order_release);        // final: absent
    return {};
#endif
}

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
    : BlasThreadGuard(n_threads, -1)
{
}

BlasThreadGuard::BlasThreadGuard(int n_threads, int blocktime_ms) noexcept
    : saved_threads_(1), prev_setting_(0), prev_counted_(tl_counted)
{
    const int requested = (n_threads > 0) ? n_threads : 1;
    {
        std::lock_guard<std::mutex> lock(blas_guard_mutex());
        saved_threads_ = blas_get_num_threads();
        if (saved_threads_ <= 0) saved_threads_ = 1;
        prev_setting_ = blas_set_num_threads(requested);
    }
    tl_counted = requested;
    count_change(requested - prev_counted_);
    // After the first MKL call above, so the Intel runtime is loaded.
    if (blocktime_ms >= 0) {
        const int prev = mkl_blocktime_get();
        if (prev >= 0 && prev != blocktime_ms) {
            mkl_blocktime_set(blocktime_ms);
            prev_blocktime_ = prev;
        }
    }
}

BlasThreadGuard::~BlasThreadGuard() noexcept
{
    if (prev_blocktime_ >= 0) mkl_blocktime_set(prev_blocktime_);
    {
        std::lock_guard<std::mutex> lock(blas_guard_mutex());
        blas_restore(prev_setting_);
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

int blas_max_threads() noexcept { return blas_get_num_threads(); }

bool mkl_blocktime_supported() noexcept { return static_cast<bool>(blocktime_api()); }

int mkl_blocktime_get() noexcept {
    const BlocktimeApi api = blocktime_api();
    return api ? api.get_fn() : -1;
}

int mkl_blocktime_set(int ms) noexcept {
    if (ms < 0) return -1;
    const BlocktimeApi api = blocktime_api();
    if (!api) return -1;
    const int prev = api.get_fn();
    api.set_fn(ms);
    g_bt_applied.fetch_add(1, std::memory_order_relaxed);
    return prev;
}

long long mkl_blocktime_apply_count() noexcept { return g_bt_applied.load(std::memory_order_relaxed); }

void mkl_blocktime_set_hooks_for_test(void (*set_fn)(int), int (*get_fn)()) noexcept {
    g_hook_set.store(set_fn, std::memory_order_relaxed);
    g_hook_get.store(get_fn, std::memory_order_relaxed);
    g_bt_hooked.store(true, std::memory_order_release);
}

void mkl_blocktime_reset_hooks_for_test() noexcept { g_bt_hooked.store(false, std::memory_order_release); }

int blas_threads_active() noexcept { return g_active.load(std::memory_order_relaxed); }
int blas_threads_peak() noexcept { return g_peak.load(std::memory_order_relaxed); }
void blas_threads_reset_peak() noexcept {
    g_peak.store(g_active.load(std::memory_order_relaxed), std::memory_order_relaxed);
}

} // namespace smf
