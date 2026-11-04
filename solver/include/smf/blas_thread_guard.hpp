#pragma once
// blas_thread_guard.hpp — RAII guard that sets BLAS thread count and restores
// on destruction.
//
// Compile-time selection:
//   SMF_USE_MKL=1      → MKL_Set_Num_Threads_Local (this thread) / MKL_Get_Max_Threads
//   SMF_HAS_OPENBLAS=1 → openblas_set_num_threads / openblas_get_num_threads
//   otherwise          → no-op (guard compiles but does nothing)
//
// Usage inside an OpenMP task (force serial BLAS):
//   smf::BlasThreadGuard _g(1);
//
// Usage for a large front (parallel BLAS):
//   smf::BlasThreadGuard _g(control.num_threads);

#include <optional>

namespace smf {

struct Control;

/// Intel OpenMP runtime blocktime (KMP_BLOCKTIME) control, for the threads
/// that drive MKL's thread teams (the MKL "intel" threading layer). The
/// setting is per calling thread (kmp_set_blocktime sets the internal control
/// variable of that thread's team), so BlasThreadGuard applies it on whichever
/// thread constructs it: every pool worker running a top front, and the
/// caller of a serial-tree factorization.
///
/// The functions are resolved at run time from an Intel runtime that is
/// ALREADY loaded in the process (Windows: libiomp5md.dll via GetModuleHandle;
/// Linux: libiomp5.so via dlopen(RTLD_NOLOAD)); smf never loads it. When the
/// symbol is absent (SMF_USE_MKL off, OpenBLAS, sequential / TBB / GNU MKL,
/// LLVM libomp only) every function below is a no-op returning "unsupported".
/// smf's own OpenMP (LLVM libomp) is a different runtime and is not affected.

/// True when the Intel runtime's kmp_set_blocktime / kmp_get_blocktime are
/// available (resolved lazily; meaningful after the first MKL call).
bool mkl_blocktime_supported() noexcept;

/// This thread's blocktime in ms (kmp_get_blocktime), -1 when unsupported.
int mkl_blocktime_get() noexcept;

/// Set this thread's blocktime in ms (>= 0); returns the previous value, or
/// -1 when unsupported or ms < 0 (nothing changed).
int mkl_blocktime_set(int ms) noexcept;

/// Number of mkl_blocktime_set() calls that reached the runtime since process
/// start (tests / diagnostics).
long long mkl_blocktime_apply_count() noexcept;

/// Test seam: replace the runtime functions. Both null: behave as if the
/// symbol is absent. mkl_blocktime_reset_hooks_for_test() goes back to the
/// real runtime. Not thread-safe against concurrent guards.
void mkl_blocktime_set_hooks_for_test(void (*set_fn)(int), int (*get_fn)()) noexcept;
void mkl_blocktime_reset_hooks_for_test() noexcept;

/// Test seam for the runtime lookup itself: replaces the "is libiomp5 loaded"
/// probe with `resolver` (fills set_fn/get_fn, returns true on a hit) and sets
/// the miss-retry throttle (default ~100 ms). Clears the cached result. A miss
/// is never final: the lookup is retried, throttled; a hit is cached forever.
void mkl_blocktime_set_resolver_for_test(bool (*resolver)(void (**set_fn)(int), int (**get_fn)()),
                                         int retry_interval_ms) noexcept;
void mkl_blocktime_reset_resolver_for_test() noexcept;

/// RAII guard: sets BLAS thread count on construct, restores previous value on
/// destruct.  Non-copyable. MKL: the calling thread's local setting
/// (MKL_Set_Num_Threads_Local), restored to the previous local setting, so
/// a thread without one keeps following the global mkl_set_num_threads
/// value afterwards. OpenBLAS: process-wide, the previous count restored.
class BlasThreadGuard {
public:
    /// Construct: save current thread count, then set to \p n_threads.
    /// If \p n_threads <= 0 it is clamped to 1.
    explicit BlasThreadGuard(int n_threads) noexcept;

    /// As above, plus the Intel runtime blocktime of this thread
    /// (Control::mkl_blocktime_ms) for the guard's lifetime: blocktime_ms >= 0
    /// sets it and the destructor restores the previous value; negative leaves
    /// it alone. No-op when the runtime is absent (mkl_blocktime_supported()).
    BlasThreadGuard(int n_threads, int blocktime_ms) noexcept;

    /// Destruct: restore the previously saved thread count.
    ~BlasThreadGuard() noexcept;

    // Non-copyable, non-movable
    BlasThreadGuard(const BlasThreadGuard&)            = delete;
    BlasThreadGuard& operator=(const BlasThreadGuard&) = delete;
    BlasThreadGuard(BlasThreadGuard&&)                 = delete;
    BlasThreadGuard& operator=(BlasThreadGuard&&)      = delete;

    /// Return the thread count that was saved on construction.
    int saved_threads() const noexcept { return saved_threads_; }

private:
    int saved_threads_;
    int prev_setting_; // what the destructor restores (MKL: the previous local setting, 0 = none)
    int prev_counted_; // this thread's counted BLAS threads before the guard
    int prev_blocktime_ = -1; // blocktime the destructor restores (-1: not changed)
};

/// Accounting only (the BLAS setting is untouched): this thread counts no
/// BLAS threads while the scope lives, e.g. a pool worker between tasks
/// whose calling thread holds an outer guard. Guards nested inside count
/// as usual.
class BlasIdleScope {
public:
    BlasIdleScope() noexcept;
    ~BlasIdleScope() noexcept;
    BlasIdleScope(const BlasIdleScope&)            = delete;
    BlasIdleScope& operator=(const BlasIdleScope&) = delete;

private:
    int prev_counted_;
};

/// BLAS threads in use, as set by the live guards: each thread inside a
/// guard counts the count of its innermost guard (a thread outside every
/// guard counts 0). Process-wide, for tests and diagnostics.
int blas_threads_active() noexcept;
/// Highest blas_threads_active() since the last blas_threads_reset_peak().
int blas_threads_peak() noexcept;
void blas_threads_reset_peak() noexcept;

/// The BLAS thread count a BLAS call from this thread would get now (MKL:
/// mkl_get_max_threads, 1 with sequential MKL; OpenBLAS: its process-wide
/// count; 1 without a threading API).
/// For tests and diagnostics.
int blas_max_threads() noexcept;

/// Convenience alias: construct with 1 to enforce serial BLAS inside OpenMP
/// tasks.
struct BlasSerialGuard : BlasThreadGuard {
    BlasSerialGuard() noexcept : BlasThreadGuard(1) {}
};

/// The BLAS thread count a public factor or solve call holds on its calling
/// thread for the whole call (CallBlasGuard): 1 in deterministic mode, else
/// Control::num_threads when it is >= 1, else 0: num_threads <= 0 holds no
/// guard, so the call runs on the caller's own BLAS setting (one tree
/// thread; with threaded MKL every core by default).
int call_blas_threads(const Control &ctrl) noexcept;

/// RAII: BlasThreadGuard(call_blas_threads(ctrl)) for the lifetime of a
/// public factor or solve call (nothing at 0), so num_threads = 1 means one
/// BLAS thread on every path, the caller's BLAS setting (e.g. MKL's global
/// count, all cores by default) restored on return. At more than one thread
/// it also applies Control::mkl_blocktime_ms. Nested guards (per-front
/// shares, serial BLAS in subtrees and tiles) work inside it as before.
class CallBlasGuard {
public:
    explicit CallBlasGuard(const Control &ctrl) noexcept;
    CallBlasGuard(const CallBlasGuard&)            = delete;
    CallBlasGuard& operator=(const CallBlasGuard&) = delete;

private:
    std::optional<BlasThreadGuard> guard_;
};

} // namespace smf
