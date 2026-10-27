#pragma once
// blas_thread_guard.hpp — RAII guard that sets BLAS thread count and restores
// on destruction.
//
// Compile-time selection:
//   SMF_USE_MKL=1      → MKL_Set_Num_Threads / MKL_Get_Max_Threads
//   SMF_HAS_OPENBLAS=1 → openblas_set_num_threads / openblas_get_num_threads
//   otherwise          → no-op (guard compiles but does nothing)
//
// Usage inside an OpenMP task (force serial BLAS):
//   smf::BlasThreadGuard _g(1);
//
// Usage for a large front (parallel BLAS):
//   smf::BlasThreadGuard _g(control.num_threads);

namespace smf {

/// RAII guard: sets BLAS thread count on construct, restores previous value on
/// destruct.  Non-copyable.
class BlasThreadGuard {
public:
    /// Construct: save current thread count, then set to \p n_threads.
    /// If \p n_threads <= 0 it is clamped to 1.
    explicit BlasThreadGuard(int n_threads) noexcept;

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
    int prev_counted_; // this thread's counted BLAS threads before the guard
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

/// Convenience alias: construct with 1 to enforce serial BLAS inside OpenMP
/// tasks.
struct BlasSerialGuard : BlasThreadGuard {
    BlasSerialGuard() noexcept : BlasThreadGuard(1) {}
};

} // namespace smf
