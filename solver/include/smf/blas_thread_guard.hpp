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
};

/// Convenience alias: construct with 1 to enforce serial BLAS inside OpenMP
/// tasks.
struct BlasSerialGuard : BlasThreadGuard {
    BlasSerialGuard() noexcept : BlasThreadGuard(1) {}
};

} // namespace smf
