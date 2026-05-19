// blas_thread_guard.cpp — implementation of BlasThreadGuard.
//
// Compile-time selection:
//   SMF_USE_MKL=1      → MKL thread-local API
//   SMF_HAS_OPENBLAS=1 → OpenBLAS global API
//   otherwise          → no-op

#include "smf/blas_thread_guard.hpp"

// --------------------------------------------------------------------------
// Backend selection
// --------------------------------------------------------------------------

#if defined(SMF_USE_MKL) && SMF_USE_MKL
#  include <mkl.h>
// Use thread-local MKL interface so that nested guards in different threads
// don't interfere with each other.
static int  blas_get_num_threads()        { return MKL_Get_Max_Threads(); }
static void blas_set_num_threads(int n)   { MKL_Set_Num_Threads(n); }
#  define SMF_BLAS_THREAD_API 1

#elif defined(SMF_HAS_OPENBLAS) && SMF_HAS_OPENBLAS
// The pthread variant of OpenBLAS ships openblas_get_num_threads /
// openblas_set_num_threads.  These are global (not per-thread), but they are
// the standard OpenBLAS control knobs.
#  include <cblas.h>
static int  blas_get_num_threads()        { return openblas_get_num_threads(); }
static void blas_set_num_threads(int n)   { openblas_set_num_threads(n); }
#  define SMF_BLAS_THREAD_API 1

#else
// No thread-count API available — guard is a no-op.
static int  blas_get_num_threads()        { return 1; }
static void blas_set_num_threads(int /*n*/) {}
#  define SMF_BLAS_THREAD_API 0
#endif

// --------------------------------------------------------------------------
// BlasThreadGuard implementation
// --------------------------------------------------------------------------

namespace smf {

BlasThreadGuard::BlasThreadGuard(int n_threads) noexcept
    : saved_threads_(blas_get_num_threads())
{
    // Treat 0 returned by BLAS (no threading built in) as 1.
    if (saved_threads_ <= 0) saved_threads_ = 1;
    const int requested = (n_threads > 0) ? n_threads : 1;
    blas_set_num_threads(requested);
}

BlasThreadGuard::~BlasThreadGuard() noexcept
{
    blas_set_num_threads(saved_threads_);
}

} // namespace smf
