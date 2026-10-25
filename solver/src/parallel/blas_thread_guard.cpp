#include "smf/blas_thread_guard.hpp"

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
} // namespace

BlasThreadGuard::BlasThreadGuard(int n_threads) noexcept
    : saved_threads_(1)
{
    std::lock_guard<std::mutex> lock(blas_guard_mutex());
    saved_threads_ = blas_get_num_threads();
    if (saved_threads_ <= 0) saved_threads_ = 1;
    const int requested = (n_threads > 0) ? n_threads : 1;
    blas_set_num_threads(requested);
}

BlasThreadGuard::~BlasThreadGuard() noexcept
{
    std::lock_guard<std::mutex> lock(blas_guard_mutex());
    blas_set_num_threads(saved_threads_);
}

} // namespace smf
