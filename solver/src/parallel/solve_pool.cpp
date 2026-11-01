#include "smf/solve_pool.hpp"

#include "smf/blas_thread_guard.hpp"

#include <chrono>
#include <exception>

namespace smf {

namespace {
// Poll this long before a worker or the caller blocks.
constexpr auto kSpin = std::chrono::microseconds(50);

inline void cpu_relax() { solve_pool_relax(); }

std::atomic<long long> g_threads_started{0};

// Runs one thread's share of a job. An exception cannot propagate: the other
// threads still run the job on the caller's stack (and wait on its mutexes),
// which unwinding would destroy. Fail fast, as an exception escaping a plain
// std::thread did before the pool.
inline void run_share(const std::function<void(int)> &job, int tid) noexcept {
    try {
        job(tid);
    } catch (...) {
        std::terminate();
    }
}

// Spins until ready() or kSpin elapses; true when ready() held.
template <class Pred> bool spin_until(Pred ready) {
    const auto t0 = std::chrono::steady_clock::now();
    for (unsigned i = 1;; ++i) {
        if (ready()) return true;
        cpu_relax();
        if ((i & 255u) == 0 && std::chrono::steady_clock::now() - t0 > kSpin) return ready();
    }
}
} // namespace

SolvePool::SolvePool(int threads) {
    const int extra = threads > 1 ? threads - 1 : 0;
    workers_.reserve(static_cast<std::size_t>(extra));
    for (int t = 1; t <= extra; ++t) {
        workers_.emplace_back(&SolvePool::worker, this, t);
        g_threads_started.fetch_add(1, std::memory_order_relaxed);
    }
}

long long SolvePool::threads_started() { return g_threads_started.load(std::memory_order_relaxed); }

SolvePool::~SolvePool() {
    {
        std::lock_guard<std::mutex> lk(mtx_);
        stop_ = true;
        generation_.fetch_add(1, std::memory_order_release);
    }
    wake_.notify_all();
    for (auto &w : workers_) w.join();
}

void SolvePool::run(const std::function<void(int)> &fn, bool workers_count_blas) {
    if (workers_.empty()) {
        fn(0);
        return;
    }
    {
        std::lock_guard<std::mutex> lk(mtx_);
        job_ = &fn;
        job_counts_blas_ = workers_count_blas;
        pending_.store(static_cast<int>(workers_.size()), std::memory_order_relaxed);
        generation_.fetch_add(1, std::memory_order_release);
    }
    wake_.notify_all();
    run_share(fn, 0);
    if (spin_until([&] { return pending_.load(std::memory_order_acquire) == 0; })) return;
    std::unique_lock<std::mutex> lk(mtx_);
    done_.wait(lk, [&] { return pending_.load(std::memory_order_acquire) == 0; });
}

void SolvePool::worker(int tid) {
    // Kernels inside a parallel solve run on serial BLAS.
    BlasSerialGuard serial_blas;
    unsigned seen = 0;
    const auto finish = [&] {
        if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard<std::mutex> lk(mtx_);
            done_.notify_one();
        }
    };
    for (;;) {
        const std::function<void(int)> *job = nullptr;
        {
            // Waiting for a job counts no BLAS threads; a job counts what its
            // own guards set (1 from the guard above by default). A job that
            // sets its own guards runs in this scope, and the worker stays in
            // it until a job that does not arrives: leaving it counts the
            // worker's thread, which must not happen while the caller may be
            // using BLAS threads.
            BlasIdleScope idle;
            for (;;) {
                const bool fresh =
                    spin_until([&] { return generation_.load(std::memory_order_acquire) != seen; });
                bool counts_blas;
                {
                    std::unique_lock<std::mutex> lk(mtx_);
                    if (!fresh)
                        wake_.wait(lk, [&] { return generation_.load(std::memory_order_acquire) != seen; });
                    if (stop_) return;
                    seen = generation_.load(std::memory_order_acquire);
                    job = job_;
                    counts_blas = job_counts_blas_;
                }
                if (counts_blas) break;
                run_share(*job, tid);
                finish();
            }
        }
        run_share(*job, tid);
        finish();
    }
}

} // namespace smf
