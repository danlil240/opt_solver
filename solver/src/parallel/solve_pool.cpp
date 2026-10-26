#include "smf/solve_pool.hpp"

#include "smf/blas_thread_guard.hpp"

#include <chrono>

namespace smf {

namespace {
// Poll this long before a worker or the caller blocks.
constexpr auto kSpin = std::chrono::microseconds(50);

inline void cpu_relax() { solve_pool_relax(); }

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
    for (int t = 1; t <= extra; ++t)
        workers_.emplace_back(&SolvePool::worker, this, t);
}

SolvePool::~SolvePool() {
    {
        std::lock_guard<std::mutex> lk(mtx_);
        stop_ = true;
        generation_.fetch_add(1, std::memory_order_release);
    }
    wake_.notify_all();
    for (auto &w : workers_) w.join();
}

void SolvePool::run(const std::function<void(int)> &fn) {
    if (workers_.empty()) {
        fn(0);
        return;
    }
    {
        std::lock_guard<std::mutex> lk(mtx_);
        job_ = &fn;
        pending_.store(static_cast<int>(workers_.size()), std::memory_order_relaxed);
        generation_.fetch_add(1, std::memory_order_release);
    }
    wake_.notify_all();
    fn(0);
    if (spin_until([&] { return pending_.load(std::memory_order_acquire) == 0; })) return;
    std::unique_lock<std::mutex> lk(mtx_);
    done_.wait(lk, [&] { return pending_.load(std::memory_order_acquire) == 0; });
}

void SolvePool::worker(int tid) {
    // Kernels inside a parallel solve run on serial BLAS.
    BlasSerialGuard serial_blas;
    unsigned seen = 0;
    for (;;) {
        const bool fresh =
            spin_until([&] { return generation_.load(std::memory_order_acquire) != seen; });
        const std::function<void(int)> *job = nullptr;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            if (!fresh)
                wake_.wait(lk, [&] { return generation_.load(std::memory_order_acquire) != seen; });
            if (stop_) return;
            seen = generation_.load(std::memory_order_acquire);
            job = job_;
        }
        (*job)(tid);
        if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard<std::mutex> lk(mtx_);
            done_.notify_one();
        }
    }
}

} // namespace smf
