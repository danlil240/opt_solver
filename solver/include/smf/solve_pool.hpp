#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace smf {

/// One spin-wait step: a pause instruction, not a scheduler call.
inline void solve_pool_relax() {
#if defined(_M_X64) || defined(__x86_64__)
    _mm_pause();
#else
    std::this_thread::yield();
#endif
}

/// Persistent worker threads for the parallel triangular solves. A solve
/// takes milliseconds, so starting threads per call (tens of microseconds
/// each on Windows) would eat the gain; workers spin briefly after a job
/// before sleeping, since IPOPT-style callers solve several times in a row.
/// std::thread, not OpenMP, for the same reason as the factorization.
class SolvePool {
public:
    explicit SolvePool(int threads);
    ~SolvePool();
    SolvePool(const SolvePool &) = delete;
    SolvePool &operator=(const SolvePool &) = delete;

    int size() const { return static_cast<int>(workers_.size()) + 1; }

    /// Runs fn(tid) for tid = 0 .. size() - 1, tid 0 on the calling thread,
    /// and returns when every call has finished.
    void run(const std::function<void(int)> &fn);

private:
    void worker(int tid);

    std::vector<std::thread> workers_;
    std::mutex mtx_;
    std::condition_variable wake_;
    std::condition_variable done_;
    const std::function<void(int)> *job_ = nullptr;
    std::atomic<unsigned> generation_{0};
    std::atomic<int> pending_{0};
    bool stop_ = false;
};

/// Runs `count` DAG nodes on the pool, starting from `ready`:
/// process(i, tid, release) handles node i, calls release(j) for successors
/// it makes ready, and returns one more ready successor to run next on the
/// same thread (or -1). Continuing along a chain keeps its data in one
/// core's cache and keeps tiny fronts off the shared queue. Idle threads
/// spin (a solve's DAG is a few hundred fronts over milliseconds).
template <class Process>
void run_dag(SolvePool &pool, int count, std::vector<int> ready, Process process) {
    std::mutex mtx;
    std::atomic<int> remaining{count};
    // Idle threads poll this count, not the mutex, so they do not contend
    // with the threads releasing work.
    std::atomic<int> nready{static_cast<int>(ready.size())};
    pool.run([&](int tid) {
        const auto release = [&](int j) {
            std::lock_guard<std::mutex> lk(mtx);
            ready.push_back(j);
            nready.fetch_add(1, std::memory_order_release);
        };
        for (;;) {
            if (nready.load(std::memory_order_acquire) == 0) {
                if (remaining.load(std::memory_order_acquire) == 0) return;
                solve_pool_relax();
                continue;
            }
            int i = -1;
            {
                std::lock_guard<std::mutex> lk(mtx);
                if (!ready.empty()) {
                    i = ready.back();
                    ready.pop_back();
                    nready.fetch_sub(1, std::memory_order_relaxed);
                }
            }
            while (i >= 0) {
                const int next = process(i, tid, release);
                remaining.fetch_sub(1, std::memory_order_acq_rel);
                i = next;
            }
        }
    });
}

} // namespace smf
