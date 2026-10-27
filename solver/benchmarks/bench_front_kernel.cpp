#include "smf/blocked_indef.hpp"
#include "smf/control.hpp"
#include "smf/ldlt_aptp.hpp"
#include "smf/ldlt_kernel.hpp"
#include "smf/numeric_front.hpp"
#include "smf/workspace.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

namespace {

// --aptp: ldlt_aptp_front vs ldlt_factor_front on dense fronts (run with
// MKL_NUM_THREADS=1 / OPENBLAS_NUM_THREADS=1 for single-thread numbers).
std::vector<double> make_front(int f, bool kkt, std::mt19937& rng) {
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    std::vector<double> A(static_cast<std::size_t>(f) * f, 0.0);
    auto at = [&](std::vector<double>& M, int i, int j) -> double& {
        return M[static_cast<std::size_t>(j) * f + i];
    };
    if (!kkt) {
        for (int j = 0; j < f; ++j)
            for (int i = j; i < f; ++i) at(A, i, j) = U(rng);
        return A;
    }
    // [[H, J^T], [J, 0]], H diagonally weighted, J dense, symmetrically
    // permuted at random (primal and dual columns interleaved as in a front).
    const int np = (2 * f) / 3;
    std::vector<double> K(static_cast<std::size_t>(f) * f, 0.0);
    for (int j = 0; j < np; ++j)
        for (int i = j; i < np; ++i)
            at(K, i, j) = (i == j) ? 1.0 + 9.0 * (0.5 + 0.5 * U(rng)) : 0.1 * U(rng);
    for (int j = 0; j < np; ++j)
        for (int i = np; i < f; ++i) at(K, i, j) = U(rng);
    std::vector<int> perm(f);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    for (int j = 0; j < f; ++j)
        for (int i = j; i < f; ++i) {
            const int a = std::max(perm[i], perm[j]), b = std::min(perm[i], perm[j]);
            at(A, i, j) = at(K, a, b);
        }
    return A;
}

int run_aptp_bench(int only_f) {
    std::puts("=== bench_front_kernel --aptp: APTP vs ldlt_factor_front (single thread) ===");
    std::puts("  type    f     elig | kernel  elim  fail      ms    GFLOP/s");
    std::mt19937 rng(7);
    for (const bool kkt : {false, true}) {
        for (const int f : {128, 512, 2048}) {
            if (only_f > 0 && f != only_f) continue;
            for (const int elig : {f, f / 2}) {
                const std::vector<double> A = make_front(f, kkt, rng);
                const int reps = f <= 128 ? 400 : (f <= 512 ? 20 : 3);
                std::vector<double> F(A.size());
                std::vector<smf::Int> rows(f);
                std::vector<int8_t> tags(f);

                smf::LdltKernelParams kp;
                std::vector<double> wk(smf::ldlt_work_size(f, kp));
                smf::AptpParams ap;
                std::vector<double> wa(smf::aptp_work_size(f, elig, ap));

                for (int which = 0; which < 2; ++which) {
                    double best = 1e300;
                    double flops = 0.0;
                    int elim = 0;
                    for (int r = 0; r < reps + 1; ++r) {
                        F = A;
                        std::iota(rows.begin(), rows.end(), 0);
                        const auto t0 = Clock::now();
                        if (which == 0) {
                            const auto res = smf::ldlt_factor_front(F.data(), f, elig, rows.data(), kp,
                                                                    tags.data(), wk.data());
                            flops = res.flops;
                            elim = res.num_accepted;
                        } else {
                            const auto res = smf::ldlt_aptp_front(F.data(), f, elig, rows.data(), ap,
                                                                  tags.data(), wa.data());
                            flops = res.flops;
                            elim = res.num_eliminated;
                        }
                        const auto t1 = Clock::now();
                        if (r > 0)
                            best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
                    }
                    std::printf("  %-6s %5d %5d | %-6s %5d %5d %9.3f %9.2f\n", kkt ? "kkt" : "random", f,
                                elig, which == 0 ? "tpp" : "aptp", elim, elig - elim, best * 1e3,
                                flops / best * 1e-9);
                }
            }
        }
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--aptp") == 0) return run_aptp_bench(argc > 2 ? std::atoi(argv[2]) : 0);
    std::puts("=== bench_front_kernel: reference vs blocked indefinite front ===");
    const int f = (argc > 1) ? std::atoi(argv[1]) : 96;
    const int p = (argc > 2) ? std::atoi(argv[2]) : 64;
    std::vector<smf::Int> rows(f);
    std::iota(rows.begin(), rows.end(), 0);

    smf::NumericFront base_front(rows, p);
    for (int j = 0; j < f; ++j) {
        for (int i = j; i < f; ++i) {
            if (i == j) {
                base_front.at(i, j) = (i % 5 == 0) ? 0.0 : ((i % 2 == 0) ? 20.0 : -18.0);
            } else if (i == j + 1 && (j % 5 == 0)) {
                base_front.at(i, j) = 5.0;
            } else {
                base_front.at(i, j) = 0.05 / static_cast<double>(i - j + 1);
            }
        }
    }

    smf::Control ctrl;
    ctrl.pivot_u = 0.01;

    constexpr int kReps = 50;
    double ref_ms = 0.0;
    for (int r = 0; r < kReps; ++r) {
        smf::NumericFront f_ref = base_front;
        auto t0 = Clock::now();
        auto res = smf::factor_front_reference(f_ref, ctrl);
        auto t1 = Clock::now();
        if (res.status != smf::ErrorCode::Success) return 1;
        ref_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    const double ref_avg = ref_ms / kReps;

    for (const int nb : {4, 8, 16, 32, 64}) {
        ctrl.indef_panel_size = nb;
        smf::NumericWorkspace ws;
        double blk_ms = 0.0;
        for (int r = 0; r < kReps; ++r) {
            smf::NumericFront f_blk = base_front;
            auto t0 = Clock::now();
            auto res = smf::factor_front_blocked(f_blk, ctrl, ws);
            auto t1 = Clock::now();
            if (res.status != smf::ErrorCode::Success) return 1;
            blk_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        const double avg = blk_ms / kReps;
        std::printf("Front (%dx%d): reference=%.3f ms  blocked(nb=%d)=%.3f ms  speedup=%.2fx\n",
                    f, p, ref_avg, nb, avg,
                    avg > 0.0 ? (ref_avg / avg) : 0.0);
    }
    return 0;
}
