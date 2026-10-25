#include "smf/blocked_indef.hpp"
#include "smf/control.hpp"
#include "smf/numeric_front.hpp"
#include "smf/workspace.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
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
