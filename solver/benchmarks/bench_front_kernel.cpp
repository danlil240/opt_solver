#include "smf/blocked_indef.hpp"
#include "smf/control.hpp"
#include "smf/numeric_front.hpp"
#include "smf/workspace.hpp"

#include <chrono>
#include <cstdio>
#include <numeric>
#include <vector>

using Clock = std::chrono::steady_clock;

int main() {
    std::puts("=== bench_front_kernel: reference vs blocked indefinite front ===");
    const int f = 96;
    const int p = 64;
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
    ctrl.indef_panel_size = 32;

    constexpr int kReps = 20;
    double ref_ms = 0.0;
    for (int r = 0; r < kReps; ++r) {
        smf::NumericFront f_ref = base_front;
        auto t0 = Clock::now();
        auto res = smf::factor_front_reference(f_ref, ctrl);
        auto t1 = Clock::now();
        if (res.status != smf::ErrorCode::Success) return 1;
        ref_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

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

    std::printf("Front (%dx%d): reference=%.3f ms  blocked(nb=32)=%.3f ms  speedup=%.2fx\n",
                f, p, ref_ms / kReps, blk_ms / kReps,
                blk_ms > 0.0 ? (ref_ms / blk_ms) : 0.0);
    return 0;
}
