#include "../benchmarks/common/validation.hpp"
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/info.hpp"
#include "smf/solver.hpp"
#include "smf/workspace.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <thread>
#include <vector>

using namespace smf;

namespace {

// Build a 4x4 lower-CSC matrix that deliberately includes unsorted and duplicate entries
// in its original pattern to verify that AnalysisKeep + NumericWorkspace remap them without
// mutating AnalysisKeep!
CscLower make_unsorted_duplicate_base() {
    CscLower A;
    A.n = 4;
    // col 0 has rows: 2, 0 (dup), 1, 0 (dup) -> 4 entries
    // col 1 has rows: 3, 1 -> 2 entries
    // col 2 has rows: 2, 3, 2 (dup) -> 3 entries
    // col 3 has rows: 3 -> 1 entry
    A.col_ptr = {0, 4, 6, 9, 10};
    A.row_idx = {
        2, 0, 1, 0,
        3, 1,
        2, 3, 2,
        3
    };
    A.values = {
        -0.5, 3.0, 1.0, 2.0,
        0.5,  4.0,
        -2.0, 1.0, -1.0,
        -5.0
    };
    return A;
}

} // namespace

TEST(NumericWorkspace, ChangedValuesSamePattern) {
    const CscLower base = make_unsorted_duplicate_base();
    const std::vector<double> orig_clean_vals_snapshot;

    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    ctrl.nemin = 1;
    ctrl.pivot_u = 0.01;

    Solver solver;
    Info info_analyse;
    const auto ak = solver.analyse(base, ctrl, info_analyse);
    ASSERT_NE(ak.get(), nullptr);
    ASSERT_EQ(info_analyse.status, ErrorCode::Success);

    // Record snapshot of ak->cleaned.values to prove AnalysisKeep is NEVER mutated
    const std::vector<double> cleaned_before = ak->cleaned.values;

    NumericWorkspace nws;
    SolveWorkspace sws;
    FactorKeep fk;

    bool saw_inertia_change = false;
    int first_neg = -1;

    // Replay 110 value updates with changing values and changing inertia
    for (int iter = 0; iter < 110; ++iter) {
        CscLower updated = base;
        const double sign_flip = (iter % 2 == 0) ? 1.0 : -1.0;
        const double scale = 1.0 + 0.02 * static_cast<double>(iter);
        // Vary diagonal and off-diagonal entries, flipping signs of diagonal blocks
        updated.values[0] = -0.5 * (1.0 + 0.01 * (iter % 7));
        updated.values[1] = 3.0 * scale * sign_flip;
        updated.values[2] = 1.0 + 0.03 * (iter % 5);
        updated.values[3] = 2.0 * scale * sign_flip;
        updated.values[4] = 0.5 - 0.01 * (iter % 9);
        updated.values[5] = (iter % 3 == 0 ? -4.0 : 4.0) * scale;
        updated.values[6] = -2.0 * scale;
        updated.values[7] = 1.0;
        updated.values[8] = -1.0 * scale;
        updated.values[9] = -5.0 * sign_flip * scale;

        Info info_f;
        FactorStatus fs = solver.factor(*ak, updated, ctrl, info_f, fk, nws);
        ASSERT_EQ(fs, FactorStatus::Success) << "Failed at iter=" << iter;
        EXPECT_EQ(ak->cleaned.values, cleaned_before)
            << "AnalysisKeep was mutated at iter=" << iter;

        if (iter == 0) {
            first_neg = info_f.num_negative;
        } else if (info_f.num_negative != first_neg) {
            saw_inertia_change = true;
        }

        std::vector<double> b = {1.0, -2.0, 3.0, -4.0, 0.5, 1.5, -2.5, 3.5};
        std::vector<double> x = b;
        Info info_s;
        ASSERT_EQ(solver.solve(fk, ctrl, info_s, x.data(), 4, 2, sws), 0)
            << "Solve failed at iter=" << iter;

        auto val = smf::bench::validate(updated, b, x, 2);
        EXPECT_TRUE(smf::bench::is_qualified_validation(val))
            << "iter=" << iter << " eta_inf=" << val.eta_inf << " omega=" << val.omega;
    }
    EXPECT_TRUE(saw_inertia_change);

    // PatternChanged must be rejected and never reuse stale symbolic maps!
    {
        CscLower bad_pattern = base;
        bad_pattern.row_idx[0] = 3; // changed row index from 2 to 3!
        Info bad_info;
        FactorKeep bad_fk;
        FactorStatus st = solver.factor(*ak, bad_pattern, ctrl, bad_info, bad_fk, nws);
        EXPECT_NE(st, FactorStatus::Success);
        EXPECT_EQ(bad_info.status, ErrorCode::IllegalValue);
    }
}

TEST(NumericWorkspace, ConcurrentSolve) {
    CscLower A = make_unsorted_duplicate_base();
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;

    Solver solver;
    Info info;
    auto ak = solver.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);

    NumericWorkspace nws;
    FactorKeep fk;
    ASSERT_EQ(solver.factor(*ak, A, ctrl, info, fk, nws), FactorStatus::Success);

    const FactorKeep &const_fk = fk;
    constexpr int kNumThreads = 4;
    constexpr int kItersPerThread = 25;
    std::vector<int> thread_ok(kNumThreads, 1);

    std::vector<std::thread> threads;
    for (int t = 0; t < kNumThreads; ++t) {
        threads.emplace_back([&, t]() {
            SolveWorkspace local_sws;
            for (int it = 0; it < kItersPerThread; ++it) {
                const double s = static_cast<double>((t + 1) * (it + 1));
                std::vector<double> b = {1.0 * s, -2.0 * s, 3.0 * s, 0.5 * s};
                std::vector<double> x = b;
                Info local_info;
                if (solver.solve(const_fk, ctrl, local_info, x.data(), 4, 1, local_sws) != 0) {
                    thread_ok[static_cast<std::size_t>(t)] = 0;
                    break;
                }
                auto val = smf::bench::validate(A, b, x, 1);
                if (!smf::bench::is_qualified_validation(val)) {
                    thread_ok[static_cast<std::size_t>(t)] = 0;
                    break;
                }
            }
        });
    }
    for (auto &th : threads) {
        th.join();
    }
    for (int t = 0; t < kNumThreads; ++t) {
        EXPECT_EQ(thread_ok[static_cast<std::size_t>(t)], 1) << "Race/failure on thread " << t;
    }
}

TEST(NumericWorkspace, StorageOverflow) {
    LongInt out = 0;
    EXPECT_TRUE(checked_add_i64(100, 200, out));
    EXPECT_EQ(out, 300);
    EXPECT_FALSE(checked_add_i64(std::numeric_limits<LongInt>::max(), 1, out));
    EXPECT_FALSE(checked_mul_i64(std::numeric_limits<LongInt>::max() / 2 + 2, 2, out));
    EXPECT_FALSE(checked_storage_elements(100000, 100000, out)); // 10^10 > kMaxFactorStorageEntries

    // Metadata-sized overflow test on AnalysisKeep without allocating multi-GB buffers:
    AnalysisKeep bogus_ak;
    bogus_ak.n = 100000;
    bogus_ak.cleaned.n = 100000;
    bogus_ak.cleaned.col_ptr.assign(100001, 0);
    bogus_ak.orig_col_ptr = bogus_ak.cleaned.col_ptr;
    bogus_ak.perm.resize(100000, 0);
    bogus_ak.iperm.resize(100000, 0);

    Supernode big_sn;
    big_sn.col_start = 0;
    big_sn.col_end = 50000;
    big_sn.parent = -1;
    bogus_ak.supernodes.push_back(big_sn);

    FrontalInfo big_fi;
    // Set row_indices size via metadata check: 50000 x 50000 = 2.5e9 > kMaxFactorStorageEntries
    big_fi.row_indices.resize(50000, 0);
    bogus_ak.fronts.push_back(big_fi);
    bogus_ak.postorder = {0};
    bogus_ak.solve_postorder = {0};

    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    Info info;
    FactorKeep fk;
    NumericWorkspace nws;
    Solver solver;

    FactorStatus st = solver.factor(bogus_ak, bogus_ak.cleaned, ctrl, info, fk, nws);
    EXPECT_NE(st, FactorStatus::Success);
    EXPECT_EQ(info.status, ErrorCode::AllocationError);
    EXPECT_TRUE(fk.factor_values.empty());
}

TEST(NumericWorkspace, WarmedNoPerFrontHeapAllocationAndGrowthReported) {
    // 1. Stable-pivot fixture: after warmup call, subsequent calls do 0 per-front heap allocations
    CscLower A;
    A.n = 3;
    A.col_ptr = {0, 2, 4, 5};
    A.row_idx = {0, 2, 1, 2, 2};
    A.values = {10.0, 1.0, 10.0, 1.0, 10.0};

    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    ctrl.nemin = 1;
    ctrl.pivot_u = 0.01;

    Solver solver;
    Info info;
    auto ak = solver.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);

    NumericWorkspace nws;
    FactorKeep fk;

    // Warmup call (iter 0)
    ASSERT_EQ(solver.factor(*ak, A, ctrl, info, fk, nws), FactorStatus::Success);

    // Warmed call (iter 1) with stable pivots
    CscLower A2 = A;
    A2.values = {12.0, 1.0, 11.0, 1.0, 9.0};
    ASSERT_EQ(solver.factor(*ak, A2, ctrl, info, fk, nws), FactorStatus::Success);
    EXPECT_EQ(nws.front_heap_allocations_last_call, 0);
    EXPECT_EQ(nws.capacity_growths_last_call, 0);

    // Now trigger a delayed pivot on the same pattern (A(0,0) = 0.0 delays col 0 into parent 2),
    // which expands parent 2's front from 1x1 to 2x2 and must report capacity growth!
    CscLower A_delayed = A;
    A_delayed.values = {0.0, 1.0, 10.0, 1.0, 2.0};
    ASSERT_EQ(solver.factor(*ak, A_delayed, ctrl, info, fk, nws), FactorStatus::Success);
    EXPECT_GT(info.delayed_pivots, 0);
    EXPECT_GT(nws.capacity_growths_last_call, 0);
}
