// A warm refactorization (same analysis, NumericWorkspace and FactorKeep,
// new values) allocates no buffer of the matrix's size: per-call copies of
// that size are page-faulted in and freed on every factorization, a fixed
// cost that grew with the thread count in the benchmarks (Task 6b4).
//
// The global operator new is replaced in this executable (smf links
// statically) to count, while armed, the allocations of at least a given
// size.

#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/info.hpp"
#include "smf/solver.hpp"
#include "smf/workspace.hpp"
#include "kkt_test_matrices.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <new>
#include <span>
#include <vector>

namespace {

std::atomic<bool> g_armed{false};
std::atomic<std::size_t> g_min_bytes{0};
std::atomic<int> g_big{0};

void note(std::size_t bytes) {
    if (g_armed.load(std::memory_order_relaxed) &&
        bytes >= g_min_bytes.load(std::memory_order_relaxed))
        g_big.fetch_add(1, std::memory_order_relaxed);
}

void *counted_alloc(std::size_t bytes) {
    note(bytes);
    if (void *p = std::malloc(bytes != 0 ? bytes : 1)) return p;
    throw std::bad_alloc();
}

} // namespace

void *operator new(std::size_t bytes) { return counted_alloc(bytes); }
void *operator new[](std::size_t bytes) { return counted_alloc(bytes); }
void operator delete(void *p) noexcept { std::free(p); }
void operator delete[](void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }
void operator delete[](void *p, std::size_t) noexcept { std::free(p); }

using namespace smf;

namespace {

// Banded symmetric indefinite matrix (lower CSC): diagonal +-20 alternating,
// `band` sub-diagonals of small entries. No delays at u = 0.01.
CscLower banded_indefinite(Int n, Int band) {
    CscLower A;
    A.n = n;
    A.col_ptr.push_back(0);
    for (Int j = 0; j < n; ++j) {
        for (Int i = j; i <= std::min<Int>(n - 1, j + band); ++i) {
            A.row_idx.push_back(i);
            A.values.push_back(i == j ? ((j % 2) != 0 ? -20.0 : 20.0)
                                      : 1.0 / static_cast<double>(i - j + 1));
        }
        A.col_ptr.push_back(static_cast<Int>(A.row_idx.size()));
    }
    return A;
}

// Big allocations of the second (warm) factorization, at least `min_bytes`.
// `matching_scale`: the analysis must have kept a matching scale (the
// premise of ScalingMethod::Auto's user-supplied path).
int big_allocations_of_refactor(const CscLower &A, Control ctrl, std::size_t min_bytes,
                                bool matching_scale = false) {
    Solver solver;
    Info info;
    const auto ak = solver.analyse(A, ctrl, info);
    EXPECT_NE(ak.get(), nullptr);
    if (!ak) return -1;
    if (matching_scale) {
        EXPECT_FALSE(ak->matching_scale.empty()) << "premise: the analysis kept a matching scale";
        if (ak->matching_scale.empty()) return -1;
    }
    FactorKeep fk;
    NumericWorkspace ws;
    std::vector<double> v = A.values;
    EXPECT_EQ(solver.factor(*ak, std::span<const double>(v), ctrl, info, fk, ws),
              FactorStatus::Success);
    for (double &x : v) x *= 1.25;  // new values, same pattern
    g_big.store(0);
    g_min_bytes.store(min_bytes);
    g_armed.store(true);
    const FactorStatus st = solver.factor(*ak, std::span<const double>(v), ctrl, info, fk, ws);
    g_armed.store(false);
    EXPECT_EQ(st, FactorStatus::Success);
    return g_big.load();
}

} // namespace

TEST(RefactorAllocations, CounterSeesBigAllocations) {
    g_big.store(0);
    g_min_bytes.store(1000);
    g_armed.store(true);
    std::vector<double> big(1000);
    g_armed.store(false);
    EXPECT_EQ(g_big.load(), 1);
}

// User-supplied scaling: the scale is read, the cleaned matrix is not
// copied. At 1 and 16 threads (the tree is light: serial engine at both).
TEST(RefactorAllocations, UserScaleCopiesNoMatrix) {
    const CscLower A = banded_indefinite(2000, 8);
    const std::size_t nnz = A.row_idx.size();
    std::vector<double> scale(static_cast<std::size_t>(A.n));
    for (std::size_t i = 0; i < scale.size(); ++i) scale[i] = 0.5 + 0.25 * static_cast<double>(i % 3);
    for (const int threads : {1, 16}) {
        Control ctrl;
        ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
        ctrl.pivot_u = 0.01;
        ctrl.scaling = ScalingMethod::UserSupplied;
        ctrl.user_scale = scale.data();
        ctrl.num_threads = threads;
        EXPECT_EQ(big_allocations_of_refactor(A, ctrl, nnz * sizeof(double)), 0)
            << "threads " << threads;
    }
}

// ScalingMethod::Auto applies the analysis' matching scale through the
// user-supplied path, so it copies no matrix either: a saddle-point KKT whose
// auto ordering kept the matching (and its scale), refactored with new
// values.
TEST(RefactorAllocations, AutoMatchingScaleCopiesNoMatrix) {
    const CscLower A = opf_like_kkt(10000, 1000, 7);
    const std::size_t nnz = A.row_idx.size();
    for (const int threads : {1, 16}) {
        Control ctrl;
        ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
        ctrl.ordering = OrderingMethod::AutoSerial;
        ctrl.scaling = ScalingMethod::Auto;
        ctrl.num_threads = threads;
        EXPECT_EQ(big_allocations_of_refactor(A, ctrl, nnz * sizeof(double), true), 0)
            << "threads " << threads;
    }
}
