// determinism.hpp — Deterministic parallel mode.
//
// With Control::deterministic = true, or a build with SMF_DETERMINISTIC=ON
// (either enables it), analyse + factor + solve give bitwise-identical
// orderings, factors, inertia and solutions across runs and across every
// Control::num_threads:
//   - ordering: AutoParallel's METIS candidate is parallel nested dissection
//     for every thread count (its permutation does not depend on the thread
//     count), not serial METIS at 1 thread;
//   - factor: each front is assembled from its children in the fixed
//     assembly-tree order (never completion order), every BLAS call runs on
//     one thread, and the plain threshold fronts of >= 512 rows are factored
//     as 2D tile tasks wherever they sit in the tree and whatever num_threads
//     is (the tile kernel's result does not depend on its thread count); the
//     top of the tree is laid out in postorder, not completion order;
//   - solve: one thread, fronts in postorder;
//   - MKL builds: conditional numerical reproducibility,
//     mkl_cbwr_set(MKL_CBWR_AVX2), once at first use (process-wide); on CPUs
//     where MKL rejects that branch (non-Intel), MKL_CBWR_AUTO.
// Default off.
#pragma once
#include "smf/control.hpp"

#include <algorithm>
#include <vector>

namespace smf {

/// True when deterministic mode is on: Control::deterministic, or always in
/// an SMF_DETERMINISTIC build.
inline bool deterministic_mode(const Control& ctrl) noexcept {
#if defined(SMF_DETERMINISTIC) && SMF_DETERMINISTIC
    (void)ctrl;
    return true;
#else
    return ctrl.deterministic;
#endif
}

/// Deterministic mode, MKL builds: turns on MKL's conditional numerical
/// reproducibility the first time it is called in the process
/// (mkl_cbwr_set(MKL_CBWR_AVX2), or MKL_CBWR_AUTO where MKL does not
/// support the AVX2 branch); later calls do nothing. No-op without MKL.
void enable_blas_reproducibility() noexcept;

/// The mkl_cbwr_set result of the mode in effect (MKL_CBWR_SUCCESS = 0);
/// 1 before any enable_blas_reproducibility() call or without MKL.
int blas_reproducibility_status() noexcept;

/// The CNR branch in effect (MKL_CBWR_AVX2 = 10, MKL_CBWR_AUTO = 2), 0 if
/// none.
int blas_reproducibility_branch() noexcept;

/// Sort child node IDs in ascending order for deterministic task spawning.
void sort_children_deterministic(std::vector<int>& children);

/// Deterministic reduction: sum contribution vectors in a fixed index order.
/// This avoids floating-point non-associativity from variable task completion order.
void deterministic_reduce(std::vector<double>& accumulator,
                           const std::vector<std::vector<double>>& contributions,
                           int size);

} // namespace smf
