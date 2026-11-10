#pragma once
// Contiguous-range loops on the factorization's worker pool (NumericWorkspace::
// ensure_pool), for the passes around the tree that touch every matrix entry
// (Control::parallel_value_prep). The work stays deterministic: a range is a
// fixed slice of [0, n), and fn decides what each slice writes.

#include "smf/blas_thread_guard.hpp"
#include "smf/solve_pool.hpp"
#include "smf/types.hpp"
#include "smf/workspace.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <span>
#include <vector>

namespace smf {

// fn(tid, begin, end) over [0, n) in `threads` contiguous ranges (range t is
// [n t / threads, n (t + 1) / threads)) on ws's pool. threads <= 1 or n <
// min_n: fn(0, 0, n) on the caller.
template <class F>
void parallel_ranges(NumericWorkspace &ws, int threads, std::size_t n, std::size_t min_n, F &&fn) {
  if (threads <= 1 || n < min_n || n < static_cast<std::size_t>(threads)) {
    fn(0, std::size_t{0}, n);
    return;
  }
  const std::size_t T = static_cast<std::size_t>(threads);
  const auto job_fn = [&](int tid) {
    const std::size_t t = static_cast<std::size_t>(tid);
    fn(tid, n * t / T, n * (t + 1) / T);
  };
  BlasIdleScope idle;
  const std::function<void(int)> job(std::cref(job_fn));
  ws.ensure_pool(threads).run(job, /*workers_count_blas=*/false);
}

// Column boundaries splitting a CSC pattern's columns into `parts` ranges of
// about equal entry counts: bounds[t] .. bounds[t + 1] (bounds.size() ==
// parts + 1, bounds[0] = 0, bounds[parts] = n).
inline std::vector<Int> balanced_columns(std::span<const Int> col_ptr, Int n, int parts) {
  std::vector<Int> bounds(static_cast<std::size_t>(parts) + 1, n);
  bounds[0] = 0;
  const double nnz = static_cast<double>(col_ptr[static_cast<std::size_t>(n)]);
  for (int t = 1; t < parts; ++t) {
    const Int target = static_cast<Int>(nnz * t / parts);
    const auto it = std::lower_bound(col_ptr.begin(), col_ptr.begin() + n + 1, target);
    bounds[static_cast<std::size_t>(t)] =
        std::max(bounds[static_cast<std::size_t>(t) - 1], static_cast<Int>(it - col_ptr.begin()));
  }
  return bounds;
}

}  // namespace smf
