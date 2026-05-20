#include "smf/factor_indef.hpp"
#include "smf/frontal_matrix.hpp"
#include "smf/pivoting.hpp"
#include "smf/threading.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <numeric>
#include <vector>

namespace smf {

// ---------------------------------------------------------------------------
// Permute VALUES only using pre-computed pattern (col_ptr, row_idx) from analysis.
// This is a fast path that avoids re-allocating and re-sorting the pattern.
// ---------------------------------------------------------------------------
static std::vector<double> permute_values_only_indef(
    const CscLower &A_orig,
    const std::vector<Int> &iperm,
    const std::vector<Int> &perm_col_ptr,
    const std::vector<Int> &perm_row_idx) {
  
  const Int n = A_orig.n;
  const Int nnz_perm = perm_col_ptr[static_cast<std::size_t>(n)];
  std::vector<double> perm_values(static_cast<std::size_t>(nnz_perm), 0.0);

  // Scatter values from A_orig into permuted locations
  for (Int old_j = 0; old_j < n; ++old_j) {
    const Int new_j = iperm[static_cast<std::size_t>(old_j)];
    for (Int k = A_orig.col_ptr[static_cast<std::size_t>(old_j)];
         k < A_orig.col_ptr[static_cast<std::size_t>(old_j) + 1]; ++k) {
      const Int old_i = A_orig.row_idx[static_cast<std::size_t>(k)];
      const Int new_i = iperm[static_cast<std::size_t>(old_i)];
      Int col, row;
      if (new_i >= new_j) {
        col = new_j;
        row = new_i;
      } else {
        col = new_i;
        row = new_j;
      }
      const double val = A_orig.values[static_cast<std::size_t>(k)];
      
      // Find the slot in permuted column 'col' where row == 'row'
      const Int col_start = perm_col_ptr[static_cast<std::size_t>(col)];
      const Int col_end = perm_col_ptr[static_cast<std::size_t>(col) + 1];
      
      // Linear search for the matching row index
      for (Int p = col_start; p < col_end; ++p) {
        if (perm_row_idx[static_cast<std::size_t>(p)] == row) {
          perm_values[static_cast<std::size_t>(p)] += val;
          break;
        }
      }
    }
  }

  return perm_values;
}

// ---------------------------------------------------------------------------
// Permute a lower-CSC matrix into the analysis ordering (LEGACY, kept for reference).
// Identical logic to factor_posdef.cpp — scatter_original expects column/row
// indices in the permuted space, so we must pass A_perm, not the raw input.
// ---------------------------------------------------------------------------
static CscLower permute_lower_csc_indef(const CscLower &A,
                                        const std::vector<Int> &iperm) {
  const Int n = A.n;

  std::vector<Int> count(static_cast<std::size_t>(n), 0);
  for (Int old_j = 0; old_j < n; ++old_j) {
    const Int new_j = iperm[static_cast<std::size_t>(old_j)];
    for (Int k = A.col_ptr[static_cast<std::size_t>(old_j)];
         k < A.col_ptr[static_cast<std::size_t>(old_j) + 1]; ++k) {
      const Int old_i = A.row_idx[static_cast<std::size_t>(k)];
      const Int new_i = iperm[static_cast<std::size_t>(old_i)];
      if (new_i >= new_j)
        ++count[static_cast<std::size_t>(new_j)];
      else
        ++count[static_cast<std::size_t>(new_i)];
    }
  }

  CscLower B;
  B.n = n;
  B.col_ptr.resize(static_cast<std::size_t>(n) + 1, 0);
  for (Int j = 0; j < n; ++j)
    B.col_ptr[static_cast<std::size_t>(j) + 1] =
        B.col_ptr[static_cast<std::size_t>(j)] +
        count[static_cast<std::size_t>(j)];
  const Int nnz = B.col_ptr[static_cast<std::size_t>(n)];
  B.row_idx.resize(static_cast<std::size_t>(nnz));
  B.values.resize(static_cast<std::size_t>(nnz), 0.0);

  std::vector<Int> pos(B.col_ptr.begin(),
                       B.col_ptr.begin() + static_cast<std::ptrdiff_t>(n));

  for (Int old_j = 0; old_j < n; ++old_j) {
    const Int new_j = iperm[static_cast<std::size_t>(old_j)];
    for (Int k = A.col_ptr[static_cast<std::size_t>(old_j)];
         k < A.col_ptr[static_cast<std::size_t>(old_j) + 1]; ++k) {
      const Int old_i = A.row_idx[static_cast<std::size_t>(k)];
      const Int new_i = iperm[static_cast<std::size_t>(old_i)];
      Int col, row;
      if (new_i >= new_j) {
        col = new_j;
        row = new_i;
      } else {
        col = new_i;
        row = new_j;
      }
      const Int slot = pos[static_cast<std::size_t>(col)]++;
      B.row_idx[static_cast<std::size_t>(slot)] = row;
      B.values[static_cast<std::size_t>(slot)] =
          A.values[static_cast<std::size_t>(k)];
    }
  }

  // Sort each column's row indices (and corresponding values)
  for (Int j = 0; j < n; ++j) {
    const Int start = B.col_ptr[static_cast<std::size_t>(j)];
    const Int end   = B.col_ptr[static_cast<std::size_t>(j) + 1];
    const Int len   = end - start;
    if (len <= 1)
      continue;

    std::vector<Int> idx(static_cast<std::size_t>(len));
    std::iota(idx.begin(), idx.end(), Int{0});
    std::sort(idx.begin(), idx.end(), [&](Int a, Int b) {
      return B.row_idx[static_cast<std::size_t>(start + a)] <
             B.row_idx[static_cast<std::size_t>(start + b)];
    });

    std::vector<Int> sr(static_cast<std::size_t>(len));
    std::vector<double> sv(static_cast<std::size_t>(len));
    for (Int k = 0; k < len; ++k) {
      sr[static_cast<std::size_t>(k)] =
          B.row_idx[static_cast<std::size_t>(start + idx[static_cast<std::size_t>(k)])];
      sv[static_cast<std::size_t>(k)] =
          B.values[static_cast<std::size_t>(start + idx[static_cast<std::size_t>(k)])];
    }
    for (Int k = 0; k < len; ++k) {
      B.row_idx[static_cast<std::size_t>(start + k)] = sr[static_cast<std::size_t>(k)];
      B.values[static_cast<std::size_t>(start + k)]  = sv[static_cast<std::size_t>(k)];
    }
  }

  return B;
}

// ---------------------------------------------------------------------------
// Parallel indefinite factorization helper
// ---------------------------------------------------------------------------
#ifdef SMF_PARALLEL
static FactorStatus
factor_indef_parallel(const AnalysisKeep &keep, const CscLower &Ap,
                      const Control &ctrl,
                      FactorKeep &fkeep, InertiaCounts &out_inertia,
                      int &out_delayed) {
  const Int ns = static_cast<Int>(keep.supernodes.size());

  // Per-node contribution blocks (already heap-allocated, ideal for tasks).
  std::vector<std::vector<double>> contrib(static_cast<std::size_t>(ns));

  // Per-node inertia and delayed counts — accumulated serially after the
  // parallel region to avoid atomic ops on the hot path.
  std::vector<InertiaCounts> node_inertia(static_cast<std::size_t>(ns));
  std::vector<int> node_delayed(static_cast<std::size_t>(ns), 0);

  // Shared atomic error flag (0 = ok, 1 = singular).
  std::atomic<int> err_flag{0};

  auto process_node = [&](Int s) {
    const std::size_t si = static_cast<std::size_t>(s);
    const Supernode &sn = keep.supernodes[si];
    const FrontalInfo &fi = keep.fronts[si];
    const Int p = sn.width();
    const Int f = fi.front_size();
    const Int ext = f - p;

    if (p == 0)
      return;

    // Per-task arena.
    AlignedArena arena(static_cast<std::size_t>(8) << 20); // 8 MiB per task
    AlignedArena::Marker mark = arena.save();

    std::vector<Int> col_idx(fi.row_indices.cbegin(),
                             fi.row_indices.cbegin() +
                                 static_cast<std::ptrdiff_t>(p));
    FrontalMatrix front(f, p, fi.row_indices, col_idx, arena);

    // Scatter from permuted matrix (analysis ordering).
    front.scatter_original(Ap.col_ptr, Ap.row_idx, Ap.values);

    // a22[col*ext+row] for row>=col: accumulates child A22 contributions.
    std::vector<double> a22_par;
    if (ext > 0)
      a22_par.assign(static_cast<std::size_t>(ext) * static_cast<std::size_t>(ext), 0.0);

    // Assemble children contributions (all done by taskwait).
    for (const Int c : sn.children) {
      const std::size_t ci = static_cast<std::size_t>(c);
      if (contrib[ci].empty())
        continue;

      const Supernode &csn = keep.supernodes[ci];
      const FrontalInfo &cfi = keep.fronts[ci];
      const Int cp = csn.width();
      const Int cext = cfi.front_size() - cp;
      if (cext <= 0)
        continue;

      std::vector<Int> prows(static_cast<std::size_t>(cext));
      for (Int i = 0; i < cext; ++i) {
        const Int glob = cfi.row_indices[static_cast<std::size_t>(cp + i)];
        const auto it = std::lower_bound(fi.row_indices.cbegin(),
                                         fi.row_indices.cend(), glob);
        prows[static_cast<std::size_t>(i)] =
            static_cast<Int>(it - fi.row_indices.cbegin());
      }

      front.assemble_contrib(contrib[ci].data(), cext, prows);

      // Accumulate A22-type child contributions into a22_par.
      if (ext > 0) {
        const double *cc = contrib[ci].data();
        for (Int jj = 0; jj < cext; ++jj) {
          const Int pr_j = prows[static_cast<std::size_t>(jj)];
          if (pr_j < p) continue;
          const Int lcol = pr_j - p;
          for (Int ii = jj; ii < cext; ++ii) {
            const Int pr_i = prows[static_cast<std::size_t>(ii)];
            if (pr_i < p) continue;
            const Int lrow = pr_i - p;
            const double val =
                cc[static_cast<std::size_t>(jj) *
                       static_cast<std::size_t>(cext) +
                   static_cast<std::size_t>(ii)];
            if (lrow >= lcol)
              a22_par[static_cast<std::size_t>(lcol) *
                          static_cast<std::size_t>(ext) +
                      static_cast<std::size_t>(lrow)] += val;
            else
              a22_par[static_cast<std::size_t>(lrow) *
                          static_cast<std::size_t>(ext) +
                      static_cast<std::size_t>(lcol)] += val;
          }
        }
      }

      contrib[ci].clear();
      contrib[ci].shrink_to_fit();
    }

    // BBK pivot loop on the fully-summed block.
    const int num_fs = static_cast<int>(p);
    std::vector<int8_t> pivot_tag(static_cast<std::size_t>(num_fs), int8_t{0});
    InertiaCounts local_inertia{};
    int local_delayed = 0;

    int k = 0;
    while (k < num_fs) {
      PivotResult pr =
          choose_pivot(front, k, num_fs, ctrl.pivot_u, ctrl.small_pivot);

      if (pr.decision == PivotDecision::Reject) {
        ++local_delayed;
        ++k;
        continue;
      }

      if (pr.decision == PivotDecision::Accept1x1) {
        if (pr.col0 != k)
          sym_swap_front(front, k, pr.col0, num_fs);
        const double d00 =
            front.at(static_cast<Int>(k), static_cast<Int>(k));
        add_inertia_1x1(d00, ctrl.small_pivot, local_inertia);
        pivot_tag[static_cast<std::size_t>(k)] = int8_t{1};
        apply_pivot_1x1(front, k, num_fs);
        ++k;
      } else {
        assert(k + 1 < num_fs);
        if (pr.col1 != k + 1)
          sym_swap_front(front, k + 1, pr.col1, num_fs);
        const double d00 =
            front.at(static_cast<Int>(k), static_cast<Int>(k));
        const double d10 =
            front.at(static_cast<Int>(k + 1), static_cast<Int>(k));
        const double d11 =
            front.at(static_cast<Int>(k + 1), static_cast<Int>(k + 1));
        add_inertia_2x2(d00, d10, d11, ctrl.small_pivot, local_inertia);
        pivot_tag[static_cast<std::size_t>(k)] = int8_t{2};
        pivot_tag[static_cast<std::size_t>(k + 1)] = int8_t{-1};
        apply_pivot_2x2(front, k, num_fs, ctrl.small_pivot);
        k += 2;
      }
    }

    // Rejected pivots → zero eigenvalues.
    local_inertia.zero += local_delayed;
    if (local_delayed > 0) {
      int expected = 0;
      err_flag.compare_exchange_strong(expected, 1,
                                       std::memory_order_relaxed);
    }

    node_inertia[si] = local_inertia;
    node_delayed[si] = local_delayed;

    // Store factor columns.
    {
      const std::size_t lsize =
          static_cast<std::size_t>(f) * static_cast<std::size_t>(p);
      const std::size_t base =
          static_cast<std::size_t>(fkeep.factor_col_ptr[si]);
      const double *src = front.data();
      for (std::size_t qi = 0; qi < lsize; ++qi)
        fkeep.factor_values[base + qi] = src[qi];
    }

    if (si < fkeep.pivot_types.size())
      fkeep.pivot_types[si] = pivot_tag;

    // Compute contribution block for parent.
    if (ext > 0 && sn.parent >= 0) {
      const std::size_t ext_sz = static_cast<std::size_t>(ext);
      // Initialize with accumulated a22 contributions.
      contrib[si] = a22_par;
      double *cb = contrib[si].data();

      for (int pk = 0; pk < num_fs; ++pk) {
        const int8_t tag = pivot_tag[static_cast<std::size_t>(pk)];
        if (tag == int8_t{0} || tag == int8_t{-1})
          continue;

        if (tag == int8_t{1}) {
          const double d = front.at(static_cast<Int>(pk), static_cast<Int>(pk));
          if (std::abs(d) < ctrl.small_pivot)
            continue;
          // After apply_pivot_1x1, extension rows already store L factor entries.
          for (Int j = 0; j < ext; ++j) {
            const double lj =
                front.at(p + j, static_cast<Int>(pk));
            for (Int i = j; i < ext; ++i) {
              const double li =
                  front.at(p + i, static_cast<Int>(pk));
              cb[static_cast<std::size_t>(j) * ext_sz +
                 static_cast<std::size_t>(i)] -= li * d * lj;
            }
          }
        } else {
          const double d00 =
              front.at(static_cast<Int>(pk), static_cast<Int>(pk));
          const double d10 =
              front.at(static_cast<Int>(pk + 1), static_cast<Int>(pk));
          const double d11 =
              front.at(static_cast<Int>(pk + 1), static_cast<Int>(pk + 1));
          const double det = d00 * d11 - d10 * d10;
          if (std::abs(det) < ctrl.small_pivot)
            continue;

          // After apply_pivot_2x2, extension rows already store L factor entries.
          for (Int j = 0; j < ext; ++j) {
            const double lj0 = front.at(p + j, static_cast<Int>(pk));
            const double lj1 = front.at(p + j, static_cast<Int>(pk + 1));
            for (Int i = j; i < ext; ++i) {
              const double li0 = front.at(p + i, static_cast<Int>(pk));
              const double li1 = front.at(p + i, static_cast<Int>(pk + 1));

              cb[static_cast<std::size_t>(j) * ext_sz +
                 static_cast<std::size_t>(i)] -=
                  li0 * d00 * lj0 + li0 * d10 * lj1 + li1 * d10 * lj0 +
                  li1 * d11 * lj1;
            }
          }
        }
      }
    }

    arena.reset_to(mark);
  };

  run_parallel_postorder(keep.supernodes, process_node, ctrl.num_threads);

  // Accumulate per-node statistics serially.
  InertiaCounts total_inertia{};
  int total_delayed = 0;
  for (Int s = 0; s < ns; ++s) {
    const std::size_t si = static_cast<std::size_t>(s);
    total_inertia.positive += node_inertia[si].positive;
    total_inertia.negative += node_inertia[si].negative;
    total_inertia.zero += node_inertia[si].zero;
    total_delayed += node_delayed[si];
  }

  out_inertia = total_inertia;
  out_delayed = total_delayed;

  return (err_flag.load() != 0) ? FactorStatus::Singular : FactorStatus::Success;
}
#endif // SMF_PARALLEL

FactorStatus factor_indef(const AnalysisKeep &keep, const Control &ctrl,
                          Info &info, FactorKeep &fkeep) {
  const Int ns = static_cast<Int>(keep.supernodes.size());

  // ---------------------------------------------------------------
  // Initialise output bookkeeping
  // ---------------------------------------------------------------
  info.delayed_pivots = 0;

  // Compute total factor storage and build factor_col_ptr
  fkeep.factor_col_ptr.resize(static_cast<std::size_t>(ns) + 1, 0);
  {
    int offset = 0;
    for (Int s = 0; s < ns; ++s) {
      const std::size_t si = static_cast<std::size_t>(s);
      fkeep.factor_col_ptr[si] = offset;
      const Int w = keep.supernodes[si].width();
      const Int f = keep.fronts[si].front_size();
      offset += static_cast<int>(f) * static_cast<int>(w);
    }
    fkeep.factor_col_ptr[static_cast<std::size_t>(ns)] = offset;
    fkeep.factor_values.assign(static_cast<std::size_t>(offset), 0.0);
  }

  fkeep.perm.assign(keep.perm.begin(), keep.perm.end());
  fkeep.iperm.assign(keep.iperm.begin(), keep.iperm.end());
  fkeep.analysis = &keep;
  fkeep.is_posdef = false;
  fkeep.pivot_types.assign(static_cast<std::size_t>(ns), {});

  if (ns == 0) {
    info.num_positive = 0;
    info.num_negative = 0;
    info.num_zero = 0;
    return FactorStatus::Success;
  }

#ifdef SMF_PARALLEL
  if (ctrl.num_threads > 1) {
    // Use the pre-computed permuted matrix pattern from analysis, permute values only.
    std::vector<double> Ap_values = permute_values_only_indef(
        keep.cleaned, keep.iperm, keep.perm_col_ptr, keep.perm_row_idx);
    CscLower Ap;
    Ap.n = keep.n;
    Ap.col_ptr = keep.perm_col_ptr;
    Ap.row_idx = keep.perm_row_idx;
    Ap.values = std::move(Ap_values);
    
    InertiaCounts par_inertia{};
    int par_delayed = 0;
    const FactorStatus ps =
        factor_indef_parallel(keep, Ap, ctrl, fkeep, par_inertia, par_delayed);
    info.delayed_pivots = par_delayed;
    info.num_positive = par_inertia.positive;
    info.num_negative = par_inertia.negative;
    info.num_zero = par_inertia.zero;
    info.numerical_rank =
        static_cast<int>(keep.n) - par_inertia.zero;
    if (par_inertia.zero > 0)
      return FactorStatus::Singular;
    return ps;
  }
#endif // SMF_PARALLEL

  // ---- Serial path -----------------------------------------

  // Use the pre-computed permuted matrix pattern from analysis, permute values only.
  std::vector<double> Ap_values = permute_values_only_indef(
      keep.cleaned, keep.iperm, keep.perm_col_ptr, keep.perm_row_idx);
  CscLower Ap;
  Ap.n = keep.n;
  Ap.col_ptr = keep.perm_col_ptr;
  Ap.row_idx = keep.perm_row_idx;
  Ap.values = std::move(Ap_values);

  // Per-supernode contribution blocks (ext×ext, column-major, heap-alloc).
  std::vector<std::vector<double>> contrib(static_cast<std::size_t>(ns));

  // Arena for FrontalMatrix buffers — reset after each supernode.
  AlignedArena arena(static_cast<std::size_t>(32) << 20); // 32 MiB

  InertiaCounts inertia{};

  for (Int s = 0; s < ns; ++s) {
    const std::size_t si = static_cast<std::size_t>(s);
    const Supernode &sn = keep.supernodes[si];
    const FrontalInfo &fi = keep.fronts[si];
    const Int p = sn.width();      // pivot (fully-summed) columns
    const Int f = fi.front_size(); // total front rows (pivot + extension)
    const Int ext = f - p;         // extension rows

    if (p == 0)
      continue;

    // Column indices for this supernode = first p rows of the front.
    std::vector<Int> col_idx(fi.row_indices.cbegin(),
                             fi.row_indices.cbegin() +
                                 static_cast<std::ptrdiff_t>(p));

    AlignedArena::Marker mark = arena.save();
    FrontalMatrix front(f, p, fi.row_indices, col_idx, arena);

    // ---- Scatter permuted A values ----------------------------------
    front.scatter_original(Ap.col_ptr, Ap.row_idx, Ap.values);

    // a22[col*ext+row] for row>=col: accumulates child A22 contributions
    // that land in extension×extension positions of this front.
    std::vector<double> a22;
    if (ext > 0)
      a22.assign(static_cast<std::size_t>(ext) * static_cast<std::size_t>(ext), 0.0);

    // ---- Assemble child contribution blocks -------------------------
    for (Int c : sn.children) {
      const std::size_t ci = static_cast<std::size_t>(c);
      if (contrib[ci].empty())
        continue;

      const Supernode &csn = keep.supernodes[ci];
      const FrontalInfo &cfi = keep.fronts[ci];
      const Int cp = csn.width();
      const Int cext = cfi.front_size() - cp;
      if (cext <= 0)
        continue;

      std::vector<Int> prows(static_cast<std::size_t>(cext));
      for (Int i = 0; i < cext; ++i) {
        const Int glob = cfi.row_indices[static_cast<std::size_t>(cp + i)];
        const auto it = std::lower_bound(fi.row_indices.cbegin(),
                                         fi.row_indices.cend(), glob);
        prows[static_cast<std::size_t>(i)] =
            static_cast<Int>(it - fi.row_indices.cbegin());
      }

      front.assemble_contrib(contrib[ci].data(), cext, prows);

      // Accumulate A22-type entries (both row and col map to extension rows)
      // into a22, so the contribution block carries the full Schur complement.
      if (ext > 0) {
        const double *cc = contrib[ci].data();
        for (Int jj = 0; jj < cext; ++jj) {
          const Int pr_j = prows[static_cast<std::size_t>(jj)];
          if (pr_j < p)
            continue;
          const Int lcol = pr_j - p;
          for (Int ii = jj; ii < cext; ++ii) {
            const Int pr_i = prows[static_cast<std::size_t>(ii)];
            if (pr_i < p)
              continue;
            const Int lrow = pr_i - p;
            const double val =
                cc[static_cast<std::size_t>(jj) *
                       static_cast<std::size_t>(cext) +
                   static_cast<std::size_t>(ii)];
            // Store in lower-triangle position (lrow >= lcol or swap)
            if (lrow >= lcol)
              a22[static_cast<std::size_t>(lcol) *
                      static_cast<std::size_t>(ext) +
                  static_cast<std::size_t>(lrow)] += val;
            else
              a22[static_cast<std::size_t>(lrow) *
                      static_cast<std::size_t>(ext) +
                  static_cast<std::size_t>(lcol)] += val;
          }
        }
      }

      contrib[ci].clear();
      contrib[ci].shrink_to_fit();
    }

    // ---- BBK pivot loop on the p×p fully-summed block ---------------
    // pivot_tag[k]: 0=delayed, 1=1×1 accepted, 2=first of 2×2, -1=second of 2×2
    const int num_fs = static_cast<int>(p);
    std::vector<int8_t> pivot_tag(static_cast<std::size_t>(num_fs), int8_t{0});

    int delayed_here = 0;
    int k = 0;
    while (k < num_fs) {
      PivotResult pr =
          choose_pivot(front, k, num_fs, ctrl.pivot_u, ctrl.small_pivot);

      if (pr.decision == PivotDecision::Reject) {
        // column k is delayed — leave tag at 0
        ++delayed_here;
        ++k;
        continue;
      }

      if (pr.decision == PivotDecision::Accept1x1) {
        if (pr.col0 != k)
          sym_swap_front(front, k, pr.col0, num_fs);

        const double d00 = front.at(static_cast<Int>(k), static_cast<Int>(k));
        add_inertia_1x1(d00, ctrl.small_pivot, inertia);
        pivot_tag[static_cast<std::size_t>(k)] = int8_t{1};
        apply_pivot_1x1(front, k, num_fs);
        ++k;

      } else { // Accept2x2
        assert(k + 1 < num_fs);
        if (pr.col1 != k + 1)
          sym_swap_front(front, k + 1, pr.col1, num_fs);

        const double d00 = front.at(static_cast<Int>(k), static_cast<Int>(k));
        const double d10 =
            front.at(static_cast<Int>(k + 1), static_cast<Int>(k));
        const double d11 =
            front.at(static_cast<Int>(k + 1), static_cast<Int>(k + 1));
        add_inertia_2x2(d00, d10, d11, ctrl.small_pivot, inertia);
        pivot_tag[static_cast<std::size_t>(k)] = int8_t{2};
        pivot_tag[static_cast<std::size_t>(k + 1)] = int8_t{-1};
        apply_pivot_2x2(front, k, num_fs, ctrl.small_pivot);
        k += 2;
      }
    }

    info.delayed_pivots += delayed_here;

    // DECISION LOG (M5.S2): Rejected pivots are zero eigenvalues.
    // In this implementation there is no real "delay-to-parent" mechanism —
    // a Reject at any supernode means the column can never be factored and
    // its eigenvalue contribution is zero.  Count each rejected column in
    // inertia.zero so that info.num_zero and info.numerical_rank are correct.
    inertia.zero += delayed_here;

    // ---- Store the f×p factored front in fkeep.factor_values --------
    {
      const std::size_t lsize =
          static_cast<std::size_t>(f) * static_cast<std::size_t>(p);
      const std::size_t base =
          static_cast<std::size_t>(fkeep.factor_col_ptr[si]);
      const double *src = front.data();
      for (std::size_t q = 0; q < lsize; ++q)
        fkeep.factor_values[base + q] = src[q];
    }

    // ---- Store pivot tags for solve_diag ----------------------------
    if (si < fkeep.pivot_types.size())
      fkeep.pivot_types[si] = pivot_tag;

    // ---- Compute contribution block for parent ----------------------
    // CB[j*ext+i] = a22[j*ext+i] - ∑_k L_ext[i,k]·D_k·L_ext[j,k]  (i>=j)
    // a22 carries assembled child A22-type contributions; the Schur
    // complement from the pivot columns is subtracted below.
    if (ext > 0 && sn.parent >= 0) {
      const std::size_t ext_sz = static_cast<std::size_t>(ext);
      // Initialize contribution block from accumulated a22
      contrib[si] = a22;
      double *cb = contrib[si].data();

      for (int pk = 0; pk < num_fs; ++pk) {
        const int8_t tag = pivot_tag[static_cast<std::size_t>(pk)];
        if (tag == int8_t{0} || tag == int8_t{-1})
          continue;

        if (tag == int8_t{1}) {
          // 1×1 pivot at column pk
          // After apply_pivot_1x1, extension rows already store L factor entries
          // (divided by d), so read them directly — no extra inv_d needed.
          const double d = front.at(static_cast<Int>(pk), static_cast<Int>(pk));
          if (std::abs(d) < ctrl.small_pivot)
            continue;

          for (Int j = 0; j < ext; ++j) {
            const double lj = front.at(p + j, static_cast<Int>(pk));
            for (Int i = j; i < ext; ++i) {
              const double li = front.at(p + i, static_cast<Int>(pk));
              cb[static_cast<std::size_t>(j) * ext_sz +
                 static_cast<std::size_t>(i)] -= li * d * lj;
            }
          }

        } else { // tag == 2 : 2×2 pivot at (pk, pk+1)
          // After apply_pivot_2x2, extension rows already store L factor entries
          // for both columns pk and pk+1 — read them directly.
          const double d00 =
              front.at(static_cast<Int>(pk), static_cast<Int>(pk));
          const double d10 =
              front.at(static_cast<Int>(pk + 1), static_cast<Int>(pk));
          const double d11 =
              front.at(static_cast<Int>(pk + 1), static_cast<Int>(pk + 1));
          const double det = d00 * d11 - d10 * d10;
          if (std::abs(det) < ctrl.small_pivot)
            continue;

          for (Int j = 0; j < ext; ++j) {
            const double lj0 = front.at(p + j, static_cast<Int>(pk));
            const double lj1 = front.at(p + j, static_cast<Int>(pk + 1));

            for (Int i = j; i < ext; ++i) {
              const double li0 = front.at(p + i, static_cast<Int>(pk));
              const double li1 = front.at(p + i, static_cast<Int>(pk + 1));

              cb[static_cast<std::size_t>(j) * ext_sz +
                 static_cast<std::size_t>(i)] -=
                  li0 * d00 * lj0 + li0 * d10 * lj1 + li1 * d10 * lj0 +
                  li1 * d11 * lj1;
            }
          }
        }
      } // pk loop
    }

    arena.reset_to(mark);
  } // supernode loop

  // Accumulate inertia into info
  info.num_positive = inertia.positive;
  info.num_negative = inertia.negative;
  info.num_zero = inertia.zero;

  // DECISION LOG (M5.S2): numerical_rank = n minus zero-eigenvalue count.
  // All rejected pivots have been folded into inertia.zero above.
  info.numerical_rank = static_cast<int>(keep.n) - inertia.zero;

  // Return Singular whenever any pivot was rejected (zero) — regardless of
  // continue_on_singular.  The flag controls only whether the caller proceeds
  // to solve with the degraded factorization; it does not change the status.
  if (inertia.zero > 0)
    return FactorStatus::Singular;
  return FactorStatus::Success;
}

} // namespace smf
