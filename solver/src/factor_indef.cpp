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
// Permute VALUES into pre-computed permuted positions using the scatter map
// built during analyse().  O(nnz) — replaces the former O(nnz * avg_col_width)
// linear-search in permute_values_only_indef.
// ---------------------------------------------------------------------------
static std::vector<double> scatter_values_indef(
    const CscLower &A_orig,
    const std::vector<Int> &orig_to_perm_idx,
    Int nnz_perm) {
  std::vector<double> perm_values(static_cast<std::size_t>(nnz_perm), 0.0);
  const Int nnz = static_cast<Int>(A_orig.values.size());
  for (Int k = 0; k < nnz; ++k)
    perm_values[static_cast<std::size_t>(
        orig_to_perm_idx[static_cast<std::size_t>(k)])] +=
        A_orig.values[static_cast<std::size_t>(k)];
  return perm_values;
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

    // Per-task arena — not shared with other concurrent tasks.
    AlignedArena arena(static_cast<std::size_t>(8) << 20); // 8 MiB per task
    // Column indices precomputed at analyse() time.
    AlignedArena::Marker mark = arena.save();
    FrontalMatrix front(f, p, fi.row_indices, fi.col_indices, arena);

    // Scatter from permuted matrix (two-pointer merge: O(nnz_per_sn + f)).
    for (Int j = 0; j < p; ++j) {
      const Int orig_col = fi.col_indices[static_cast<std::size_t>(j)];
      const Int cs = Ap.col_ptr[static_cast<std::size_t>(orig_col)];
      const Int ce = Ap.col_ptr[static_cast<std::size_t>(orig_col) + 1];
      std::size_t fri = static_cast<std::size_t>(j);
      for (Int k = cs; k < ce; ++k) {
        const Int orig_row = Ap.row_idx[static_cast<std::size_t>(k)];
        while (fi.row_indices[fri] < orig_row) ++fri;
        front.data()[static_cast<std::size_t>(j) *
                         static_cast<std::size_t>(f) + fri] +=
            Ap.values[static_cast<std::size_t>(k)];
      }
    }

    // a22[col*ext+row] for row>=col: accumulates child A22 contributions.
    std::vector<double> a22_par;
    if (ext > 0)
      a22_par.assign(static_cast<std::size_t>(ext) * static_cast<std::size_t>(ext), 0.0);

    // Assemble children contributions (all done by taskwait).
    for (std::size_t ck = 0; ck < sn.children.size(); ++ck) {
      const Int c = sn.children[ck];
      const std::size_t ci = static_cast<std::size_t>(c);
      if (contrib[ci].empty())
        continue;

      const Int cext = keep.fronts[ci].front_size() -
                       keep.supernodes[ci].width();
      if (cext <= 0)
        continue;

      // Raw pointer into flat CPR data (no per-call allocation).
      const Int* prows = keep.cpr_data.data() +
          static_cast<std::size_t>(
              keep.cpr_ch_off[static_cast<std::size_t>(keep.cpr_sn_off[si]) + ck]);

      // Inline assemble_contrib: A11/A21 entries (pr_j < p → pivot col).
      // Column-major: front.data()[col*f + row].
      {
        double* Fdata = front.data();
        for (Int j = 0; j < cext; ++j) {
          const Int pr_j = prows[static_cast<std::size_t>(j)];
          if (pr_j >= p) continue;
          for (Int i = j; i < cext; ++i) {
            Fdata[static_cast<std::size_t>(pr_j) * static_cast<std::size_t>(f) +
                  static_cast<std::size_t>(prows[static_cast<std::size_t>(i)])] +=
                contrib[ci].data()[static_cast<std::size_t>(j) *
                                       static_cast<std::size_t>(cext) +
                                   static_cast<std::size_t>(i)];
          }
        }
      }

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
    // O(nnz) scatter using pre-built map from analyse().
    const Int nnz_perm_p = keep.perm_col_ptr[static_cast<std::size_t>(keep.n)];
    std::vector<double> Ap_values =
        scatter_values_indef(keep.cleaned, keep.orig_to_perm_idx, nnz_perm_p);
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

  // O(nnz) scatter using pre-built map from analyse().
  const Int nnz_perm_s = keep.perm_col_ptr[static_cast<std::size_t>(keep.n)];
  std::vector<double> Ap_values =
      scatter_values_indef(keep.cleaned, keep.orig_to_perm_idx, nnz_perm_s);
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

    // Column indices precomputed at analyse() time.
    AlignedArena::Marker mark = arena.save();
    FrontalMatrix front(f, p, fi.row_indices, fi.col_indices, arena);

    // ---- Scatter permuted A values (two-pointer merge: O(nnz_per_sn + f)) --
    for (Int j = 0; j < p; ++j) {
      const Int orig_col = fi.col_indices[static_cast<std::size_t>(j)];
      const Int cs = Ap.col_ptr[static_cast<std::size_t>(orig_col)];
      const Int ce = Ap.col_ptr[static_cast<std::size_t>(orig_col) + 1];
      std::size_t fri = static_cast<std::size_t>(j);
      for (Int k = cs; k < ce; ++k) {
        const Int orig_row = Ap.row_idx[static_cast<std::size_t>(k)];
        while (fi.row_indices[fri] < orig_row) ++fri;
        front.data()[static_cast<std::size_t>(j) *
                         static_cast<std::size_t>(f) + fri] +=
            Ap.values[static_cast<std::size_t>(k)];
      }
    }

    // a22[col*ext+row] for row>=col: accumulates child A22 contributions
    // that land in extension×extension positions of this front.
    std::vector<double> a22;
    if (ext > 0)
      a22.assign(static_cast<std::size_t>(ext) * static_cast<std::size_t>(ext), 0.0);

    // ---- Assemble child contribution blocks -------------------------
    for (std::size_t ck = 0; ck < sn.children.size(); ++ck) {
      const Int c = sn.children[ck];
      const std::size_t ci = static_cast<std::size_t>(c);
      if (contrib[ci].empty())
        continue;

      const Int cext = keep.fronts[ci].front_size() -
                       keep.supernodes[ci].width();
      if (cext <= 0)
        continue;

      // Raw pointer into flat CPR data (no per-call allocation).
      const Int* prows = keep.cpr_data.data() +
          static_cast<std::size_t>(
              keep.cpr_ch_off[static_cast<std::size_t>(keep.cpr_sn_off[si]) + ck]);

      // Inline assemble_contrib: A11/A21 entries (pr_j < p → pivot col).
      // Column-major: front.data()[col*f + row].
      {
        double* Fdata = front.data();
        for (Int j = 0; j < cext; ++j) {
          const Int pr_j = prows[static_cast<std::size_t>(j)];
          if (pr_j >= p) continue;
          for (Int i = j; i < cext; ++i) {
            Fdata[static_cast<std::size_t>(pr_j) * static_cast<std::size_t>(f) +
                  static_cast<std::size_t>(prows[static_cast<std::size_t>(i)])] +=
                contrib[ci].data()[static_cast<std::size_t>(j) *
                                       static_cast<std::size_t>(cext) +
                                   static_cast<std::size_t>(i)];
          }
        }
      }

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
