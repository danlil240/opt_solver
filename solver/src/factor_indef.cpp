#include "smf/factor_indef.hpp"
#include "smf/blas_thread_guard.hpp"
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

namespace {

struct NodePivotData {
  std::vector<Int> front_rows;
  std::vector<int8_t> pivot_tag;
  std::vector<double> d00;
  std::vector<double> d10;
  std::vector<double> d11;
};

void finalize_indef_factor_metadata(const AnalysisKeep &keep,
                                    const std::vector<NodePivotData> &node_data,
                                    FactorKeep &fkeep) {
  const int n = static_cast<int>(keep.n);
  const int ns = static_cast<int>(keep.supernodes.size());

  fkeep.n = keep.n;
  fkeep.num_perm.resize(static_cast<std::size_t>(n));
  std::iota(fkeep.num_perm.begin(), fkeep.num_perm.end(), 0);

  for (int s = 0; s < ns; ++s) {
    const std::size_t si = static_cast<std::size_t>(s);
    const Supernode &sn = keep.supernodes[si];
    const int p = static_cast<int>(sn.width());
    const auto &frows = node_data[si].front_rows;
    if (static_cast<int>(frows.size()) < p)
      continue;
    for (int k = 0; k < p; ++k) {
      fkeep.num_perm[static_cast<std::size_t>(sn.col_start + k)] =
          static_cast<int>(frows[static_cast<std::size_t>(k)]);
    }
  }

  fkeep.num_iperm.assign(static_cast<std::size_t>(n), 0);
  for (int new_piv = 0; new_piv < n; ++new_piv) {
    const int sym_idx = fkeep.num_perm[static_cast<std::size_t>(new_piv)];
    if (sym_idx >= 0 && sym_idx < n) {
      fkeep.num_iperm[static_cast<std::size_t>(sym_idx)] = new_piv;
    }
  }

  fkeep.perm.resize(static_cast<std::size_t>(n));
  fkeep.iperm.resize(static_cast<std::size_t>(n));
  for (int new_piv = 0; new_piv < n; ++new_piv) {
    const int sym_idx = fkeep.num_perm[static_cast<std::size_t>(new_piv)];
    const int old_idx =
        (sym_idx >= 0 && sym_idx < static_cast<int>(keep.perm.size()))
            ? static_cast<int>(keep.perm[static_cast<std::size_t>(sym_idx)])
            : new_piv;
    fkeep.perm[static_cast<std::size_t>(new_piv)] = old_idx;
    if (old_idx >= 0 && old_idx < n) {
      fkeep.iperm[static_cast<std::size_t>(old_idx)] = new_piv;
    }
  }

  // Build D blocks exclusively in fkeep.diag_entries
  fkeep.diag_entries.clear();
  fkeep.diag_entries.reserve(static_cast<std::size_t>(n));
  for (int s = 0; s < ns; ++s) {
    const std::size_t si = static_cast<std::size_t>(s);
    const Supernode &sn = keep.supernodes[si];
    const int p = static_cast<int>(sn.width());
    const auto &nd = node_data[si];
    if (nd.pivot_tag.empty())
      continue;

    for (int k = 0; k < p; ++k) {
      const std::size_t ki = static_cast<std::size_t>(k);
      const int8_t tag = nd.pivot_tag[ki];
      const Int sym_r0 = nd.front_rows[ki];
      const Int row0 = static_cast<Int>(fkeep.num_iperm[static_cast<std::size_t>(sym_r0)]);
      if (tag == 1) {
        DiagSolveEntry entry;
        entry.tag = 1;
        entry.row0 = row0;
        entry.d00 = nd.d00[ki];
        fkeep.diag_entries.push_back(entry);
      } else if (tag == 2 && k + 1 < p) {
        const Int sym_r1 = nd.front_rows[ki + 1];
        const Int row1 = static_cast<Int>(fkeep.num_iperm[static_cast<std::size_t>(sym_r1)]);
        DiagSolveEntry entry;
        entry.tag = 2;
        entry.row0 = row0;
        entry.row1 = row1;
        entry.d00 = nd.d00[ki];
        entry.d10 = nd.d10[ki];
        entry.d11 = nd.d11[ki];
        fkeep.diag_entries.push_back(entry);
      } else if (tag == 0) {
        DiagSolveEntry entry;
        entry.tag = 1;
        entry.row0 = row0;
        entry.d00 = 0.0;
        fkeep.diag_entries.push_back(entry);
      }
    }
  }

  // Build solve_steps and solve_row_indices in total-permutation coordinates
  fkeep.solve_steps.clear();
  fkeep.solve_row_indices.clear();
  std::vector<Int> fallback_order;
  const std::vector<Int> *order_ptr = &keep.solve_postorder;
  if (order_ptr->empty()) {
    if (!keep.postorder.empty()) {
      order_ptr = &keep.postorder;
    } else {
      fallback_order.resize(static_cast<std::size_t>(ns));
      std::iota(fallback_order.begin(), fallback_order.end(), 0);
      order_ptr = &fallback_order;
    }
  }

  fkeep.solve_steps.reserve(order_ptr->size());
  std::size_t row_count = 0;
  int max_p = 0;
  int max_q = 0;
  for (Int s : *order_ptr) {
    const std::size_t si = static_cast<std::size_t>(s);
    const int p = static_cast<int>(keep.supernodes[si].width());
    const int f = static_cast<int>(keep.fronts[si].front_size());
    row_count += static_cast<std::size_t>(f);
    max_p = std::max(max_p, p);
    max_q = std::max(max_q, f - p);
  }
  fkeep.solve_row_indices.reserve(row_count);

  for (Int s : *order_ptr) {
    const std::size_t si = static_cast<std::size_t>(s);
    const FrontalInfo &fi = keep.fronts[si];
    const Supernode &sn = keep.supernodes[si];
    const auto &frows = node_data[si].front_rows.empty()
                            ? fi.row_indices
                            : node_data[si].front_rows;
    SolveStep step;
    step.row_offset = static_cast<Int>(fkeep.solve_row_indices.size());
    step.factor_offset = fkeep.factor_col_ptr[si];
    step.p = static_cast<int>(sn.width());
    step.f = static_cast<int>(fi.front_size());
    fkeep.solve_steps.push_back(step);
    for (Int sym_r : frows) {
      fkeep.solve_row_indices.push_back(
          static_cast<Int>(fkeep.num_iperm[static_cast<std::size_t>(sym_r)]));
    }
  }

  fkeep.solve_tmp.resize(static_cast<std::size_t>(keep.n));
  fkeep.solve_loc.resize(static_cast<std::size_t>(max_p));
  fkeep.solve_ext.resize(static_cast<std::size_t>(max_q));
}

void factor_single_front(FrontalMatrix &front,
                         int num_fs,
                         const Control &ctrl,
                         double eff_small,
                         NodePivotData &nd,
                         InertiaCounts &local_inertia,
                         int &local_delayed) {
  nd.pivot_tag.assign(static_cast<std::size_t>(num_fs), int8_t{0});
  nd.d00.assign(static_cast<std::size_t>(num_fs), 0.0);
  nd.d10.assign(static_cast<std::size_t>(num_fs), 0.0);
  nd.d11.assign(static_cast<std::size_t>(num_fs), 0.0);

  int k = 0;
  while (k < num_fs) {
    PivotResult pr =
        choose_pivot(front, k, num_fs, ctrl.pivot_u, eff_small);

    if (pr.decision == PivotDecision::Reject) {
      ++local_delayed;
      ++k;
      continue;
    }

    if (pr.decision == PivotDecision::Accept1x1) {
      if (pr.col0 != k) {
        sym_swap_front(front, k, pr.col0, num_fs);
        std::swap(nd.front_rows[static_cast<std::size_t>(k)],
                  nd.front_rows[static_cast<std::size_t>(pr.col0)]);
      }
      const double d00 = front.at(static_cast<Int>(k), static_cast<Int>(k));
      add_inertia_1x1(d00, eff_small, local_inertia);
      nd.pivot_tag[static_cast<std::size_t>(k)] = int8_t{1};
      nd.d00[static_cast<std::size_t>(k)] = d00;
      apply_pivot_1x1(front, k, num_fs);
      ++k;
    } else {
      assert(k + 1 < num_fs);
      if (pr.col1 != k + 1) {
        sym_swap_front(front, k + 1, pr.col1, num_fs);
        std::swap(nd.front_rows[static_cast<std::size_t>(k + 1)],
                  nd.front_rows[static_cast<std::size_t>(pr.col1)]);
      }
      const double d00 = front.at(static_cast<Int>(k), static_cast<Int>(k));
      const double d10 =
          front.at(static_cast<Int>(k + 1), static_cast<Int>(k));
      const double d11 =
          front.at(static_cast<Int>(k + 1), static_cast<Int>(k + 1));
      add_inertia_2x2(d00, d10, d11, ctrl.small_pivot, local_inertia);
      nd.pivot_tag[static_cast<std::size_t>(k)] = int8_t{2};
      nd.pivot_tag[static_cast<std::size_t>(k + 1)] = int8_t{-1};
      nd.d00[static_cast<std::size_t>(k)] = d00;
      nd.d10[static_cast<std::size_t>(k)] = d10;
      nd.d11[static_cast<std::size_t>(k)] = d11;
      apply_pivot_2x2(front, k, num_fs, ctrl.small_pivot);
      k += 2;
    }
  }
}

void store_unit_lower_front(const FrontalMatrix &front,
                            int f,
                            int p,
                            const NodePivotData &nd,
                            double *dst) {
  const std::size_t lsize =
      static_cast<std::size_t>(f) * static_cast<std::size_t>(p);
  const double *src = front.data();
  for (std::size_t q = 0; q < lsize; ++q) {
    dst[q] = src[q];
  }
  for (int k = 0; k < p; ++k) {
    // Zero upper triangle above diagonal in column k
    for (int r = 0; r < k; ++r) {
      dst[static_cast<std::size_t>(k) * f + r] = 0.0;
    }
    const int8_t tag = nd.pivot_tag[static_cast<std::size_t>(k)];
    if (tag == 1) {
      dst[static_cast<std::size_t>(k) * f + k] = 1.0;
    } else if (tag == 2 && k + 1 < p) {
      dst[static_cast<std::size_t>(k) * f + k] = 1.0;
      dst[static_cast<std::size_t>(k) * f + (k + 1)] = 0.0;
      for (int r = 0; r <= k; ++r) {
        dst[static_cast<std::size_t>(k + 1) * f + r] = 0.0;
      }
      dst[static_cast<std::size_t>(k + 1) * f + (k + 1)] = 1.0;
    } else if (tag == 0) {
      for (int r = k; r < f; ++r) {
        dst[static_cast<std::size_t>(k) * f + r] = (r == k) ? 1.0 : 0.0;
      }
    }
  }
}

void compute_front_contrib(const FrontalMatrix &front,
                           Int p,
                           Int ext,
                           const NodePivotData &nd,
                           std::vector<double> &cb_vec) {
  const std::size_t ext_sz = static_cast<std::size_t>(ext);
  double *cb = cb_vec.data();
  const int num_fs = static_cast<int>(p);

  for (int pk = 0; pk < num_fs; ++pk) {
    const std::size_t pki = static_cast<std::size_t>(pk);
    const int8_t tag = nd.pivot_tag[pki];
    if (tag == int8_t{0} || tag == int8_t{-1})
      continue;

    if (tag == int8_t{1}) {
      const double d = nd.d00[pki];
      if (d == 0.0)
        continue;
      for (Int j = 0; j < ext; ++j) {
        const double w_j = d * front.at(p + j, static_cast<Int>(pk));
        for (Int i = j; i < ext; ++i) {
          const double li = front.at(p + i, static_cast<Int>(pk));
          cb[static_cast<std::size_t>(j) * ext_sz +
             static_cast<std::size_t>(i)] -= li * w_j;
        }
      }
    } else if (tag == int8_t{2}) {
      const double d00 = nd.d00[pki];
      const double d10 = nd.d10[pki];
      const double d11 = nd.d11[pki];
      for (Int j = 0; j < ext; ++j) {
        const double lj0 = front.at(p + j, static_cast<Int>(pk));
        const double lj1 = front.at(p + j, static_cast<Int>(pk + 1));
        const double wj0 = d00 * lj0 + d10 * lj1;
        const double wj1 = d10 * lj0 + d11 * lj1;
        for (Int i = j; i < ext; ++i) {
          const double li0 = front.at(p + i, static_cast<Int>(pk));
          const double li1 = front.at(p + i, static_cast<Int>(pk + 1));
          cb[static_cast<std::size_t>(j) * ext_sz +
             static_cast<std::size_t>(i)] -= li0 * wj0 + li1 * wj1;
        }
      }
    }
  }
}

} // namespace

// ---------------------------------------------------------------------------
// Permute VALUES into pre-computed permuted positions using the scatter map
// built during analyse().
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
                      FactorKeep &fkeep,
                      std::vector<NodePivotData> &node_data,
                      InertiaCounts &out_inertia,
                      int &out_delayed) {
  const Int ns = static_cast<Int>(keep.supernodes.size());

  std::vector<std::vector<double>> contrib(static_cast<std::size_t>(ns));
  std::vector<InertiaCounts> node_inertia(static_cast<std::size_t>(ns));
  std::vector<int> node_delayed(static_cast<std::size_t>(ns), 0);
  std::atomic<int> err_flag{0};

  double matrix_max_abs = 0.0;
  for (double v : Ap.values) {
    const double av = std::abs(v);
    if (av > matrix_max_abs) matrix_max_abs = av;
  }
  const double eff_small =
      (matrix_max_abs > 0.0 && matrix_max_abs < ctrl.small_pivot)
          ? (ctrl.small_pivot * matrix_max_abs)
          : ctrl.small_pivot;

  auto process_node = [&](Int s) {
    const std::size_t si = static_cast<std::size_t>(s);
    const Supernode &sn = keep.supernodes[si];
    const FrontalInfo &fi = keep.fronts[si];
    const Int p = sn.width();
    const Int f = fi.front_size();
    const Int ext = f - p;

    node_data[si].front_rows = fi.row_indices;
    if (p == 0)
      return;

    AlignedArena arena(static_cast<std::size_t>(8) << 20);
    AlignedArena::Marker mark = arena.save();

    FrontalMatrix front(f, p, fi.row_indices.data(), fi.row_indices.data(), arena);
    front.scatter_original(Ap.col_ptr, Ap.row_idx, Ap.values);

    std::vector<double> a22_par;
    if (ext > 0)
      a22_par.assign(static_cast<std::size_t>(ext) * static_cast<std::size_t>(ext), 0.0);

    std::vector<Int> prows_local;
    for (std::size_t k = 0; k < sn.children.size(); ++k) {
      const Int c = sn.children[k];
      const std::size_t ci = static_cast<std::size_t>(c);
      if (contrib[ci].empty())
        continue;

      const FrontalInfo &cfi = keep.fronts[ci];
      const Int cp = keep.supernodes[ci].width();
      const Int cext = cfi.front_size() - cp;
      if (cext <= 0)
        continue;

      const std::vector<Int> *prows_ptr = nullptr;
      if (k < fi.child_parent_rows.size() &&
          fi.child_parent_rows[k].size() == static_cast<std::size_t>(cext)) {
        prows_ptr = &fi.child_parent_rows[k];
      } else {
        prows_local.resize(static_cast<std::size_t>(cext));
        for (Int i = 0; i < cext; ++i) {
          const Int glob = cfi.row_indices[static_cast<std::size_t>(cp + i)];
          const auto it = std::lower_bound(fi.row_indices.cbegin(),
                                           fi.row_indices.cend(), glob);
          prows_local[static_cast<std::size_t>(i)] =
              static_cast<Int>(it - fi.row_indices.cbegin());
        }
        prows_ptr = &prows_local;
      }
      const std::vector<Int> &prows = *prows_ptr;

      front.assemble_contrib(contrib[ci].data(), cext, prows);

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
    }

    const int num_fs = static_cast<int>(p);
    InertiaCounts local_inertia{};
    int local_delayed = 0;

    factor_single_front(front, num_fs, ctrl, eff_small, node_data[si], local_inertia, local_delayed);

    local_inertia.zero += local_delayed;
    if (local_delayed > 0) {
      int expected = 0;
      err_flag.compare_exchange_strong(expected, 1,
                                       std::memory_order_relaxed);
    }

    node_inertia[si] = local_inertia;
    node_delayed[si] = local_delayed;

    store_unit_lower_front(
        front,
        static_cast<int>(f),
        static_cast<int>(p),
        node_data[si],
        fkeep.factor_values.data() + static_cast<std::size_t>(fkeep.factor_col_ptr[si]));

    if (si < fkeep.pivot_types.size())
      fkeep.pivot_types[si] = node_data[si].pivot_tag;

    if (ext > 0 && sn.parent >= 0) {
      contrib[si] = std::move(a22_par);
      compute_front_contrib(front, p, ext, node_data[si], contrib[si]);
    }

    arena.reset_to(mark);
  };

  run_parallel_postorder(keep.supernodes, process_node, ctrl.num_threads);

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

  info.delayed_pivots = 0;

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

  fkeep.n = keep.n;
  fkeep.perm.assign(keep.perm.begin(), keep.perm.end());
  fkeep.iperm.assign(keep.iperm.begin(), keep.iperm.end());
  fkeep.num_perm.resize(static_cast<std::size_t>(keep.n));
  std::iota(fkeep.num_perm.begin(), fkeep.num_perm.end(), 0);
  fkeep.num_iperm = fkeep.num_perm;
  fkeep.analysis = &keep;
  fkeep.is_posdef = false;
  fkeep.status = FactorStatus::Success;
  fkeep.pivot_types.assign(static_cast<std::size_t>(ns), {});

  if (ns == 0) {
    info.num_positive = 0;
    info.num_negative = 0;
    info.num_zero = 0;
    info.factor_status = FactorStatus::Success;
    info.status = ErrorCode::Success;
    return FactorStatus::Success;
  }

  std::vector<NodePivotData> node_data(static_cast<std::size_t>(ns));

#ifdef SMF_PARALLEL
  if (ctrl.num_threads > 1) {
    BlasSerialGuard blas_guard;
    CscLower Ap;
    Ap.n = keep.n;
    if (keep.orig_to_perm_idx.empty()) {
      Ap.col_ptr = keep.cleaned.col_ptr;
      Ap.row_idx = keep.cleaned.row_idx;
      Ap.values = keep.cleaned.values;
    } else {
      const Int nnz_perm_p =
          keep.perm_col_ptr[static_cast<std::size_t>(keep.n)];
      std::vector<double> Ap_values =
          scatter_values_indef(keep.cleaned, keep.orig_to_perm_idx, nnz_perm_p);
      Ap.col_ptr = keep.perm_col_ptr;
      Ap.row_idx = keep.perm_row_idx;
      Ap.values = std::move(Ap_values);
    }

    InertiaCounts par_inertia{};
    int par_delayed = 0;
    const FactorStatus ps = factor_indef_parallel(
        keep, Ap, ctrl, fkeep, node_data, par_inertia, par_delayed);
    info.delayed_pivots = par_delayed;
    info.num_positive = par_inertia.positive;
    info.num_negative = par_inertia.negative;
    info.num_zero = par_inertia.zero;
    info.numerical_rank = static_cast<int>(keep.n) - par_inertia.zero;
    finalize_indef_factor_metadata(keep, node_data, fkeep);
    fkeep.status = (par_inertia.zero > 0) ? FactorStatus::Singular : ps;
    info.factor_status = fkeep.status;
    info.status = (fkeep.status == FactorStatus::Success)
                      ? ErrorCode::Success
                      : ErrorCode::SingularMatrix;
    return fkeep.status;
  }
#endif // SMF_PARALLEL

  // ---- Serial path -----------------------------------------

  CscLower Ap;
  Ap.n = keep.n;
  if (keep.orig_to_perm_idx.empty()) {
    Ap.col_ptr = keep.cleaned.col_ptr;
    Ap.row_idx = keep.cleaned.row_idx;
    Ap.values = keep.cleaned.values;
  } else {
    const Int nnz_perm_s = keep.perm_col_ptr[static_cast<std::size_t>(keep.n)];
    std::vector<double> Ap_values =
        scatter_values_indef(keep.cleaned, keep.orig_to_perm_idx, nnz_perm_s);
    Ap.col_ptr = keep.perm_col_ptr;
    Ap.row_idx = keep.perm_row_idx;
    Ap.values = std::move(Ap_values);
  }

  std::vector<std::vector<double>> contrib(static_cast<std::size_t>(ns));
  AlignedArena arena(static_cast<std::size_t>(32) << 20);
  InertiaCounts inertia{};

  double matrix_max_abs = 0.0;
  for (double v : Ap.values) {
    const double av = std::abs(v);
    if (av > matrix_max_abs) matrix_max_abs = av;
  }
  const double eff_small =
      (matrix_max_abs > 0.0 && matrix_max_abs < ctrl.small_pivot)
          ? (ctrl.small_pivot * matrix_max_abs)
          : ctrl.small_pivot;

  for (Int s = 0; s < ns; ++s) {
    const std::size_t si = static_cast<std::size_t>(s);
    const Supernode &sn = keep.supernodes[si];
    const FrontalInfo &fi = keep.fronts[si];
    const Int p = sn.width();
    const Int f = fi.front_size();
    const Int ext = f - p;

    node_data[si].front_rows = fi.row_indices;
    if (p == 0)
      continue;

    AlignedArena::Marker mark = arena.save();
    FrontalMatrix front(f, p, fi.row_indices.data(), fi.row_indices.data(), arena);

    front.scatter_original(Ap.col_ptr, Ap.row_idx, Ap.values);

    std::vector<double> a22;
    if (ext > 0)
      a22.assign(static_cast<std::size_t>(ext) * static_cast<std::size_t>(ext), 0.0);

    std::vector<Int> prows_local;
    for (std::size_t k = 0; k < sn.children.size(); ++k) {
      const Int c = sn.children[k];
      const std::size_t ci = static_cast<std::size_t>(c);
      if (contrib[ci].empty())
        continue;

      const FrontalInfo &cfi = keep.fronts[ci];
      const Int cp = keep.supernodes[ci].width();
      const Int cext = cfi.front_size() - cp;
      if (cext <= 0)
        continue;

      const std::vector<Int> *prows_ptr = nullptr;
      if (k < fi.child_parent_rows.size() &&
          fi.child_parent_rows[k].size() == static_cast<std::size_t>(cext)) {
        prows_ptr = &fi.child_parent_rows[k];
      } else {
        prows_local.resize(static_cast<std::size_t>(cext));
        for (Int i = 0; i < cext; ++i) {
          const Int glob = cfi.row_indices[static_cast<std::size_t>(cp + i)];
          const auto it = std::lower_bound(fi.row_indices.cbegin(),
                                           fi.row_indices.cend(), glob);
          prows_local[static_cast<std::size_t>(i)] =
              static_cast<Int>(it - fi.row_indices.cbegin());
        }
        prows_ptr = &prows_local;
      }
      const std::vector<Int> &prows = *prows_ptr;

      front.assemble_contrib(contrib[ci].data(), cext, prows);

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
    }

    const int num_fs = static_cast<int>(p);
    int delayed_here = 0;
    factor_single_front(front, num_fs, ctrl, eff_small, node_data[si], inertia, delayed_here);

    info.delayed_pivots += delayed_here;
    inertia.zero += delayed_here;

    store_unit_lower_front(
        front,
        static_cast<int>(f),
        static_cast<int>(p),
        node_data[si],
        fkeep.factor_values.data() + static_cast<std::size_t>(fkeep.factor_col_ptr[si]));

    if (si < fkeep.pivot_types.size())
      fkeep.pivot_types[si] = node_data[si].pivot_tag;

    if (ext > 0 && sn.parent >= 0) {
      contrib[si] = std::move(a22);
      compute_front_contrib(front, p, ext, node_data[si], contrib[si]);
    }

    arena.reset_to(mark);
  }

  info.num_positive = inertia.positive;
  info.num_negative = inertia.negative;
  info.num_zero = inertia.zero;
  info.numerical_rank = static_cast<int>(keep.n) - inertia.zero;

  finalize_indef_factor_metadata(keep, node_data, fkeep);

  fkeep.status = (inertia.zero > 0) ? FactorStatus::Singular : FactorStatus::Success;
  info.factor_status = fkeep.status;
  info.status = (fkeep.status == FactorStatus::Success)
                    ? ErrorCode::Success
                    : ErrorCode::SingularMatrix;
  return fkeep.status;
}

} // namespace smf
