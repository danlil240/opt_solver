#include "smf/factor_indef.hpp"
#include "smf/blas_thread_guard.hpp"
#include "smf/numeric_front.hpp"
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

std::vector<double> scatter_values_indef(
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

FrontResult build_and_factor_supernode_front(
    Int s,
    const AnalysisKeep &keep,
    const CscLower &Ap,
    const Control &ctrl,
    double eff_small,
    std::vector<FrontResult> &node_results) {
  const std::size_t si = static_cast<std::size_t>(s);
  const Supernode &sn = keep.supernodes[si];
  const FrontalInfo &fi = keep.fronts[si];
  const int p_sym = static_cast<int>(sn.width());
  const int f_sym = static_cast<int>(fi.front_size());

  // 1. Collect eligible columns: child delayed columns + supernode's own symbolic columns
  std::vector<Int> eligible_rows;
  eligible_rows.reserve(static_cast<std::size_t>(p_sym + 8));
  for (Int c : sn.children) {
    const auto &c_res = node_results[static_cast<std::size_t>(c)];
    for (Int dcol : c_res.delayed_col_ids) {
      if (std::find(eligible_rows.begin(), eligible_rows.end(), dcol) ==
          eligible_rows.end()) {
        eligible_rows.push_back(dcol);
      }
    }
  }
  for (int k = 0; k < p_sym; ++k) {
    const Int r_id = fi.row_indices[static_cast<std::size_t>(k)];
    if (std::find(eligible_rows.begin(), eligible_rows.end(), r_id) ==
        eligible_rows.end()) {
      eligible_rows.push_back(r_id);
    }
  }
  const int num_eligible = static_cast<int>(eligible_rows.size());

  // 2. Collect extension rows: predicted extension rows + any child contribution rows not in eligible_rows
  std::vector<Int> ext_rows;
  ext_rows.reserve(static_cast<std::size_t>(std::max(0, f_sym - p_sym) + 8));
  for (int k = p_sym; k < f_sym; ++k) {
    const Int r_id = fi.row_indices[static_cast<std::size_t>(k)];
    if (std::find(eligible_rows.begin(), eligible_rows.end(), r_id) ==
        eligible_rows.end()) {
      ext_rows.push_back(r_id);
    }
  }
  for (Int c : sn.children) {
    const auto &c_res = node_results[static_cast<std::size_t>(c)];
    for (Int crow : c_res.contrib_row_ids) {
      if (std::find(eligible_rows.begin(), eligible_rows.end(), crow) ==
          eligible_rows.end()) {
        ext_rows.push_back(crow);
      }
    }
  }
  std::sort(ext_rows.begin(), ext_rows.end());
  ext_rows.erase(std::unique(ext_rows.begin(), ext_rows.end()), ext_rows.end());

  std::vector<Int> all_rows;
  all_rows.reserve(eligible_rows.size() + ext_rows.size());
  all_rows.insert(all_rows.end(), eligible_rows.begin(), eligible_rows.end());
  all_rows.insert(all_rows.end(), ext_rows.begin(), ext_rows.end());

  NumericFront front(std::move(all_rows), num_eligible);

  // 3. Scatter original columns owned by supernode s once
  front.scatter_original_columns(Ap, sn.col_start, sn.col_end);

  // 4. Assemble child contributions once and release child contribution buffers
  for (Int c : sn.children) {
    auto &c_res = node_results[static_cast<std::size_t>(c)];
    if (!c_res.contrib.empty()) {
      front.assemble_contribution(c_res.contrib_row_ids, c_res.contrib);
      std::vector<double>().swap(c_res.contrib);
    }
  }

  // 5. Factor the realized front
  const bool is_root = (sn.parent == -1);
  return factor_front_reference(front, ctrl, eff_small, is_root);
}

void finalize_factor_from_front_results(
    const AnalysisKeep &keep,
    const std::vector<FrontResult> &node_results,
    FactorKeep &fkeep,
    Info &info,
    InertiaCounts &total_inertia,
    int &total_delayed,
    double &total_flops,
    int &max_realized_front) {
  const int n = static_cast<int>(keep.n);
  const int ns = static_cast<int>(keep.supernodes.size());

  total_inertia = InertiaCounts{};
  total_delayed = 0;
  total_flops = 0.0;
  max_realized_front = 0;

  fkeep.n = keep.n;
  fkeep.factor_col_ptr.assign(static_cast<std::size_t>(ns) + 1, 0);
  fkeep.pivot_types.assign(static_cast<std::size_t>(ns), {});

  int offset = 0;
  for (int s = 0; s < ns; ++s) {
    const std::size_t si = static_cast<std::size_t>(s);
    const FrontResult &fr = node_results[si];
    fkeep.factor_col_ptr[si] = offset;
    offset += static_cast<int>(fr.l_panel.size());
    fkeep.pivot_types[si] = fr.pivot_tags;

    total_inertia.positive += fr.inertia.positive;
    total_inertia.negative += fr.inertia.negative;
    total_inertia.zero += fr.inertia.zero;
    total_delayed += fr.delayed_count;
    total_flops += fr.flops;
    if (fr.front_size > max_realized_front) {
      max_realized_front = fr.front_size;
    }
  }
  fkeep.factor_col_ptr[static_cast<std::size_t>(ns)] = offset;
  fkeep.factor_values.assign(static_cast<std::size_t>(offset), 0.0);
  for (int s = 0; s < ns; ++s) {
    const std::size_t si = static_cast<std::size_t>(s);
    const auto &panel = node_results[si].l_panel;
    if (!panel.empty()) {
      std::copy(panel.begin(),
                panel.end(),
                fkeep.factor_values.begin() + fkeep.factor_col_ptr[si]);
    }
  }

  // Build realized numerical permutation num_perm[new_piv] = sym_idx
  fkeep.num_perm.clear();
  fkeep.num_perm.reserve(static_cast<std::size_t>(n));
  std::vector<uint8_t> seen(static_cast<std::size_t>(n), 0);

  for (int s = 0; s < ns; ++s) {
    for (Int sym_r : node_results[static_cast<std::size_t>(s)].accepted_row_ids) {
      if (sym_r >= 0 && sym_r < n && !seen[static_cast<std::size_t>(sym_r)]) {
        seen[static_cast<std::size_t>(sym_r)] = 1;
        fkeep.num_perm.push_back(static_cast<int>(sym_r));
      }
    }
  }
  for (int sym_r = 0; sym_r < n; ++sym_r) {
    if (!seen[static_cast<std::size_t>(sym_r)]) {
      fkeep.num_perm.push_back(sym_r);
    }
  }

  fkeep.num_iperm.assign(static_cast<std::size_t>(n), 0);
  for (int new_piv = 0; new_piv < n; ++new_piv) {
    const int sym_idx = fkeep.num_perm[static_cast<std::size_t>(new_piv)];
    fkeep.num_iperm[static_cast<std::size_t>(sym_idx)] = new_piv;
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

  // Build D blocks exclusively in fkeep.diag_entries using realized pivot indices
  fkeep.diag_entries.clear();
  fkeep.diag_entries.reserve(static_cast<std::size_t>(n));
  for (int s = 0; s < ns; ++s) {
    for (const auto &db : node_results[static_cast<std::size_t>(s)].d_blocks) {
      DiagSolveEntry entry;
      entry.tag = db.tag;
      entry.row0 = static_cast<Int>(
          fkeep.num_iperm[static_cast<std::size_t>(db.row0)]);
      entry.row1 = (db.tag == 2)
                       ? static_cast<Int>(
                             fkeep.num_iperm[static_cast<std::size_t>(db.row1)])
                       : 0;
      entry.d00 = db.d00;
      entry.d10 = db.d10;
      entry.d11 = db.d11;
      fkeep.diag_entries.push_back(entry);
    }
  }

  // Build solve_steps and solve_row_indices from realized front layouts
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
  std::size_t total_rows = 0;
  int max_p = 0;
  int max_q = 0;
  for (Int s : *order_ptr) {
    const auto &fr = node_results[static_cast<std::size_t>(s)];
    total_rows += fr.l_row_ids.size();
    max_p = std::max(max_p, fr.num_accepted);
    max_q = std::max(max_q, std::max(0, fr.front_size - fr.num_accepted));
  }
  fkeep.solve_row_indices.reserve(total_rows);

  for (Int s : *order_ptr) {
    const std::size_t si = static_cast<std::size_t>(s);
    const auto &fr = node_results[si];
    SolveStep step;
    step.row_offset = static_cast<Int>(fkeep.solve_row_indices.size());
    step.factor_offset = fkeep.factor_col_ptr[si];
    step.p = fr.num_accepted;
    step.f = fr.front_size;
    fkeep.solve_steps.push_back(step);
    for (Int sym_r : fr.l_row_ids) {
      fkeep.solve_row_indices.push_back(
          static_cast<Int>(fkeep.num_iperm[static_cast<std::size_t>(sym_r)]));
    }
  }

  fkeep.solve_tmp.resize(static_cast<std::size_t>(keep.n));
  fkeep.solve_loc.resize(static_cast<std::size_t>(max_p));
  fkeep.solve_ext.resize(static_cast<std::size_t>(max_q));

  info.actual_factor_entries = static_cast<LongInt>(fkeep.factor_values.size());
  info.actual_flops = total_flops;
  if (max_realized_front > info.max_front_size) {
    info.max_front_size = max_realized_front;
  }
}

} // namespace

FactorStatus factor_indef(const AnalysisKeep &keep, const Control &ctrl,
                          Info &info, FactorKeep &fkeep) {
  const Int ns = static_cast<Int>(keep.supernodes.size());

  info.delayed_pivots = 0;

  fkeep.n = keep.n;
  fkeep.perm.assign(keep.perm.begin(), keep.perm.end());
  fkeep.iperm.assign(keep.iperm.begin(), keep.iperm.end());
  fkeep.num_perm.resize(static_cast<std::size_t>(keep.n));
  std::iota(fkeep.num_perm.begin(), fkeep.num_perm.end(), 0);
  fkeep.num_iperm = fkeep.num_perm;
  fkeep.analysis = &keep;
  fkeep.is_posdef = false;
  fkeep.status = FactorStatus::Success;
  fkeep.factor_col_ptr.assign(static_cast<std::size_t>(ns) + 1, 0);
  fkeep.factor_values.clear();
  fkeep.pivot_types.assign(static_cast<std::size_t>(ns), {});

  if (ns == 0 || keep.n == 0) {
    info.num_positive = 0;
    info.num_negative = 0;
    info.num_zero = 0;
    info.numerical_rank = 0;
    info.factor_status = FactorStatus::Success;
    info.status = ErrorCode::Success;
    return FactorStatus::Success;
  }

  CscLower Ap;
  Ap.n = keep.n;
  if (keep.orig_to_perm_idx.empty()) {
    Ap.col_ptr = keep.cleaned.col_ptr;
    Ap.row_idx = keep.cleaned.row_idx;
    Ap.values = keep.cleaned.values;
  } else {
    const Int nnz_perm = keep.perm_col_ptr[static_cast<std::size_t>(keep.n)];
    std::vector<double> Ap_values =
        scatter_values_indef(keep.cleaned, keep.orig_to_perm_idx, nnz_perm);
    Ap.col_ptr = keep.perm_col_ptr;
    Ap.row_idx = keep.perm_row_idx;
    Ap.values = std::move(Ap_values);
  }

  double matrix_max_abs = 0.0;
  for (double v : Ap.values) {
    if (!std::isfinite(v)) {
      fkeep.status = FactorStatus::Singular;
      info.factor_status = FactorStatus::Singular;
      info.status = ErrorCode::IllegalValue;
      return FactorStatus::Singular;
    }
    const double av = std::abs(v);
    if (av > matrix_max_abs) matrix_max_abs = av;
  }
  const double eff_small =
      (matrix_max_abs > 0.0 && matrix_max_abs < ctrl.small_pivot)
          ? (ctrl.small_pivot * matrix_max_abs)
          : ctrl.small_pivot;

  std::vector<FrontResult> node_results(static_cast<std::size_t>(ns));
  std::atomic<int> alloc_or_input_err{0};

#ifdef SMF_PARALLEL
  if (ctrl.num_threads > 1) {
    BlasSerialGuard blas_guard;
    auto process_node = [&](Int s) {
      if (alloc_or_input_err.load(std::memory_order_relaxed) != 0)
        return;
      FrontResult fr = build_and_factor_supernode_front(
          s, keep, Ap, ctrl, eff_small, node_results);
      if (fr.status == ErrorCode::AllocationError ||
          fr.status == ErrorCode::IllegalValue) {
        alloc_or_input_err.store(static_cast<int>(fr.status),
                                 std::memory_order_relaxed);
      }
      node_results[static_cast<std::size_t>(s)] = std::move(fr);
    };
    run_parallel_postorder(keep.supernodes, process_node, ctrl.num_threads);
  } else
#endif
  {
    const std::vector<Int> *postorder_ptr = &keep.postorder;
    std::vector<Int> fallback_po;
    if (postorder_ptr->empty()) {
      fallback_po.resize(static_cast<std::size_t>(ns));
      std::iota(fallback_po.begin(), fallback_po.end(), 0);
      postorder_ptr = &fallback_po;
    }
    for (Int s : *postorder_ptr) {
      FrontResult fr = build_and_factor_supernode_front(
          s, keep, Ap, ctrl, eff_small, node_results);
      if (fr.status == ErrorCode::AllocationError ||
          fr.status == ErrorCode::IllegalValue) {
        info.status = fr.status;
        info.factor_status = FactorStatus::Singular;
        fkeep.status = FactorStatus::Singular;
        return FactorStatus::Singular;
      }
      node_results[static_cast<std::size_t>(s)] = std::move(fr);
    }
  }

  if (alloc_or_input_err.load() != 0) {
    info.status = static_cast<ErrorCode>(alloc_or_input_err.load());
    info.factor_status = FactorStatus::Singular;
    fkeep.status = FactorStatus::Singular;
    return FactorStatus::Singular;
  }

  InertiaCounts total_inertia{};
  int total_delayed = 0;
  double total_flops = 0.0;
  int max_realized_front = 0;
  finalize_factor_from_front_results(keep,
                                     node_results,
                                     fkeep,
                                     info,
                                     total_inertia,
                                     total_delayed,
                                     total_flops,
                                     max_realized_front);

  info.delayed_pivots = total_delayed;
  info.num_positive = total_inertia.positive;
  info.num_negative = total_inertia.negative;
  info.num_zero = total_inertia.zero;
  info.numerical_rank = static_cast<int>(keep.n) - total_inertia.zero;

  fkeep.status =
      (total_inertia.zero > 0) ? FactorStatus::Singular : FactorStatus::Success;
  info.factor_status = fkeep.status;
  info.status = (fkeep.status == FactorStatus::Success)
                    ? ErrorCode::Success
                    : ErrorCode::SingularMatrix;
  return fkeep.status;
}

} // namespace smf
