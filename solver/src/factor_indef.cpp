#include "smf/factor_indef.hpp"
#include "smf/blas_thread_guard.hpp"
#include "smf/blocked_indef.hpp"
#include "smf/multifrontal_serial.hpp"
#include "smf/numeric_front.hpp"
#include "smf/pivoting.hpp"
#include "smf/scaling.hpp"
#include "smf/threading.hpp"
#include "smf/workspace.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <numeric>
#include <optional>
#include <vector>

namespace smf {

namespace {

bool check_analysis_storage_bounds(const AnalysisKeep &keep) noexcept {
  LongInt total_predicted = 0;
  const std::size_t ns = keep.supernodes.size();
  if (keep.fronts.size() < ns) return false;
  for (std::size_t s = 0; s < ns; ++s) {
    const LongInt p = static_cast<LongInt>(keep.supernodes[s].width());
    const LongInt f = static_cast<LongInt>(keep.fronts[s].front_size());
    LongInt front_sq = 0;
    if (!checked_storage_elements(f, f, front_sq)) {
      return false;
    }
    LongInt panel_elems = 0;
    if (!checked_storage_elements(f, p, panel_elems)) {
      return false;
    }
    if (!checked_add_i64(total_predicted, panel_elems, total_predicted) ||
        total_predicted > kMaxFactorStorageEntries) {
      return false;
    }
  }
  return true;
}

#ifdef SMF_PARALLEL
// Per-node front build + factor for the task-parallel path.
FrontResult build_and_factor_supernode_front(
    Int s,
    const AnalysisKeep &keep,
    const CscLower &Ap,
    const Control &ctrl,
    double eff_small,
    std::vector<FrontResult> &node_results,
    NumericWorkspace *ws_opt) {
  const std::size_t si = static_cast<std::size_t>(s);
  const Supernode &sn = keep.supernodes[si];
  const FrontalInfo &fi = keep.fronts[si];
  const int p_sym = static_cast<int>(sn.width());
  const int f_sym = static_cast<int>(fi.front_size());

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

  LongInt front_sq = 0;
  if (!checked_storage_elements(static_cast<LongInt>(all_rows.size()),
                                static_cast<LongInt>(all_rows.size()),
                                front_sq)) {
    FrontResult err_res;
    err_res.status = ErrorCode::AllocationError;
    return err_res;
  }

  if (ws_opt != nullptr) {
    const std::size_t prev_cap = node_results[si].l_row_ids.capacity();
    if (all_rows.size() > prev_cap ||
        static_cast<std::size_t>(front_sq) > ws_opt->front_scratch.capacity()) {
      ++ws_opt->capacity_growths_last_call;
      ++ws_opt->front_heap_allocations_last_call;
      if (static_cast<std::size_t>(front_sq) > ws_opt->front_scratch.capacity()) {
        ws_opt->front_scratch.reserve(static_cast<std::size_t>(front_sq));
      }
    }
  }

  NumericFront front(std::move(all_rows), num_eligible);

  front.scatter_original_columns(Ap, sn.col_start, sn.col_end);

  for (Int c : sn.children) {
    auto &c_res = node_results[static_cast<std::size_t>(c)];
    if (!c_res.contrib.empty()) {
      front.assemble_contribution(c_res.contrib_row_ids, c_res.contrib);
      if (ws_opt != nullptr) {
        ws_opt->release_contrib(std::move(c_res.contrib));
      } else {
        std::vector<double>().swap(c_res.contrib);
      }
    }
  }

  const bool is_root = (sn.parent == -1);
  if (!ctrl.indef_use_reference_kernel &&
      ctrl.indef_panel_size > 1 &&
      front.num_rows() >= 16) {
    if (ws_opt != nullptr) {
      return factor_front_blocked(front, ctrl, *ws_opt, eff_small, is_root);
    }
    NumericWorkspace local_ws;
    return factor_front_blocked(front, ctrl, local_ws, eff_small, is_root);
  }
  return factor_front_reference(front, ctrl, eff_small, is_root);
}

#endif // SMF_PARALLEL

bool finalize_factor_from_front_results(
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

  LongInt offset = 0;
  for (int s = 0; s < ns; ++s) {
    const std::size_t si = static_cast<std::size_t>(s);
    const FrontResult &fr = node_results[si];
    fkeep.factor_col_ptr[si] = offset;
    if (!checked_add_i64(offset, static_cast<LongInt>(fr.l_panel.size()), offset) ||
        offset > kMaxFactorStorageEntries) {
      fkeep.factor_values.clear();
      return false;
    }
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
                fkeep.factor_values.begin() +
                    static_cast<std::ptrdiff_t>(fkeep.factor_col_ptr[si]));
    }
  }

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

  fkeep.diag_entries.clear();
  fkeep.inline_d = false;
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

  fkeep.solve_steps.clear();
  fkeep.solve_row_indices.clear();
  fkeep.solve_subtrees.clear();
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
  LongInt total_rows = 0;
  int max_p = 0;
  int max_q = 0;
  for (Int s : *order_ptr) {
    const auto &fr = node_results[static_cast<std::size_t>(s)];
    if (!checked_add_i64(total_rows, static_cast<LongInt>(fr.l_row_ids.size()), total_rows) ||
        total_rows > kMaxFactorStorageEntries) {
      fkeep.factor_values.clear();
      return false;
    }
    max_p = std::max(max_p, fr.num_accepted);
    max_q = std::max(max_q, std::max(0, fr.front_size - fr.num_accepted));
  }
  fkeep.solve_row_indices.reserve(static_cast<std::size_t>(total_rows));

  for (Int s : *order_ptr) {
    const std::size_t si = static_cast<std::size_t>(s);
    const auto &fr = node_results[si];
    SolveStep step;
    step.row_offset = static_cast<LongInt>(fkeep.solve_row_indices.size());
    step.factor_offset = fkeep.factor_col_ptr[si];
    step.p = fr.num_accepted;
    step.f = fr.front_size;
    fkeep.solve_steps.push_back(step);
    for (Int sym_r : fr.l_row_ids) {
      fkeep.solve_row_indices.push_back(
          static_cast<Int>(fkeep.num_iperm[static_cast<std::size_t>(sym_r)]));
    }
  }

  info.actual_factor_entries = static_cast<LongInt>(fkeep.factor_values.size());
  info.actual_flops = total_flops;
  if (max_realized_front > info.max_front_size) {
    info.max_front_size = max_realized_front;
  }
  return true;
}

} // namespace

FactorStatus factor_indef(const AnalysisKeep &keep, const CscLower &updated,
                          const Control &ctrl, Info &info,
                          FactorKeep &fkeep, NumericWorkspace &ws) {
  FactorValues values;
  if (!resolve_factor_values(keep, updated, values)) values.data = nullptr;
  return factor_indef(keep, values, ctrl, info, fkeep, ws);
}

FactorStatus factor_indef(const AnalysisKeep &keep, const FactorValues &values,
                          const Control &ctrl, Info &info,
                          FactorKeep &fkeep, NumericWorkspace &ws) {
  const auto t0 = std::chrono::steady_clock::now();
  const Int ns = static_cast<Int>(keep.supernodes.size());

  info.delayed_pivots = 0;
  ws.front_heap_allocations_last_call = 0;
  ws.capacity_growths_last_call = 0;

  if (!check_analysis_storage_bounds(keep)) {
    fkeep.factor_values.clear();
    fkeep.status = FactorStatus::Singular;
    info.factor_status = FactorStatus::Singular;
    info.status = ErrorCode::AllocationError;
    return FactorStatus::Singular;
  }

  const bool use_parallel_path =
#ifdef SMF_PARALLEL
      (ctrl.num_threads > 1 && ctrl.tree_parallel);
#else
      false;
#endif
  if (use_parallel_path) {
    ws.prepare_for_analysis(keep);
  }
  // ScalingMethod::Auto applies the analysis' matching scaling when it
  // computed one (saddle-point matrices under auto ordering), else none.
  ScalingMethod scaling = ctrl.scaling;
  const double *user_scale = ctrl.user_scale;
  if (scaling == ScalingMethod::Auto) {
    scaling = keep.matching_scale.empty() ? ScalingMethod::None
                                          : ScalingMethod::UserSupplied;
    user_scale = keep.matching_scale.data();
  }
  double matrix_max_abs = 0.0;
  const ErrorCode map_err = map_values_to_permuted(
      keep, values, ws, scaling != ScalingMethod::None, &matrix_max_abs);
  if (map_err != ErrorCode::Success) {
    fkeep.factor_values.clear();
    fkeep.status = FactorStatus::Singular;
    info.factor_status = FactorStatus::Singular;
    info.status = map_err;
    return FactorStatus::Singular;
  }

  fkeep.scale.clear();
  if (scaling != ScalingMethod::None && keep.n > 0) {
    // The Matching ordering already matched the analysed values; when they
    // are being factored unchanged, its scaling is the one to compute.
    const bool reuse_analysis_matching =
        scaling == ScalingMethod::Matching && !keep.matching_scale.empty() &&
        ws.clean_values == keep.cleaned.values;
    ErrorCode sc_err = ErrorCode::Success;
    if (reuse_analysis_matching) {
      fkeep.scale = keep.matching_scale;
    } else {
      CscLower clean_mat;
      clean_mat.n = keep.n;
      clean_mat.col_ptr = keep.cleaned.col_ptr;
      clean_mat.row_idx = keep.cleaned.row_idx;
      clean_mat.values = ws.clean_values;
      sc_err = compute_scaling_for_method(clean_mat, scaling, fkeep.scale,
                                          ctrl.print_level, user_scale);
    }
    if (sc_err != ErrorCode::Success) {
      fkeep.factor_values.clear();
      fkeep.status = FactorStatus::Singular;
      info.factor_status = FactorStatus::Singular;
      info.status = sc_err;
      return FactorStatus::Singular;
    }

    // Apply scale S * A * S to clean_values and rescatter into ws.perm_values
    for (Int j = 0; j < keep.n; ++j) {
      const double sj = fkeep.scale[static_cast<std::size_t>(j)];
      for (Int p = keep.cleaned.col_ptr[static_cast<std::size_t>(j)];
           p < keep.cleaned.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
        const Int i = keep.cleaned.row_idx[static_cast<std::size_t>(p)];
        ws.clean_values[static_cast<std::size_t>(p)] *=
            fkeep.scale[static_cast<std::size_t>(i)] * sj;
      }
    }
    if (keep.orig_to_perm_idx.empty()) {
      ws.perm_values = ws.clean_values;
    } else {
      std::fill(ws.perm_values.begin(), ws.perm_values.end(), 0.0);
      for (std::size_t k = 0; k < ws.clean_values.size(); ++k) {
        const Int p_slot = keep.orig_to_perm_idx[k];
        ws.perm_values[static_cast<std::size_t>(p_slot)] += ws.clean_values[k];
      }
    }
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
    const auto t1 = std::chrono::steady_clock::now();
    info.factor_seconds = std::chrono::duration<double>(t1 - t0).count();
    return FactorStatus::Success;
  }

  const std::vector<Int> &ap_col_ptr =
      keep.perm_col_ptr.empty() ? keep.cleaned.col_ptr : keep.perm_col_ptr;
  const std::vector<Int> &ap_row_idx =
      keep.perm_col_ptr.empty() ? keep.cleaned.row_idx : keep.perm_row_idx;

  if (scaling != ScalingMethod::None) {
    matrix_max_abs = 0.0;
    for (double v : ws.perm_values) matrix_max_abs = std::max(matrix_max_abs, std::abs(v));
  }
  const double eff_small =
      (matrix_max_abs > 0.0 && matrix_max_abs < ctrl.small_pivot)
          ? (ctrl.small_pivot * matrix_max_abs)
          : ctrl.small_pivot;

  if (!use_parallel_path) {
    // Threads go to BLAS inside the fronts.
    std::optional<BlasThreadGuard> blas_threads;
    if (ctrl.num_threads > 1)
      blas_threads.emplace(ctrl.num_threads);
    SerialFactorStats st;
    const double static_pivot =
        ctrl.static_pivot_tol > 0.0 ? ctrl.static_pivot_tol * matrix_max_abs : 0.0;
    const ErrorCode err = multifrontal_factor_serial(
        keep, ap_col_ptr.data(), ap_row_idx.data(), ws.perm_values.data(), ctrl,
        eff_small, static_pivot, /*posdef=*/false, fkeep, ws, st);
    blas_threads.reset();
    if (err != ErrorCode::Success) {
      std::fill(ws.row_pos.begin(), ws.row_pos.end(), Int{-1});
      fkeep.factor_values.clear();
      info.status = err;
      info.factor_status = FactorStatus::Singular;
      fkeep.status = FactorStatus::Singular;
      return FactorStatus::Singular;
    }
    info.delayed_pivots = st.delayed;
    info.static_pivots = st.static_pivots;
    info.parallel_subtrees = st.parallel_subtrees;
    info.num_positive = st.inertia.positive;
    info.num_negative = st.inertia.negative;
    info.num_zero = st.inertia.zero;
    info.numerical_rank = static_cast<int>(keep.n) - st.inertia.zero;
    info.actual_factor_entries = static_cast<LongInt>(fkeep.factor_values.size());
    info.actual_flops = st.flops;
    info.max_front_size = std::max(info.max_front_size, st.max_front);
    info.arena_growths = static_cast<int>(ws.capacity_growths_last_call);
    info.arena_peak_bytes = static_cast<long>(
        (ws.cb_values.capacity() + ws.kernel_work.capacity()) *
        sizeof(double));

    ++ws.warmup_calls;
    const auto t1 = std::chrono::steady_clock::now();
    info.factor_seconds = std::chrono::duration<double>(t1 - t0).count();
    fkeep.status =
        (st.inertia.zero > 0) ? FactorStatus::Singular : FactorStatus::Success;
    info.factor_status = fkeep.status;
    info.status = (fkeep.status == FactorStatus::Success)
                      ? ErrorCode::Success
                      : ErrorCode::SingularMatrix;
    return fkeep.status;
  }

  CscLower Ap;
  Ap.n = keep.n;
  Ap.col_ptr = ap_col_ptr;
  Ap.row_idx = ap_row_idx;
  Ap.values = ws.perm_values;

  std::atomic<int> alloc_or_input_err{0};

#ifdef SMF_PARALLEL
  {
    BlasSerialGuard blas_guard;
    auto process_node = [&](Int s) {
      if (alloc_or_input_err.load(std::memory_order_relaxed) != 0)
        return;
      FrontResult fr = build_and_factor_supernode_front(
          s, keep, Ap, ctrl, eff_small, ws.node_results, nullptr);
      if (fr.status == ErrorCode::AllocationError ||
          fr.status == ErrorCode::IllegalValue) {
        alloc_or_input_err.store(static_cast<int>(fr.status),
                                 std::memory_order_relaxed);
      }
      ws.node_results[static_cast<std::size_t>(s)] = std::move(fr);
    };
    ExecutionPolicy policy;
    policy.threads = ctrl.num_threads;
    policy.memory_budget_bytes = ctrl.memory_budget_bytes;
    policy.min_task_flops = ctrl.factor_parallel_min_flops;
    policy.deterministic = ctrl.deterministic;
    run_costed_postorder(keep.supernodes, keep.fronts, policy, process_node);
  }
#endif

  if (alloc_or_input_err.load() != 0) {
    fkeep.factor_values.clear();
    info.status = static_cast<ErrorCode>(alloc_or_input_err.load());
    info.factor_status = FactorStatus::Singular;
    fkeep.status = FactorStatus::Singular;
    return FactorStatus::Singular;
  }

  InertiaCounts total_inertia{};
  int total_delayed = 0;
  double total_flops = 0.0;
  int max_realized_front = 0;
  if (!finalize_factor_from_front_results(keep,
                                          ws.node_results,
                                          fkeep,
                                          info,
                                          total_inertia,
                                          total_delayed,
                                          total_flops,
                                          max_realized_front)) {
    fkeep.factor_values.clear();
    info.status = ErrorCode::AllocationError;
    info.factor_status = FactorStatus::Singular;
    fkeep.status = FactorStatus::Singular;
    return FactorStatus::Singular;
  }

  info.delayed_pivots = total_delayed;
  info.num_positive = total_inertia.positive;
  info.num_negative = total_inertia.negative;
  info.num_zero = total_inertia.zero;
  info.numerical_rank = static_cast<int>(keep.n) - total_inertia.zero;
  info.arena_growths = static_cast<int>(ws.capacity_growths_last_call);
  info.arena_peak_bytes = static_cast<long>(
      ws.front_scratch.capacity() * sizeof(double) + ws.peak_pool_bytes);

  ++ws.warmup_calls;
  const auto t1 = std::chrono::steady_clock::now();
  info.factor_seconds = std::chrono::duration<double>(t1 - t0).count();

  fkeep.status =
      (total_inertia.zero > 0) ? FactorStatus::Singular : FactorStatus::Success;
  info.factor_status = fkeep.status;
  info.status = (fkeep.status == FactorStatus::Success)
                    ? ErrorCode::Success
                    : ErrorCode::SingularMatrix;
  return fkeep.status;
}

FactorStatus factor_indef(const AnalysisKeep &keep, const Control &ctrl,
                          Info &info, FactorKeep &fkeep) {
  NumericWorkspace ws;
  return factor_indef(keep, keep.cleaned, ctrl, info, fkeep, ws);
}

} // namespace smf
