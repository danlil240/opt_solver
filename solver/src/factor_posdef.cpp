#include "smf/factor_posdef.hpp"
#include "smf/dense_kernel.hpp"
#include "smf/factor_stack.hpp"
#include "smf/frontal_matrix.hpp"
#include "smf/multifrontal_serial.hpp"
#include "smf/blas_thread_guard.hpp"
#include "smf/scaling.hpp"
#include "smf/threading.hpp"
#include "smf/workspace.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <numeric>
#include <optional>

namespace smf {

namespace {

static Int compute_lower_bandwidth(Int n, const Int *col_ptr, const Int *row_idx) {
  Int kd = 0;
  for (Int j = 0; j < n; ++j) {
    for (Int p = col_ptr[j]; p < col_ptr[j + 1]; ++p) {
      const Int d = row_idx[p] - j; // lower CSC => i >= j
      if (d > kd)
        kd = d;
    }
  }
  return kd;
}

static bool try_factor_banded_spd(Int n, const Int *col_ptr, const Int *row_idx,
                                  const double *values, FactorKeep &fkeep) {
  if (n < 100)
    return false;
  const Int kd = compute_lower_bandwidth(n, col_ptr, row_idx);
  if (kd <= 0 || kd > 128)
    return false;

  const int n_i = static_cast<int>(n);
  const int kd_i = static_cast<int>(kd);
  const int ldab = kd_i + 1;
  std::vector<double> ab(static_cast<std::size_t>(ldab) *
                             static_cast<std::size_t>(n_i),
                         0.0);

  for (Int j = 0; j < n; ++j) {
    for (Int p = col_ptr[j]; p < col_ptr[j + 1]; ++p) {
      const Int d = row_idx[p] - j;
      ab[static_cast<std::size_t>(d) +
         static_cast<std::size_t>(j) * static_cast<std::size_t>(ldab)] +=
          values[p];
    }
  }

  const int info = smf_dpbtrf_lower(ab.data(), n_i, kd_i, ldab);
  if (info != 0)
    return false;

  fkeep.use_banded_spd = true;
  fkeep.band_n = n_i;
  fkeep.band_kd = kd_i;
  fkeep.band_factor = std::move(ab);
  return true;
}

#ifdef SMF_PARALLEL
static void build_solve_steps(const AnalysisKeep &keep, FactorKeep &fkeep) {
  fkeep.solve_steps.clear();
  fkeep.solve_row_indices.clear();
  fkeep.solve_steps.reserve(keep.solve_postorder.size());

  std::size_t row_count = 0;
  int max_p = 0;
  int max_q = 0;
  for (Int s : keep.solve_postorder)
  {
    const std::size_t si = static_cast<std::size_t>(s);
    const int p = static_cast<int>(keep.supernodes[si].width());
    const int f = static_cast<int>(keep.fronts[si].front_size());
    row_count += keep.fronts[si].row_indices.size();
    max_p = std::max(max_p, p);
    max_q = std::max(max_q, f - p);
  }
  fkeep.solve_row_indices.reserve(row_count);

  for (Int s : keep.solve_postorder) {
    const std::size_t si = static_cast<std::size_t>(s);
    const FrontalInfo &fi = keep.fronts[si];
    const Supernode &sn = keep.supernodes[si];
    SolveStep step;
    step.row_offset = static_cast<LongInt>(fkeep.solve_row_indices.size());
    step.factor_offset = fkeep.factor_col_ptr[si];
    step.p = static_cast<int>(sn.width());
    step.f = static_cast<int>(fi.front_size());
    fkeep.solve_steps.push_back(step);
    fkeep.solve_row_indices.insert(fkeep.solve_row_indices.end(),
                                   fi.row_indices.begin(), fi.row_indices.end());
  }

  fkeep.solve_tmp.resize(static_cast<std::size_t>(keep.n));
  fkeep.solve_loc.resize(static_cast<std::size_t>(max_p));
  fkeep.solve_ext.resize(static_cast<std::size_t>(max_q));
}

#endif // SMF_PARALLEL

} // anonymous namespace

// ---------------------------------------------------------------------------
// Parallel SPD factorization helper (only compiled when SMF_PARALLEL defined)
// ---------------------------------------------------------------------------
#ifdef SMF_PARALLEL
static FactorStatus factor_posdef_parallel(const AnalysisKeep &keep,
                                           const Control &ctrl,
                                           FactorKeep &fkeep,
                                           const CscLower &Ap) {
  const Int ns = static_cast<Int>(keep.supernodes.size());

  // ---- Step 1: Pre-allocate factor storage (all sizes known from analysis) --
  // Each supernode s contributes front_size(s) × width(s) factor values.
  fkeep.factor_col_ptr.resize(static_cast<std::size_t>(ns) + 1, 0);
  {
    LongInt offset = 0;
    for (Int s = 0; s < ns; ++s) {
      fkeep.factor_col_ptr[static_cast<std::size_t>(s)] = offset;
      const LongInt f = keep.fronts[static_cast<std::size_t>(s)].front_size();
      const LongInt p = keep.supernodes[static_cast<std::size_t>(s)].width();
      offset += f * p;
    }
    fkeep.factor_col_ptr[static_cast<std::size_t>(ns)] = offset;
    fkeep.factor_values.assign(static_cast<std::size_t>(offset), 0.0);
  }

  // ---- Step 2: Per-node contribution blocks (q×q, column-major lower tri) --
  // Each element is exclusively written by its own task and read by the
  // parent task (which begins only after all children tasks finish).
  std::vector<std::vector<double>> contrib(static_cast<std::size_t>(ns));

  // ---- Step 3: Shared atomic error flag -----------------------------------
  // 0 = success so far; 1 = NotPositiveDefinite detected
  std::atomic<int> err_flag{0};

  // ---- Step 4: Per-node callback ------------------------------------------
  auto process_node = [&](Int s) {
    // Skip this node if another subtree already failed.
    if (err_flag.load(std::memory_order_relaxed) != 0)
      return;

    const std::size_t si = static_cast<std::size_t>(s);
    const FrontalInfo &fi = keep.fronts[si];
    const Supernode &sn = keep.supernodes[si];
    const Int p = sn.width();
    const Int f = fi.front_size();
    const Int q = f - p;

    if (p <= 0)
      return;

    // Per-task arena — not shared with other concurrent tasks.
    const std::size_t arena_sz =
        (static_cast<std::size_t>(f) * static_cast<std::size_t>(f) * 2 +
         static_cast<std::size_t>(q) * static_cast<std::size_t>(q)) *
            sizeof(double) +
        8192u;
    AlignedArena arena(arena_sz);

    FrontalMatrix F(f, p, fi.row_indices.data(), fi.row_indices.data(), arena);

    double *a22 = nullptr;
    if (q > 0) {
      const std::size_t nbytes =
          static_cast<std::size_t>(q) * static_cast<std::size_t>(q) *
          sizeof(double);
      a22 = static_cast<double *>(arena.allocate(nbytes));
      std::memset(a22, 0, nbytes);
    }

    // Scatter original A entries (read-only from Ap).
    F.scatter_original(Ap.col_ptr, Ap.row_idx, Ap.values);

    // Assemble child contributions (children are fully done by taskwait).
    // Children in sn.children are pre-sorted; child_parent_rows may be
    // precomputed or built on demand for large-n analyses.
    std::vector<Int> parent_rows_local;
    for (std::size_t k = 0; k < sn.children.size(); ++k) {
      const Int c = sn.children[k];
      const std::size_t ci = static_cast<std::size_t>(c);
      if (contrib[ci].empty())
        continue;

      const FrontalInfo &fi_c = keep.fronts[ci];
      const Int p_c = keep.supernodes[ci].width();
      const Int f_c = fi_c.front_size();
      const Int q_c = f_c - p_c;
      if (q_c <= 0)
        continue;

      const double *cc = contrib[ci].data();
      const std::vector<Int> *parent_rows_ptr = nullptr;
      if (k < fi.child_parent_rows.size() &&
          fi.child_parent_rows[k].size() == static_cast<std::size_t>(q_c)) {
        parent_rows_ptr = &fi.child_parent_rows[k];
      } else {
        parent_rows_local.resize(static_cast<std::size_t>(q_c));
        for (Int ii = 0; ii < q_c; ++ii) {
          const Int ext_row =
              fi_c.row_indices[static_cast<std::size_t>(p_c + ii)];
          const auto it = std::lower_bound(fi.row_indices.begin(),
                                           fi.row_indices.end(), ext_row);
          parent_rows_local[static_cast<std::size_t>(ii)] =
              static_cast<Int>(it - fi.row_indices.begin());
        }
        parent_rows_ptr = &parent_rows_local;
      }
      const std::vector<Int> &parent_rows = *parent_rows_ptr;

      // Assemble A11/A21 parts into F.
      F.assemble_contrib(cc, q_c, parent_rows);

      // Assemble A22 parts where both row and col map to parent extended rows.
      if (q > 0) {
        for (Int jj = 0; jj < q_c; ++jj) {
          const Int pr_j = parent_rows[static_cast<std::size_t>(jj)];
          if (pr_j < p)
            continue;
          const Int lcol = pr_j - p;
          for (Int ii = jj; ii < q_c; ++ii) {
            const Int pr_i = parent_rows[static_cast<std::size_t>(ii)];
            if (pr_i < p)
              continue;
            const Int lrow = pr_i - p;
            a22[static_cast<std::size_t>(lcol) *
                    static_cast<std::size_t>(q) +
                static_cast<std::size_t>(lrow)] +=
                cc[static_cast<std::size_t>(jj) *
                       static_cast<std::size_t>(q_c) +
                   static_cast<std::size_t>(ii)];
          }
        }
      }

      // Release child's contribution block — child is done, parent is done
      // reading; no other task will touch contrib[c].
      contrib[ci].clear();
    }

    // Cholesky factorization of the p×p pivot block.
    const int ret =
        smf_dpotrf_lower(F.data(), static_cast<int>(p), static_cast<int>(f));
    if (ret != 0) {
      int expected = 0;
      err_flag.compare_exchange_strong(expected, 1,
                                       std::memory_order_relaxed);
      return;
    }

    // Compute L21 block via triangular solve: A21 × L11^{-T} = L21.
    if (q > 0 && p > 0) {
      smf_dtrsm_right_lower_transpose(
          F.data() + static_cast<std::ptrdiff_t>(p), static_cast<int>(q),
          static_cast<int>(p), F.data(), static_cast<int>(f),
          static_cast<int>(f));
    }

    // Schur complement: a22 -= L21 × L21^T.
    if (q > 0 && p > 0) {
      smf_dsyrk_lower(a22, static_cast<int>(q),
                      F.data() + static_cast<std::ptrdiff_t>(p),
                      static_cast<int>(p), static_cast<int>(f),
                      static_cast<int>(q));
    }

    // Write factor columns to pre-allocated fkeep.factor_values.
    // Range [factor_col_ptr[s], factor_col_ptr[s+1]) is exclusive to this task.
    const std::size_t base =
        static_cast<std::size_t>(fkeep.factor_col_ptr[si]);
    const std::size_t sz =
        static_cast<std::size_t>(f) * static_cast<std::size_t>(p);
    std::memcpy(fkeep.factor_values.data() + base, F.data(),
                sz * sizeof(double));

    // Store contribution block for the parent.
    if (q > 0) {
      const std::size_t q2 =
          static_cast<std::size_t>(q) * static_cast<std::size_t>(q);
      contrib[si].assign(a22, a22 + q2);
    }
  };

  // ---- Step 5: Parallel postorder traversal --------------------------------
  run_parallel_postorder(keep.supernodes, process_node, ctrl.num_threads);

  if (err_flag.load() != 0)
    return FactorStatus::NotPositiveDefinite;

  return FactorStatus::Success;
}
#endif // SMF_PARALLEL

FactorStatus factor_posdef(const AnalysisKeep &keep, const Control &ctrl,
                           Info &info, FactorKeep &fkeep) {
  NumericWorkspace ws;
  return factor_posdef(keep, keep.cleaned, ctrl, info, fkeep, ws);
}

FactorStatus factor_posdef(const AnalysisKeep &keep, const CscLower &updated,
                           const Control &ctrl, Info &info,
                           FactorKeep &fkeep, NumericWorkspace &ws) {
  const auto t0 = std::chrono::steady_clock::now();
  info.factor_status = FactorStatus::Success;
  ws.front_heap_allocations_last_call = 0;
  ws.capacity_growths_last_call = 0;

  auto fail = [&](FactorStatus fs, ErrorCode ec) {
    fkeep.factor_values.clear();
    fkeep.status = fs;
    info.factor_status = fs;
    info.status = ec;
    const auto t1 = std::chrono::steady_clock::now();
    info.factor_seconds = std::chrono::duration<double>(t1 - t0).count();
    return fs;
  };

  // ScalingMethod::Auto: the analysis computes no matching for SPD matrices.
  const ScalingMethod scaling =
      ctrl.scaling == ScalingMethod::Auto ? ScalingMethod::None : ctrl.scaling;
  const ErrorCode err = map_updated_values_to_permuted(
      keep, updated, ws, scaling != ScalingMethod::None);
  if (err != ErrorCode::Success)
    return fail(FactorStatus::NotPositiveDefinite, err);

  const Int ns = static_cast<Int>(keep.supernodes.size());
  const Int n = keep.n;

  fkeep.n = n;
  fkeep.status = FactorStatus::Success;
  fkeep.is_posdef = true;
  fkeep.use_banded_spd = false;
  fkeep.band_n = 0;
  fkeep.band_kd = 0;
  fkeep.band_factor.clear();
  fkeep.scale.clear();
  fkeep.perm.assign(keep.perm.begin(), keep.perm.end());
  fkeep.iperm.assign(keep.iperm.begin(), keep.iperm.end());
  fkeep.num_perm.resize(static_cast<std::size_t>(n));
  std::iota(fkeep.num_perm.begin(), fkeep.num_perm.end(), 0);
  fkeep.num_iperm = fkeep.num_perm;
  fkeep.analysis = &keep;

  if (ns == 0 || n == 0) {
    fkeep.factor_values.clear();
    fkeep.factor_col_ptr.assign(static_cast<std::size_t>(ns) + 1, 0);
    const auto t1 = std::chrono::steady_clock::now();
    info.factor_seconds = std::chrono::duration<double>(t1 - t0).count();
    return FactorStatus::Success;
  }

  if (scaling != ScalingMethod::None) {
    CscLower clean_mat;
    clean_mat.n = n;
    clean_mat.col_ptr = keep.cleaned.col_ptr;
    clean_mat.row_idx = keep.cleaned.row_idx;
    clean_mat.values = ws.clean_values;
    const ErrorCode sc_err = compute_scaling_for_method(
        clean_mat, scaling, fkeep.scale, ctrl.print_level, ctrl.user_scale);
    if (sc_err != ErrorCode::Success)
      return fail(FactorStatus::NotPositiveDefinite, sc_err);
    apply_scale_to_matrix(clean_mat, fkeep.scale);
    if (keep.orig_to_perm_idx.empty()) {
      ws.perm_values = clean_mat.values;
    } else {
      std::fill(ws.perm_values.begin(), ws.perm_values.end(), 0.0);
      for (std::size_t k = 0; k < clean_mat.values.size(); ++k)
        ws.perm_values[static_cast<std::size_t>(keep.orig_to_perm_idx[k])] +=
            clean_mat.values[k];
    }
  }

  const std::vector<Int> &ap_col_ptr =
      keep.perm_col_ptr.empty() ? keep.cleaned.col_ptr : keep.perm_col_ptr;
  const std::vector<Int> &ap_row_idx =
      keep.perm_col_ptr.empty() ? keep.cleaned.row_idx : keep.perm_row_idx;

  // Banded SPD fast path for auto-parallel benchmark/control settings.
  // This bypasses multifrontal assembly when the matrix is sufficiently banded.
  if (ctrl.ordering == OrderingMethod::AutoParallel &&
      try_factor_banded_spd(n, ap_col_ptr.data(), ap_row_idx.data(),
                            ws.perm_values.data(), fkeep)) {
    info.actual_factor_entries = static_cast<LongInt>(fkeep.band_factor.size());
    const auto t1 = std::chrono::steady_clock::now();
    info.factor_seconds = std::chrono::duration<double>(t1 - t0).count();
    return FactorStatus::Success;
  }

#ifdef SMF_PARALLEL
  if (ctrl.num_threads > 1 && ctrl.tree_parallel) {
    CscLower Ap;
    Ap.n = n;
    Ap.col_ptr = ap_col_ptr;
    Ap.row_idx = ap_row_idx;
    Ap.values = ws.perm_values;
    BlasSerialGuard blas_guard;
    const FactorStatus ps = factor_posdef_parallel(keep, ctrl, fkeep, Ap);
    if (ps != FactorStatus::Success)
      return fail(ps, ErrorCode::NotPositiveDefinite);
    info.actual_factor_entries =
        static_cast<LongInt>(fkeep.factor_values.size());
    build_solve_steps(keep, fkeep);
    const auto t1 = std::chrono::steady_clock::now();
    info.factor_seconds = std::chrono::duration<double>(t1 - t0).count();
    return FactorStatus::Success;
  }
#endif // SMF_PARALLEL

  // Threads go to BLAS inside the fronts.
  std::optional<BlasThreadGuard> blas_threads;
  if (ctrl.num_threads > 1)
    blas_threads.emplace(ctrl.num_threads);
  SerialFactorStats st;
  const ErrorCode serr = multifrontal_factor_serial(
      keep, ap_col_ptr.data(), ap_row_idx.data(), ws.perm_values.data(), ctrl,
      0.0, 0.0, /*posdef=*/true, fkeep, ws, st);
  blas_threads.reset();
  if (serr != ErrorCode::Success)
    return fail(FactorStatus::NotPositiveDefinite, serr);

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
  return FactorStatus::Success;
}

} // namespace smf
