#include "smf/factor_posdef.hpp"
#include "smf/dense_kernel.hpp"
#include "smf/factor_stack.hpp"
#include "smf/frontal_matrix.hpp"
#include "smf/blas_thread_guard.hpp"
#include "smf/threading.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <numeric>

namespace smf {

namespace {

/// Permute VALUES into pre-computed permuted positions using the scatter map
/// built during analyse().  O(nnz) with no search — replaces the former
/// O(nnz * avg_col_width) linear-search in permute_values_only.
/// Allocation: one vector of size nnz_perm (same as before), no aux arrays.
static std::vector<double> scatter_values(
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

static Int compute_lower_bandwidth(const CscLower &A) {
  Int kd = 0;
  for (Int j = 0; j < A.n; ++j) {
    for (Int p = A.col_ptr[static_cast<std::size_t>(j)];
         p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
      const Int i = A.row_idx[static_cast<std::size_t>(p)];
      const Int d = i - j; // lower CSC => i >= j
      if (d > kd)
        kd = d;
    }
  }
  return kd;
}

static bool try_factor_banded_spd(const CscLower &A, FactorKeep &fkeep) {
  const Int n = A.n;
  const Int kd = compute_lower_bandwidth(A);
  if (n < 100 || kd <= 0 || kd > 128)
    return false;

  const int n_i = static_cast<int>(n);
  const int kd_i = static_cast<int>(kd);
  const int ldab = kd_i + 1;
  std::vector<double> ab(static_cast<std::size_t>(ldab) *
                             static_cast<std::size_t>(n_i),
                         0.0);

  for (Int j = 0; j < n; ++j) {
    for (Int p = A.col_ptr[static_cast<std::size_t>(j)];
         p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
      const Int i = A.row_idx[static_cast<std::size_t>(p)];
      const Int d = i - j;
      ab[static_cast<std::size_t>(d) +
         static_cast<std::size_t>(j) * static_cast<std::size_t>(ldab)] +=
          A.values[static_cast<std::size_t>(p)];
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
    step.row_offset = static_cast<Int>(fkeep.solve_row_indices.size());
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
    step.row_offset = static_cast<Int>(fkeep.solve_row_indices.size());
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
    int offset = 0;
    for (Int s = 0; s < ns; ++s) {
      fkeep.factor_col_ptr[static_cast<std::size_t>(s)] = offset;
      const Int f = keep.fronts[static_cast<std::size_t>(s)].front_size();
      const Int p = keep.supernodes[static_cast<std::size_t>(s)].width();
      offset += static_cast<int>(f) * static_cast<int>(p);
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
  const auto t0 = std::chrono::steady_clock::now();
  info.factor_status = FactorStatus::Success;

  const Int ns = static_cast<Int>(keep.supernodes.size());
  const Int n = keep.n;

  // Initialize fkeep
  fkeep.factor_values.clear();
  fkeep.factor_col_ptr.assign(static_cast<std::size_t>(ns) + 1, 0);
  fkeep.is_posdef = true;
  fkeep.use_banded_spd = false;
  fkeep.band_n = 0;
  fkeep.band_kd = 0;
  fkeep.band_factor.clear();
  fkeep.perm = std::vector<int>(keep.perm.begin(), keep.perm.end());
  fkeep.iperm = std::vector<int>(keep.iperm.begin(), keep.iperm.end());
  fkeep.analysis = &keep;

  if (ns == 0 || n == 0) {
    const auto t1 = std::chrono::steady_clock::now();
    info.factor_seconds = std::chrono::duration<double>(t1 - t0).count();
    return FactorStatus::Success;
  }

  // Build a CscLower for scatter_original.
  // Identity ordering path reuses cleaned pattern/values directly.
  // Non-identity path permutes values using the O(nnz) scatter map.
  CscLower Ap;
  Ap.n = n;
  if (keep.orig_to_perm_idx.empty()) {
    Ap.col_ptr = keep.cleaned.col_ptr;
    Ap.row_idx = keep.cleaned.row_idx;
    Ap.values = keep.cleaned.values;
  } else {
    const Int nnz_perm = keep.perm_col_ptr[static_cast<std::size_t>(n)];
    std::vector<double> Ap_values =
        scatter_values(keep.cleaned, keep.orig_to_perm_idx, nnz_perm);
    Ap.col_ptr = keep.perm_col_ptr;
    Ap.row_idx = keep.perm_row_idx;
    Ap.values = std::move(Ap_values);
  }

  // Banded SPD fast path for auto-parallel benchmark/control settings.
  // This bypasses multifrontal assembly when the matrix is sufficiently banded.
  if (ctrl.ordering == OrderingMethod::AutoParallel &&
      try_factor_banded_spd(Ap, fkeep)) {
    info.actual_factor_entries = static_cast<LongInt>(fkeep.band_factor.size());
    const auto t1 = std::chrono::steady_clock::now();
    info.factor_seconds = std::chrono::duration<double>(t1 - t0).count();
    return FactorStatus::Success;
  }

#ifdef SMF_PARALLEL
  if (ctrl.num_threads > 1) {
    BlasSerialGuard blas_guard;
    const FactorStatus ps = factor_posdef_parallel(keep, ctrl, fkeep, Ap);
    if (ps != FactorStatus::Success) {
      info.factor_status = ps;
      const auto t1 = std::chrono::steady_clock::now();
      info.factor_seconds = std::chrono::duration<double>(t1 - t0).count();
      return ps;
    }
    info.actual_factor_entries =
        static_cast<LongInt>(fkeep.factor_values.size());
    build_solve_steps(keep, fkeep);
    const auto t1 = std::chrono::steady_clock::now();
    info.factor_seconds = std::chrono::duration<double>(t1 - t0).count();
    return FactorStatus::Success;
  }
#endif // SMF_PARALLEL

  // ---- Serial path --------------------------------------------------------

  // Use precomputed postorder from analysis (children before parents).
  // Children are also pre-sorted descending postorder in each Supernode::children
  // (done during symbolic_analysis) — no per-call sort needed.
  const std::vector<Int>& postorder = keep.postorder;

  // Arena for frontal matrix + A22 accumulator (reused each supernode via
  // save/reset).  Size from the largest front shape instead of O(n^2)
  // over-allocation.
  LongInt max_front = 1;
  LongInt max_ext = 0;
  for (Int s = 0; s < ns; ++s) {
    const std::size_t si = static_cast<std::size_t>(s);
    const LongInt f = static_cast<LongInt>(keep.fronts[si].front_size());
    const LongInt p = static_cast<LongInt>(keep.supernodes[si].width());
    max_front = std::max(max_front, f);
    max_ext = std::max(max_ext, std::max<LongInt>(0, f - p));
  }
  const LongInt workspace_elems =
      max_front * max_front * 2 + max_ext * max_ext;
  const LongInt hint_elems =
      info.predicted_factor_entries > 0 ? info.predicted_factor_entries
                                        : LongInt{1};
  const std::size_t arena_cap =
      static_cast<std::size_t>(std::max(workspace_elems, hint_elems)) *
          sizeof(double) +
      8192u;
  AlignedArena arena(arena_cap);

  // Two-buffer stack for contribution (Schur complement) blocks
  FactorStack fstack;

  // Per-supernode contribution pointers (into fstack) and block sizes
  std::vector<double *> contrib_ptrs(static_cast<std::size_t>(ns), nullptr);
  std::vector<Int> contrib_q(static_cast<std::size_t>(ns), Int{0});

  if (info.predicted_factor_entries > 0)
    fkeep.factor_values.reserve(
        static_cast<std::size_t>(info.predicted_factor_entries));

  // ---- Main postorder loop ------------------------------------------------
  for (Int pos = 0; pos < ns; ++pos) {
    const Int s = postorder[static_cast<std::size_t>(pos)];
    const FrontalInfo &fi = keep.fronts[static_cast<std::size_t>(s)];
    const Supernode &sn = keep.supernodes[static_cast<std::size_t>(s)];

    const Int p = sn.width();      // pivot column count
    const Int f = fi.front_size(); // total front rows
    const Int q = f - p;           // extended rows = contribution block size

    // Record start of this supernode's factor
    fkeep.factor_col_ptr[static_cast<std::size_t>(s)] =
        static_cast<int>(fkeep.factor_values.size());

    if (p <= 0)
      continue; // should not occur for valid input

    // Save arena so we can recycle it after this supernode
    const AlignedArena::Marker arena_mark = arena.save();

    // Allocate and zero the f×p frontal matrix (no col_indices copy — pointer into fi.row_indices)
    FrontalMatrix F(f, p, fi.row_indices.data(), fi.row_indices.data(), arena);

    // Allocate and zero the q×q A22 accumulator in the arena
    double *a22 = nullptr;
    if (q > 0) {
      const std::size_t a22_bytes = static_cast<std::size_t>(q) *
                                    static_cast<std::size_t>(q) *
                                    sizeof(double);
      a22 = static_cast<double *>(arena.allocate(a22_bytes));
      std::memset(a22, 0, a22_bytes);
    }

    // --- Scatter original entries from permuted A -----------------------
    // A11 (rows 0..p-1, cols 0..p-1) and A21 (rows p..f-1, cols 0..p-1) → F.
    // A22 (ext-row × ext-col) is NOT scattered here: those entries belong to
    // later supernodes that own those columns as pivot columns.  The a22
    // accumulator is zero-initialised above and receives only child Schur
    // complements assembled in the loop below.
    F.scatter_original(Ap.col_ptr, Ap.row_idx, Ap.values);

    // --- Assemble child contributions -----------------------------------
    // Children are pre-sorted descending postorder in sn.children (from analysis),
    // matching the LIFO fstack order. child_parent_rows may be precomputed or
    // built on demand.
    std::vector<Int> parent_rows_local;
    for (std::size_t k = 0; k < sn.children.size(); ++k) {
      const Int c = sn.children[k];
      const Int q_c = contrib_q[static_cast<std::size_t>(c)];
      if (q_c <= 0)
        continue;

      const double *cc = contrib_ptrs[static_cast<std::size_t>(c)];
      const std::vector<Int> *parent_rows_ptr = nullptr;
      if (k < fi.child_parent_rows.size() &&
          fi.child_parent_rows[k].size() == static_cast<std::size_t>(q_c)) {
        parent_rows_ptr = &fi.child_parent_rows[k];
      } else {
        const FrontalInfo &fi_c = keep.fronts[static_cast<std::size_t>(c)];
        const Int p_c = keep.supernodes[static_cast<std::size_t>(c)].width();
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

      // Assemble A11/A21 parts into F
      F.assemble_contrib(cc, q_c, parent_rows);

      // Assemble A22 parts (where both row and col map to parent extended)
      if (q > 0) {
        for (Int jj = 0; jj < q_c; ++jj) {
          const Int pr_j = parent_rows[static_cast<std::size_t>(jj)];
          if (pr_j < p)
            continue; // maps to parent pivot col — handled above
          const Int lcol = pr_j - p;

          for (Int ii = jj; ii < q_c; ++ii) {
            const Int pr_i = parent_rows[static_cast<std::size_t>(ii)];
            if (pr_i < p)
              continue; // maps to parent pivot row
            const Int lrow = pr_i - p;
            // lrow >= lcol because row_indices is sorted and ii >= jj
            a22[static_cast<std::size_t>(lcol) * static_cast<std::size_t>(q) +
                static_cast<std::size_t>(lrow)] +=
                cc[static_cast<std::size_t>(jj) *
                       static_cast<std::size_t>(q_c) +
                   static_cast<std::size_t>(ii)];
          }
        }
      }

      // Free child contribution from fstack (LIFO order)
      fstack.free_top(static_cast<std::size_t>(q_c) *
                      static_cast<std::size_t>(q_c));
      contrib_ptrs[static_cast<std::size_t>(c)] = nullptr;
      contrib_q[static_cast<std::size_t>(c)] = Int{0};
    }

    // --- Cholesky factorization of p×p pivot block ---------------------
    const int ret =
        smf_dpotrf_lower(F.data(), static_cast<int>(p), static_cast<int>(f));
    if (ret != 0) {
      arena.reset_to(arena_mark);
      info.factor_status = FactorStatus::NotPositiveDefinite;
      const auto t1 = std::chrono::steady_clock::now();
      info.factor_seconds = std::chrono::duration<double>(t1 - t0).count();
      return FactorStatus::NotPositiveDefinite;
    }

    // --- Compute L21 block: solve A21 * L11^{-T} = L21 ----------------
    if (q > 0 && p > 0) {
      smf_dtrsm_right_lower_transpose(
          F.data() + static_cast<std::ptrdiff_t>(p), // B = A21 block
          static_cast<int>(q),                       // n_rows
          static_cast<int>(p),                       // k
          F.data(),                                  // L11
          static_cast<int>(f),                       // ldl
          static_cast<int>(f));                      // ldb
    }

    // --- Schur complement: a22 -= L21 * L21^T --------------------------
    if (q > 0 && p > 0) {
      smf_dsyrk_lower(a22, static_cast<int>(q),
                      F.data() + static_cast<std::ptrdiff_t>(p), // L21
                      static_cast<int>(p),
                      static_cast<int>(f),  // lda (leading dim of L21 in F)
                      static_cast<int>(q)); // ldc
    }

    // --- Store factor columns (L11 and L21) in fkeep -------------------
    fkeep.factor_values.insert(fkeep.factor_values.end(), F.data(),
                               F.data() + static_cast<std::ptrdiff_t>(f) *
                                              static_cast<std::ptrdiff_t>(p));

    // --- Push contribution block to fstack for parent ------------------
    if (q > 0) {
      const std::size_t q2 =
          static_cast<std::size_t>(q) * static_cast<std::size_t>(q);
      double *c_ptr = fstack.alloc(q2);
      std::memcpy(c_ptr, a22, q2 * sizeof(double));
      contrib_ptrs[static_cast<std::size_t>(s)] = c_ptr;
      contrib_q[static_cast<std::size_t>(s)] = q;
    }

    // Reset arena — factor columns are in fkeep; contribution is in fstack
    arena.reset_to(arena_mark);
  } // end postorder loop

  // Sentinel: end of last supernode's factor data
  fkeep.factor_col_ptr[static_cast<std::size_t>(ns)] =
      static_cast<int>(fkeep.factor_values.size());

  // Populate info statistics
  info.actual_factor_entries = static_cast<LongInt>(fkeep.factor_values.size());
  info.arena_peak_bytes = static_cast<long>(arena.peak_bytes());
  info.arena_growths = arena.grow_count();
  build_solve_steps(keep, fkeep);

  const auto t1 = std::chrono::steady_clock::now();
  info.factor_seconds = std::chrono::duration<double>(t1 - t0).count();

  return FactorStatus::Success;
}

} // namespace smf
