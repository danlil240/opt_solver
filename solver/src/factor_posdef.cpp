#include "smf/factor_posdef.hpp"
#include "smf/dense_kernel.hpp"
#include "smf/factor_stack.hpp"
#include "smf/frontal_matrix.hpp"
#include "smf/threading.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <numeric>

namespace smf {

namespace {

/// Permute a lower-CSC matrix: perm[new]=old, iperm[old]=new.
/// Duplicates the logic from symbolic_analysis.cpp (which is not exported).
[[maybe_unused]] static CscLower
permute_lower_csc(const CscLower &A,
                  [[maybe_unused]] const std::vector<Int> &perm,
                  const std::vector<Int> &iperm) {
  const Int n = A.n;

  // Count entries per new column
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
    const Int end = B.col_ptr[static_cast<std::size_t>(j) + 1];
    const Int len = end - start;
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
      sr[static_cast<std::size_t>(k)] = B.row_idx[static_cast<std::size_t>(
          start + idx[static_cast<std::size_t>(k)])];
      sv[static_cast<std::size_t>(k)] = B.values[static_cast<std::size_t>(
          start + idx[static_cast<std::size_t>(k)])];
    }
    for (Int k = 0; k < len; ++k) {
      B.row_idx[static_cast<std::size_t>(start + k)] =
          sr[static_cast<std::size_t>(k)];
      B.values[static_cast<std::size_t>(start + k)] =
          sv[static_cast<std::size_t>(k)];
    }
  }

  return B;
}

/// Permute VALUES only using pre-computed pattern (col_ptr, row_idx) from analysis.
/// This is a fast path that avoids re-allocating and re-sorting the pattern.
static std::vector<double> permute_values_only(
    const CscLower &A_orig,
    const std::vector<Int> &iperm,
    const std::vector<Int> &perm_col_ptr,
    const std::vector<Int> &perm_row_idx) {
  
  const Int n = A_orig.n;
  const Int nnz_perm = perm_col_ptr[static_cast<std::size_t>(n)];
  std::vector<double> perm_values(static_cast<std::size_t>(nnz_perm), 0.0);

  // Scatter values from A_orig into permuted locations
  std::vector<Int> pos(perm_col_ptr.begin(),
                       perm_col_ptr.begin() + static_cast<std::ptrdiff_t>(n));

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
      // Since perm_row_idx is sorted within each column, we can binary search
      const Int col_start = perm_col_ptr[static_cast<std::size_t>(col)];
      const Int col_end = perm_col_ptr[static_cast<std::size_t>(col) + 1];
      
      // Linear search (binary search would be faster but add complexity)
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

/// Compute a postorder traversal of the supernode tree (children before
/// parents).
static std::vector<Int>
compute_postorder(const std::vector<Supernode> &supernodes) {
  const Int ns = static_cast<Int>(supernodes.size());
  std::vector<Int> order;
  order.reserve(static_cast<std::size_t>(ns));

  // Iterative DFS: each stack element is (node_index, next_child_to_visit)
  std::vector<std::pair<Int, Int>> stk;
  stk.reserve(static_cast<std::size_t>(ns));

  // Seed with all roots (supernodes whose parent == -1)
  for (Int i = 0; i < ns; ++i) {
    if (supernodes[static_cast<std::size_t>(i)].parent == -1)
      stk.push_back({i, 0});
  }

  while (!stk.empty()) {
    auto &[node, ci] = stk.back();
    const Int nch = static_cast<Int>(
        supernodes[static_cast<std::size_t>(node)].children.size());
    if (ci < nch) {
      const Int child = supernodes[static_cast<std::size_t>(node)]
                            .children[static_cast<std::size_t>(ci)];
      ++ci;
      stk.push_back({child, 0});
    } else {
      order.push_back(node);
      stk.pop_back();
    }
  }

  return order;
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

    const std::vector<Int> col_indices(
        fi.row_indices.begin(),
        fi.row_indices.begin() + static_cast<std::ptrdiff_t>(p));

    FrontalMatrix F(f, p, fi.row_indices, col_indices, arena);
    F.zero();

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
    for (const Int c : sn.children) {
      const std::size_t ci = static_cast<std::size_t>(c);
      if (contrib[ci].empty())
        continue;

      const Int p_c = keep.supernodes[ci].width();
      const Int f_c = keep.fronts[ci].front_size();
      const Int q_c = f_c - p_c;
      if (q_c <= 0)
        continue;

      const double *cc = contrib[ci].data();
      const FrontalInfo &fi_c = keep.fronts[ci];

      std::vector<Int> parent_rows(static_cast<std::size_t>(q_c));
      for (Int ii = 0; ii < q_c; ++ii) {
        const Int ext_row =
            fi_c.row_indices[static_cast<std::size_t>(p_c + ii)];
        const auto it = std::lower_bound(fi.row_indices.begin(),
                                         fi.row_indices.end(), ext_row);
        parent_rows[static_cast<std::size_t>(ii)] =
            static_cast<Int>(it - fi.row_indices.begin());
      }

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
      contrib[ci].shrink_to_fit();
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
  fkeep.perm = std::vector<int>(keep.perm.begin(), keep.perm.end());
  fkeep.iperm = std::vector<int>(keep.iperm.begin(), keep.iperm.end());
  fkeep.analysis = &keep;

  if (ns == 0 || n == 0) {
    const auto t1 = std::chrono::steady_clock::now();
    info.factor_seconds = std::chrono::duration<double>(t1 - t0).count();
    return FactorStatus::Success;
  }

  // Permute VALUES only using pre-computed pattern from analysis.
  // This avoids re-allocating and re-sorting col_ptr/row_idx on every factorization.
  std::vector<double> Ap_values = permute_values_only(
      keep.cleaned, keep.iperm, keep.perm_col_ptr, keep.perm_row_idx);
  
  // Build a CscLower view for scatter_original (pattern is pre-computed, values are fresh)
  CscLower Ap;
  Ap.n = n;
  Ap.col_ptr = keep.perm_col_ptr;
  Ap.row_idx = keep.perm_row_idx;
  Ap.values = std::move(Ap_values);

#ifdef SMF_PARALLEL
  if (ctrl.num_threads > 1) {
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

  // ---- Serial path (unchanged) --------------------------------------------

  // Compute postorder traversal of the supernode tree
  const std::vector<Int> postorder = compute_postorder(keep.supernodes);

  // Inverse postorder: po_idx[s] = position of supernode s in postorder
  std::vector<Int> po_idx(static_cast<std::size_t>(ns));
  for (Int i = 0; i < ns; ++i)
    po_idx[static_cast<std::size_t>(postorder[static_cast<std::size_t>(i)])] =
        i;

  // Arena for frontal matrix + A22 accumulator (reused each supernode via
  // save/reset)
  const std::size_t arena_cap =
      static_cast<std::size_t>(std::max(static_cast<LongInt>(n) * n,
                                        info.predicted_factor_entries > 0
                                            ? info.predicted_factor_entries
                                            : LongInt{1})) *
          sizeof(double) * 2 +
      4096u;
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

    // Build col_indices = first p entries of row_indices (pivot columns)
    const std::vector<Int> col_indices(fi.row_indices.begin(),
                                       fi.row_indices.begin() +
                                           static_cast<std::ptrdiff_t>(p));

    // Allocate and zero the f×p frontal matrix
    FrontalMatrix F(f, p, fi.row_indices, col_indices, arena);
    F.zero();

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
    // Sort children by postorder index descending to match LIFO free order
    std::vector<Int> sorted_ch(sn.children.begin(), sn.children.end());
    std::sort(sorted_ch.begin(), sorted_ch.end(), [&](Int a, Int b) {
      return po_idx[static_cast<std::size_t>(a)] >
             po_idx[static_cast<std::size_t>(b)];
    });

    for (const Int c : sorted_ch) {
      const Int q_c = contrib_q[static_cast<std::size_t>(c)];
      if (q_c <= 0)
        continue;

      const double *cc = contrib_ptrs[static_cast<std::size_t>(c)];
      const FrontalInfo &fi_c = keep.fronts[static_cast<std::size_t>(c)];
      const Int p_c = keep.supernodes[static_cast<std::size_t>(c)].width();

      // Build parent_rows: child extended row i → parent front row index
      std::vector<Int> parent_rows(static_cast<std::size_t>(q_c));
      for (Int ii = 0; ii < q_c; ++ii) {
        const Int ext_row =
            fi_c.row_indices[static_cast<std::size_t>(p_c + ii)];
        const auto it = std::lower_bound(fi.row_indices.begin(),
                                         fi.row_indices.end(), ext_row);
        parent_rows[static_cast<std::size_t>(ii)] =
            static_cast<Int>(it - fi.row_indices.begin());
      }

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
