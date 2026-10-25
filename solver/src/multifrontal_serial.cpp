#include "smf/multifrontal_serial.hpp"
#include "smf/ldlt_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace smf {

ErrorCode multifrontal_factor_serial(const AnalysisKeep &keep, const Int *acp,
                                     const Int *ari, const double *av,
                                     const Control &ctrl, double eff_small,
                                     double static_pivot, bool posdef,
                                     FactorKeep &fkeep,
                                     NumericWorkspace &ws, SerialFactorStats &st) {
  const std::size_t n = static_cast<std::size_t>(keep.n);
  const std::size_t ns = keep.supernodes.size();

  // Size-as-high-water-mark growth: only the excess beyond the previous size
  // is value-initialised, and capacity grows geometrically. `plan` sizes
  // from the symbolic structure are not counted as growth.
  auto reserve_plan = [](auto &vec, std::size_t need) {
    if (vec.size() < need) vec.resize(need);
  };
  auto grow = [&ws](auto &vec, std::size_t need) {
    if (vec.size() >= need) return;
    if (vec.capacity() < need) {
      ++ws.capacity_growths_last_call;
      ++ws.front_heap_allocations_last_call;
      vec.reserve(std::max(need, vec.capacity() + vec.capacity() / 2));
    }
    vec.resize(need);
  };

  LdltKernelParams prm;
  prm.u = ctrl.pivot_u;
  prm.eff_small = eff_small;
  prm.small_pivot = ctrl.small_pivot;
  prm.nb = std::max(2, ctrl.indef_panel_size);
  prm.posdef = posdef;
  prm.static_pivot = posdef ? 0.0 : static_pivot;

  // Symbolic plan: largest front, factor size, and the peak of the
  // contribution stack when no pivot is delayed.
  std::size_t max_f = 1;
  std::size_t pred_factor = 0;
  std::size_t pred_rows = 0;
  ws.cb_value_off.resize(ns);
  ws.cb_row_off.resize(ns);
  ws.cb_size.resize(ns);
  ws.cb_delayed.resize(ns);
  {
    std::size_t vtop = 0, rtop = 0, vpeak = 0, rpeak = 0;
    for (const Int s : keep.postorder) {
      const std::size_t si = static_cast<std::size_t>(s);
      const Supernode &sn = keep.supernodes[si];
      const std::size_t f = keep.fronts[si].row_indices.size();
      const std::size_t p = static_cast<std::size_t>(sn.width());
      max_f = std::max(max_f, f);
      pred_factor += f * p;
      pred_rows += f;
      for (const Int c : sn.children) {
        const std::size_t ci = static_cast<std::size_t>(c);
        if (ws.cb_size[ci] > 0) {
          vtop = std::min(vtop, static_cast<std::size_t>(ws.cb_value_off[ci]));
          rtop = std::min(rtop, static_cast<std::size_t>(ws.cb_row_off[ci]));
        }
      }
      const std::size_t q = f - p;
      if (sn.parent >= 0 && q > 0) {
        ws.cb_value_off[si] = static_cast<LongInt>(vtop);
        ws.cb_row_off[si] = static_cast<LongInt>(rtop);
        ws.cb_size[si] = static_cast<Int>(q);
        vtop += q * (q + 1) / 2;
        rtop += q;
        vpeak = std::max(vpeak, vtop);
        rpeak = std::max(rpeak, rtop);
      } else {
        ws.cb_size[si] = 0;
      }
    }
    reserve_plan(ws.cb_values, vpeak);
    reserve_plan(ws.cb_rows, rpeak);
  }

  if (ws.row_pos.size() != n) ws.row_pos.assign(n, -1);
  reserve_plan(ws.front_rows, max_f);
  reserve_plan(ws.child_map, max_f);
  reserve_plan(ws.pivot_tags, max_f);
  reserve_plan(ws.kernel_work, ldlt_work_size(static_cast<int>(max_f), prm));

  auto &fv = fkeep.factor_values;
  auto &sri = fkeep.solve_row_indices;
  reserve_plan(fv, pred_factor);
  reserve_plan(sri, pred_rows);
  fkeep.factor_col_ptr.resize(ns + 1);
  if (!posdef) fkeep.pivot_types.resize(ns);
  fkeep.solve_steps.clear();
  fkeep.solve_steps.reserve(ns);
  fkeep.diag_entries.clear();
  if (!posdef) fkeep.diag_entries.reserve(n);

  Int *const pos = ws.row_pos.data();
  std::size_t fo = 0;     // next free slot in factor_values
  std::size_t ro = 0;     // next free slot in solve_row_indices
  std::size_t cv_top = 0; // contribution value stack top
  std::size_t cr_top = 0; // contribution row stack top
  ErrorCode status = ErrorCode::Success;

  for (const Int s : keep.postorder) {
    const std::size_t si = static_cast<std::size_t>(s);
    const Supernode &sn = keep.supernodes[si];
    const FrontalInfo &fi = keep.fronts[si];
    const int p_sym = static_cast<int>(sn.width());
    const int f_sym = static_cast<int>(fi.front_size());

    // Front rows: delayed columns from children, own pivots, then the
    // symbolic extension rows.
    int nd = 0;
    for (const Int c : sn.children) nd += ws.cb_delayed[static_cast<std::size_t>(c)];
    grow(ws.front_rows, static_cast<std::size_t>(nd + f_sym));
    Int *rows = ws.front_rows.data();
    int f = 0;
    if (nd > 0) {
      for (const Int c : sn.children) {
        const std::size_t ci = static_cast<std::size_t>(c);
        const Int *cr = ws.cb_rows.data() + ws.cb_row_off[ci];
        for (Int t = 0; t < ws.cb_delayed[ci]; ++t) {
          rows[f] = cr[t];
          pos[cr[t]] = f++;
        }
      }
    }
    for (int t = 0; t < f_sym; ++t) {
      const Int r = fi.row_indices[static_cast<std::size_t>(t)];
      rows[f] = r;
      pos[r] = f++;
    }
    const int num_elig = nd + p_sym;
    // Defensive: a child row outside the symbolic structure joins the front.
    for (const Int c : sn.children) {
      const std::size_t ci = static_cast<std::size_t>(c);
      const Int *cr = ws.cb_rows.data() + ws.cb_row_off[ci];
      for (Int t = ws.cb_delayed[ci]; t < ws.cb_size[ci]; ++t) {
        if (pos[cr[t]] < 0) {
          grow(ws.front_rows, static_cast<std::size_t>(f) + 1);
          rows = ws.front_rows.data();
          rows[f] = cr[t];
          pos[cr[t]] = f++;
        }
      }
    }

    const std::size_t sf = static_cast<std::size_t>(f);
    // The front is assembled in place at its factor offset: after the
    // kernel, columns [0, p) are already L in the solve layout, and the
    // trailing block is dead once pushed to the stack (the next front
    // starts at fo + f * p and reuses it).
    grow(fv, fo + sf * sf);
    grow(ws.child_map, sf);
    grow(ws.pivot_tags, sf);
    grow(ws.kernel_work, ldlt_work_size(f, prm));
    double *F = fv.data() + fo;
    for (std::size_t j = 0; j < sf; ++j) {
      std::fill_n(F + j * sf + j, sf - j, 0.0);
    }

    // Original entries of the pivot columns (lower CSC, so pos[row] >= pos[col]).
    for (Int j = sn.col_start; j < sn.col_end; ++j) {
      double *Fj = F + static_cast<std::size_t>(pos[j]) * sf;
      for (Int k = acp[j]; k < acp[j + 1]; ++k) {
        Fj[pos[ari[k]]] += av[k];
      }
    }

    // Children's contribution blocks (top of the stack).
    std::size_t new_cv_top = cv_top;
    std::size_t new_cr_top = cr_top;
    Int *cm = ws.child_map.data();
    for (const Int c : sn.children) {
      const std::size_t ci = static_cast<std::size_t>(c);
      const Int q = ws.cb_size[ci];
      if (q <= 0) continue;
      const Int *cr = ws.cb_rows.data() + ws.cb_row_off[ci];
      const double *cv = ws.cb_values.data() + ws.cb_value_off[ci];
      bool monotone = true;
      Int prev = -1;
      for (Int t = 0; t < q; ++t) {
        const Int l = pos[cr[t]];
        cm[t] = l;
        monotone = monotone && (l > prev);
        prev = l;
      }
      if (monotone) {
        for (Int jj = 0; jj < q; ++jj) {
          double *Fc = F + static_cast<std::size_t>(cm[jj]) * sf;
          for (Int ii = jj; ii < q; ++ii) Fc[cm[ii]] += *cv++;
        }
      } else {
        for (Int jj = 0; jj < q; ++jj) {
          const std::size_t lj = static_cast<std::size_t>(cm[jj]);
          for (Int ii = jj; ii < q; ++ii) {
            const std::size_t li = static_cast<std::size_t>(cm[ii]);
            if (li >= lj) F[lj * sf + li] += *cv++;
            else F[li * sf + lj] += *cv++;
          }
        }
      }
      new_cv_top = std::min(new_cv_top, static_cast<std::size_t>(ws.cb_value_off[ci]));
      new_cr_top = std::min(new_cr_top, static_cast<std::size_t>(ws.cb_row_off[ci]));
    }
    cv_top = new_cv_top;
    cr_top = new_cr_top;

    int8_t *tags = ws.pivot_tags.data();
    const LdltKernelResult kr =
        ldlt_factor_front(F, f, num_elig, rows, prm, tags, ws.kernel_work.data());
    const int p = kr.num_accepted;
    st.inertia.positive += kr.inertia.positive;
    st.inertia.negative += kr.inertia.negative;
    st.inertia.zero += kr.inertia.zero;
    st.static_pivots += kr.static_pivots;
    st.flops += kr.flops;
    st.max_front = std::max(st.max_front, f);

    if (posdef && p < num_elig) {
      for (int t = 0; t < f; ++t) pos[rows[t]] = -1;
      status = ErrorCode::NotPositiveDefinite;
      break;
    }

    // A root has no parent to delay to: unaccepted columns are zero pivots.
    const bool is_root = (sn.parent == -1);
    int p_store = p;
    if (is_root && p < num_elig) {
      for (int c = p; c < num_elig; ++c) tags[c] = 0;
      st.inertia.zero += num_elig - p;
      p_store = num_elig;
    }

    // Overflow anywhere in a front propagates into a later pivot of the same
    // or an ancestor front, so checking every D entry catches it without a
    // per-entry test elsewhere. Only the pivot blocks are rewritten: D is
    // read out first, then the strict upper part is zeroed and the unit (or,
    // for SPD, sqrt(d)) diagonal written.
    bool finite_d = true;
    if (posdef) {
      for (int c = 0; c < p_store; ++c) {
        double *Lc = F + static_cast<std::size_t>(c) * sf;
        const double sd = std::sqrt(Lc[c]);
        std::fill_n(Lc, c, 0.0);
        Lc[c] = sd;
        for (int i = c + 1; i < f; ++i) Lc[i] *= sd;
      }
    } else {
      for (int c = 0; c < p_store; ++c) {
        double *Lc = F + static_cast<std::size_t>(c) * sf;
        const int8_t tag = tags[c];
        if (tag == 0) {
          fkeep.diag_entries.push_back(DiagSolveEntry{1, rows[c], 0, 0.0, 0.0, 0.0});
          std::fill_n(Lc + c + 1, f - c - 1, 0.0);
        } else if (tag == 2) {
          fkeep.diag_entries.push_back(DiagSolveEntry{
              2, rows[c], rows[c + 1], Lc[c], Lc[c + 1], Lc[sf + c + 1]});
          finite_d = finite_d && is_finite_fast(Lc[c]) && is_finite_fast(Lc[c + 1]) &&
                     is_finite_fast(Lc[sf + c + 1]);
          Lc[c + 1] = 0.0;
        } else if (tag == 1) {
          fkeep.diag_entries.push_back(DiagSolveEntry{1, rows[c], 0, Lc[c], 0.0, 0.0});
          finite_d = finite_d && is_finite_fast(Lc[c]);
        }
        std::fill_n(Lc, c, 0.0);
        Lc[c] = 1.0;
      }
      fkeep.pivot_types[si].assign(tags, tags + p_store);
    }
    fkeep.factor_col_ptr[si] = static_cast<LongInt>(fo);
    SolveStep step;
    step.row_offset = static_cast<LongInt>(ro);
    step.factor_offset = static_cast<LongInt>(fo);
    step.p = p_store;
    step.f = f;
    fkeep.solve_steps.push_back(step);
    grow(sri, ro + sf);
    std::copy(rows, rows + f, sri.begin() + static_cast<std::ptrdiff_t>(ro));
    fo += sf * static_cast<std::size_t>(p_store);
    ro += sf;

    // Push the Schur complement (delayed columns first) for the parent.
    const int q = f - p;
    if (!is_root && q > 0) {
      const std::size_t sq = static_cast<std::size_t>(q);
      grow(ws.cb_values, cv_top + sq * (sq + 1) / 2);
      grow(ws.cb_rows, cr_top + sq);
      double *dst = ws.cb_values.data() + cv_top;
      for (int jj = 0; jj < q; ++jj) {
        const double *src = F + static_cast<std::size_t>(p + jj) * sf + (p + jj);
        std::copy(src, src + (q - jj), dst);
        dst += q - jj;
      }
      std::copy(rows + p, rows + f, ws.cb_rows.begin() + static_cast<std::ptrdiff_t>(cr_top));
      ws.cb_value_off[si] = static_cast<LongInt>(cv_top);
      ws.cb_row_off[si] = static_cast<LongInt>(cr_top);
      ws.cb_size[si] = q;
      ws.cb_delayed[si] = num_elig - p;
      cv_top += sq * (sq + 1) / 2;
      cr_top += sq;
      st.delayed += num_elig - p;
    } else {
      ws.cb_size[si] = 0;
      ws.cb_delayed[si] = 0;
    }

    for (int t = 0; t < f; ++t) pos[rows[t]] = -1;
    if (!finite_d) {
      status = ErrorCode::IllegalValue;
      break;
    }
  }

  // Leave the stack bookkeeping consistent for the next planning pass.
  if (status != ErrorCode::Success) {
    std::fill(ws.cb_size.begin(), ws.cb_size.end(), Int{0});
    std::fill(ws.cb_delayed.begin(), ws.cb_delayed.end(), Int{0});
    return status;
  }
  fkeep.factor_col_ptr[ns] = static_cast<LongInt>(fo);
  fv.resize(fo);
  sri.resize(ro);
  return ErrorCode::Success;
}

} // namespace smf
