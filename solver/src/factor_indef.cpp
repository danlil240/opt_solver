#include "smf/factor_indef.hpp"
#include "smf/frontal_matrix.hpp"
#include "smf/pivoting.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

namespace smf {

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

    // ---- Scatter original A values ----------------------------------
    front.scatter_original(keep.cleaned.col_ptr, keep.cleaned.row_idx,
                           keep.cleaned.values);

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

    // ---- Compute simplified contribution block for parent ------------
    // CB[j*ext+i] = -∑_k  L_ext[i,k] · D_k · L_ext[j,k]   (i >= j)
    // The extension rows are untouched by the BBK loop (num_fs == p),
    // so front.at(p+i, k) is the raw assembled value for ext row i, col k.
    if (ext > 0 && sn.parent >= 0) {
      const std::size_t ext_sz = static_cast<std::size_t>(ext);
      contrib[si].assign(ext_sz * ext_sz, 0.0);
      double *cb = contrib[si].data();

      for (int pk = 0; pk < num_fs; ++pk) {
        const int8_t tag = pivot_tag[static_cast<std::size_t>(pk)];
        if (tag == int8_t{0} || tag == int8_t{-1})
          continue;

        if (tag == int8_t{1}) {
          // 1×1 pivot at column pk
          const double d = front.at(static_cast<Int>(pk), static_cast<Int>(pk));
          if (std::abs(d) < ctrl.small_pivot)
            continue;
          const double inv_d = 1.0 / d;

          for (Int j = 0; j < ext; ++j) {
            const double lj = front.at(p + j, static_cast<Int>(pk)) * inv_d;
            for (Int i = j; i < ext; ++i) {
              const double li = front.at(p + i, static_cast<Int>(pk)) * inv_d;
              cb[static_cast<std::size_t>(j) * ext_sz +
                 static_cast<std::size_t>(i)] -= li * d * lj;
            }
          }

        } else { // tag == 2 : 2×2 pivot at (pk, pk+1)
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
            const double r0j = front.at(p + j, static_cast<Int>(pk));
            const double r1j = front.at(p + j, static_cast<Int>(pk + 1));
            const double lj0 = (d11 * r0j - d10 * r1j) / det;
            const double lj1 = (-d10 * r0j + d00 * r1j) / det;

            for (Int i = j; i < ext; ++i) {
              const double r0i = front.at(p + i, static_cast<Int>(pk));
              const double r1i = front.at(p + i, static_cast<Int>(pk + 1));
              const double li0 = (d11 * r0i - d10 * r1i) / det;
              const double li1 = (-d10 * r0i + d00 * r1i) / det;

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

  if (info.delayed_pivots > 0 && !ctrl.continue_on_singular)
    return FactorStatus::Singular;
  return FactorStatus::Success;
}

} // namespace smf
