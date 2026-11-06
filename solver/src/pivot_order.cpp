#include "smf/pivot_order.hpp"

namespace smf {

RealizedOrder realized_pivot_order(const AnalysisKeep &ak, const FactorKeep &fk) {
  RealizedOrder out;
  const std::size_t n = static_cast<std::size_t>(ak.n);
  if (fk.is_posdef) {
    out.perm.assign(ak.perm.begin(), ak.perm.end());
    return out;
  }
  out.perm.reserve(n);
  std::vector<Int> zeros;
  bool bad = false;
  // A D entry's rows are pivot indices of the factor's own final order, which
  // fk.perm maps to original columns on every engine: the lean engines number
  // them in the analysis order (fk.perm is ak.perm), the legacy tree-parallel
  // one in its numerical order (num_iperm).
  const auto orig = [&](Int piv) -> Int {
    if (piv < 0 || static_cast<std::size_t>(piv) >= fk.perm.size() || fk.perm.size() != n) {
      bad = true;
      return 0;
    }
    return fk.perm[static_cast<std::size_t>(piv)];
  };
  for (const DiagSolveEntry &e : fk.diag_entries) {
    if (e.was_zero) {
      zeros.push_back(orig(e.row0));
    } else if (e.tag == 2) {
      const Int a = orig(e.row0), b = orig(e.row1);
      out.perm.push_back(a);
      out.perm.push_back(b);
      out.pairs.push_back(a);
      out.pairs.push_back(b);
    } else {
      out.perm.push_back(orig(e.row0));
    }
    if (bad) break;
  }
  out.zero_pivots = static_cast<Int>(zeros.size());
  out.perm.insert(out.perm.end(), zeros.begin(), zeros.end());
  if (bad || out.perm.size() != n) {
    out.perm.clear();
    out.pairs.clear();
    out.zero_pivots = 0;
  }
  return out;
}

} // namespace smf
