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
  const auto orig = [&](Int sym) -> Int {
    return ak.perm[static_cast<std::size_t>(sym)];
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
  }
  out.zero_pivots = static_cast<Int>(zeros.size());
  out.perm.insert(out.perm.end(), zeros.begin(), zeros.end());
  if (out.perm.size() != n) {
    out.perm.clear();
    out.pairs.clear();
    out.zero_pivots = 0;
  }
  return out;
}

} // namespace smf
