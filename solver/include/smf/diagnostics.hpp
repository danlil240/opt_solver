#pragma once
#include "smf/analysis.hpp"
#include "smf/factor_posdef.hpp"
#include "smf/info.hpp"
#include "smf/types.hpp"
#include <string>

namespace smf {

/// Count the number of factor entries in fkeep.
LongInt count_factor_entries(const FactorKeep &fkeep);

/// Estimate floating-point operations for factorization given analysis.
/// Uses the standard formula: for each supernode s with p pivots and q extended
/// rows:
///   SPD flops ≈ p^3/3 + p^2*q  (Cholesky)
///   Indef flops ≈ p^3/3 + p^2*q (same leading term for LDLᵀ)
double count_flops(const AnalysisKeep &keep);

/// Format Info as a human-readable multi-line string.
std::string format_info(const Info &info);

} // namespace smf
