#include "smf/diagnostics.hpp"

namespace smf {

LongInt count_factor_entries(const FactorKeep &fkeep) {
  return static_cast<LongInt>(fkeep.factor_values.size());
}

double count_flops(const AnalysisKeep &keep) {
  double total = 0.0;
  const std::size_t ns = keep.supernodes.size();
  for (std::size_t s = 0; s < ns; ++s) {
    const Supernode &sn = keep.supernodes[s];
    const double p = static_cast<double>(sn.width());
    const double f = (s < keep.fronts.size())
                         ? static_cast<double>(keep.fronts[s].front_size())
                         : p;
    const double q = f - p;
    total += p * p * p / 3.0 + p * p * q;
  }
  return total;
}

std::string format_info(const Info &info) {
  std::string s;
  s += "smf::Info {\n";
  s += "  status:                   " +
       std::to_string(static_cast<int>(info.status)) + "\n";
  s += "  factor_status:            " +
       std::to_string(static_cast<int>(info.factor_status)) + "\n";
  s += "  numerical_rank:           " + std::to_string(info.numerical_rank) +
       "\n";
  s +=
      "  num_negative:             " + std::to_string(info.num_negative) + "\n";
  s += "  num_zero:                 " + std::to_string(info.num_zero) + "\n";
  s +=
      "  num_positive:             " + std::to_string(info.num_positive) + "\n";
  s += "  actual_factor_entries:    " +
       std::to_string(info.actual_factor_entries) + "\n";
  s += "  predicted_factor_entries: " +
       std::to_string(info.predicted_factor_entries) + "\n";
  s += "  predicted_flops:          " + std::to_string(info.predicted_flops) +
       "\n";
  s +=
      "  actual_flops:             " + std::to_string(info.actual_flops) + "\n";
  s += "  delayed_pivots:           " + std::to_string(info.delayed_pivots) +
       "\n";
  s += "  factor_seconds:           " + std::to_string(info.factor_seconds) +
       "\n";
  s += "  solve_seconds:            " + std::to_string(info.solve_seconds) +
       "\n";
  s += "  arena_peak_bytes:         " + std::to_string(info.arena_peak_bytes) +
       "\n";
  s += "  arena_growths:            " + std::to_string(info.arena_growths) +
       "\n";
  s += "}";
  return s;
}

} // namespace smf
