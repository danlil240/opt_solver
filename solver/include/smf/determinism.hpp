// determinism.hpp — Helpers for deterministic parallel execution mode.
// When Control::deterministic = true, subtree processing order is fixed
// by sorting child nodes by node ID before spawning tasks.
#pragma once
#include <algorithm>
#include <vector>

namespace smf {

/// Sort child node IDs in ascending order for deterministic task spawning.
void sort_children_deterministic(std::vector<int>& children);

/// Deterministic reduction: sum contribution vectors in a fixed index order.
/// This avoids floating-point non-associativity from variable task completion order.
void deterministic_reduce(std::vector<double>& accumulator,
                           const std::vector<std::vector<double>>& contributions,
                           int size);

} // namespace smf
