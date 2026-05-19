// determinism.cpp — Deterministic parallel execution helpers.
#include "smf/determinism.hpp"

namespace smf {

void sort_children_deterministic(std::vector<int>& children) {
    std::sort(children.begin(), children.end());
}

void deterministic_reduce(std::vector<double>& accumulator,
                           const std::vector<std::vector<double>>& contributions,
                           int size) {
    accumulator.assign(static_cast<std::size_t>(size), 0.0);
    for (const auto& contrib : contributions) {
        for (int i = 0; i < size; ++i) {
            accumulator[static_cast<std::size_t>(i)] +=
                contrib[static_cast<std::size_t>(i)];
        }
    }
}

} // namespace smf
