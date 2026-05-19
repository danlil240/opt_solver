#pragma once
#include "smf/ordering.hpp"  // OrderingBackend, OrderingMethod, Int
#include "smf/info.hpp"      // Info
#include <vector>

namespace smf {

/// METIS 5 NodeND fill-reducing ordering.
class MetisOrdering final : public OrderingBackend {
public:
    /// Compute fill-reducing permutation using METIS_NodeND.
    /// @param n      matrix dimension
    /// @param xadj   symmetric CSR row pointers (no self-loops), size n+1
    /// @param adjncy symmetric CSR column indices, size xadj[n]
    /// @param info   populated with ErrorCode::OrderingFailed on METIS error
    /// @returns perm[new_idx] = old_idx on success; empty on failure
    std::vector<Int> compute_ordering(
        Int n,
        const std::vector<Int>& xadj,
        const std::vector<Int>& adjncy,
        Info& info);

    /// OrderingBackend interface override — delegates to the Info overload
    /// using a local Info instance.
    std::vector<Int> compute_ordering(
        Int n,
        const std::vector<Int>& xadj,
        const std::vector<Int>& adjncy) override;
};

} // namespace smf
