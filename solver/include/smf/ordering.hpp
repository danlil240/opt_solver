#pragma once
#include "smf/types.hpp"
#include "smf/info.hpp"
#include <vector>
#include <memory>

namespace smf {

/// Abstract ordering backend interface.
class OrderingBackend {
public:
    virtual ~OrderingBackend() = default;

    /// Compute a fill-reducing permutation.
    /// @param n        number of nodes (matrix dimension)
    /// @param xadj     CSR row pointers of symmetric adjacency (no self-loops), size n+1
    /// @param adjncy   CSR column indices, size xadj[n]
    /// @returns perm such that perm[new_idx] = old_idx  (MA97 convention)
    virtual std::vector<Int> compute_ordering(
        Int n,
        const std::vector<Int>& xadj,
        const std::vector<Int>& adjncy) = 0;
};

/// AMD ordering using SuiteSparse amd_order.
class AmdOrdering final : public OrderingBackend {
public:
    std::vector<Int> compute_ordering(
        Int n,
        const std::vector<Int>& xadj,
        const std::vector<Int>& adjncy) override;
};

/// Build symmetric CSR adjacency (no self-loops, each edge once per direction)
/// from a lower-CSC matrix. Returns {xadj, adjncy}.
/// Used as input to ordering backends and the elimination tree.
std::pair<std::vector<Int>, std::vector<Int>>
build_symmetric_csr(Int n,
                    const std::vector<Int>& col_ptr,
                    const std::vector<Int>& row_idx);

// --- Factory (added in M2.S1) ---
/// Create the ordering backend for the given method.
///   AutoSerial   -> AMD
///   AutoParallel -> METIS if SMF_HAS_METIS, else AMD
///   AMD          -> AMD
///   METIS        -> METIS (returns AMD backend if not compiled in)
/// Implemented in ordering_metis.cpp (always linked as part of libsmf).
std::unique_ptr<OrderingBackend> make_ordering_backend(
    OrderingMethod method, Int n, Info& info);

} // namespace smf
