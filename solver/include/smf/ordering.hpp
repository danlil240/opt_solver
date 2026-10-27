#pragma once
#include "smf/types.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/info.hpp"
#include <cstddef>
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

/// AMD ordering of a lower-CSC pattern via amd_2, using `work` (grown as
/// needed and reusable across calls) for every AMD array. Returns false when
/// AMD is not compiled in. perm[new] = old.
bool amd_order_lower_csc(Int n,
                         const std::vector<Int>& col_ptr,
                         const std::vector<Int>& row_idx,
                         std::vector<Int>& perm,
                         std::vector<Int>& work);

/// Size of lower_csc_adjacency's `adj`: twice the off-diagonal entries of a
/// cleaned lower CSC (rows sorted and unique, all >= their column). O(n).
std::size_t lower_csc_adjacency_size(Int n, const Int *col_ptr, const Int *row_idx);

/// Symmetric adjacency (both triangles, no diagonal) of a cleaned lower CSC:
/// vertex v's neighbours are adj[xadj[v] .. xadj[v+1]), ascending, which is
/// the order amd_order_lower_csc feeds amd_2. One counting pass and one fill
/// pass; xadj has n + 1 entries, adj lower_csc_adjacency_size() entries.
void lower_csc_adjacency(Int n, const Int *col_ptr, const Int *row_idx, Int *xadj, Int *adj);

/// Ints amd_order_adjacency needs in `work` for an adjacency of nz entries:
/// amd_2's nine size-n arrays plus its Iw with amd_order's elbow room
/// (1.2 nz + n).
std::size_t amd_adjacency_work_size(Int n, std::size_t nz);

/// AMD (amd_2) on an adjacency from lower_csc_adjacency, which is copied
/// into `work` (amd_2 destroys its input), so the caller can reuse it for the
/// elimination tree and column counts. Same permutation as
/// amd_order_lower_csc on the same matrix. perm[new] = old; false when AMD is
/// not compiled in.
bool amd_order_adjacency(Int n, const Int *xadj, const Int *adj, Int *perm, Int *work);

/// How matching_order_lower_csc orders its compressed graph.
///   Amd   - AMD.
///   Metis - METIS nested dissection, vertex weight = pair size (AMD when
///           METIS is not compiled in or fails).
///   Auto  - AMD, plus METIS when AMD's predicted flops times the expected
///           refactors reach kMetisFlopsPerRow per row; the one with fewer
///           predicted flops is kept (MA97 ordering 8 parity).
enum class CompressedOrder { Amd, Metis, Auto };

/// Predicted factor flops per row from which METIS is worth trying instead
/// of AMD. One-off SuiteSparse SPD factorizations: METIS won at 8.8e5-1.1e6
/// (bmwcra_1, cfd2), tied at 2.1e5 (pwtk) and lost at 1.3e5-4.0e5
/// (af_shell7, G3_circuit, apache2).
inline constexpr double kMetisFlopsPerRow = 4.5e5;

struct MatchingOrderStats {
    Int pairs = 0;              ///< 2x2 pivot pairs kept adjacent
    Int nodes = 0;              ///< nodes of the compressed graph
    std::vector<double> scale;  ///< the matching's symmetric scaling
    /// Predicted factor flops (sum of exact column counts squared) of the
    /// returned ordering of A.
    double flops = 0.0;
    /// The compressed-graph ordering that was kept (Amd or Metis).
    CompressedOrder used = CompressedOrder::Amd;
};

/// Matching-based compressed ordering for symmetric indefinite matrices:
/// maximum-product matching, its cycles split into 2x2 pivot pairs (kept
/// when either scaled diagonal is below `pair_tol`), the graph with each
/// pair compressed to one node ordered as `how` says (see CompressedOrder;
/// the compressed graph is built once and shared by AMD and METIS), pairs
/// adjacent in the result. A must be cleaned (rows sorted and unique,
/// diagonal first). Needs values; returns false (caller falls back) when the
/// matrix is structurally singular or AMD is unavailable. perm[new] = old.
/// expected_refactors scales AMD's flops in the Auto rule for METIS.
bool matching_order_lower_csc(const CscLower& A, double pair_tol,
                              std::vector<Int>& perm, std::vector<Int>& amd_work,
                              MatchingOrderStats* stats = nullptr,
                              CompressedOrder how = CompressedOrder::Auto,
                              int expected_refactors = 1);

struct MatchingResult;

/// matching_order_lower_csc on a given matching `mr` of A, which may be
/// partial (SymmetricMatcher::initial()): free columns stay unpaired and
/// each open path of matched columns is split into pairs from its head.
/// stats->scale is mr.scale. False when mr does not fit A or AMD is
/// unavailable.
bool matching_order_from(const CscLower& A, const MatchingResult& mr, double pair_tol,
                         std::vector<Int>& perm, std::vector<Int>& amd_work,
                         MatchingOrderStats* stats = nullptr,
                         CompressedOrder how = CompressedOrder::Auto,
                         int expected_refactors = 1);

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
