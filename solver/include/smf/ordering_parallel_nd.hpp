#pragma once
#include "smf/types.hpp"
#include <vector>

namespace smf {

/// Leaves with fewer compressed vertices than this are ordered with AMD,
/// larger ones with METIS_NodeND. The brief's 20000 fails the 1.15x-of-METIS
/// flops gate on the FEM targets (AMD alone is 2.7x METIS on bmwcra_1; with
/// 20000 the order is 2.08x on bmwcra_1, 1.57x on pwtk); 200 (about where
/// METIS itself switches to minimum degree) gives 1.019x on both.
inline constexpr Int kParallelNdAmdLeafMax = 200;
/// Dissection depth: ceil(log2(8)) + 1, i.e. sized for 8 threads; fixed so
/// that the order does not depend on the thread count.
inline constexpr int kParallelNdLevels = 4;

struct ParallelNdOptions {
    int levels = kParallelNdLevels;
    Int amd_leaf_max = kParallelNdAmdLeafMax;
};

/// Parallel nested dissection: vertices with identical closed adjacency are
/// compressed into weighted supervariables; METIS vertex separators split the
/// compressed graph (disconnected parts are split along their components with
/// an empty separator) to a fixed depth; the leaves are ordered concurrently
/// on std::threads (AMD below kParallelNdAmdLeafMax supervariables,
/// METIS_NodeND above); separators are numbered last at every level; the
/// order is expanded with each supervariable's members contiguous. METIS
/// calls are serialised (metis_node_nd / metis_vertex_separator: the linked
/// METIS is not thread-safe), so the threads overlap AMD leaves and graph
/// work only.
///
/// The dissection tree depends on the graph only: num_threads changes the
/// schedule, never the result. xadj/adjncy: symmetric adjacency, both
/// directions, no self loops. vwgt: optional per-vertex weights (n entries).
/// perm[new] = old. n <= 2 gives the identity. Returns false (perm left
/// unspecified) when METIS is not compiled in or a METIS call fails.
bool parallel_nd_order(Int n, const std::vector<Int>& xadj, const std::vector<Int>& adjncy,
                       const std::vector<Int>* vwgt, int num_threads, std::vector<Int>& perm);

/// Same on raw arrays (e.g. the analyse arena's adjacency); perm has n entries.
bool parallel_nd_order(Int n, const Int* xadj, const Int* adjncy, const Int* vwgt,
                       int num_threads, Int* perm,
                       const ParallelNdOptions& options = ParallelNdOptions{});

/// Exact compression: vertices with identical closed adjacency (the same
/// neighbours, each other included) share a supervariable. super_of[v] is
/// v's supervariable, numbered in order of their smallest member. Returns
/// the number of supervariables.
Int compress_closed_adjacency(Int n, const Int* xadj, const Int* adjncy,
                              std::vector<Int>& super_of);


} // namespace smf
