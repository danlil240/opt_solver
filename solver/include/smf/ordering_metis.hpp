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

/// True when METIS calls may run concurrently: smf is built with the
/// vendored METIS (SMF_VENDORED_METIS, the default), whose GKlib state is
/// thread-local. With an external METIS (find_package) it is false: the
/// conda-forge 5.2.1 DLL on Windows keeps GKlib's memory core in
/// process-global state, and two concurrent calls abort the process
/// ("Could not find pointer ... in mcore").
inline constexpr bool kMetisThreadSafe =
#if defined(SMF_METIS_THREADSAFE) && SMF_METIS_THREADSAFE
    true;
#else
    false;
#endif
bool metis_is_thread_safe() noexcept;

/// The METIS entry points of smf; every METIS call goes through these. Unless
/// metis_is_thread_safe(), they hold one process-wide mutex (calls are
/// serialised). Options: METIS defaults with SEED = 0 and NUMBERING = 0, and
/// METIS's random stream is per thread and reseeded by every call, so a
/// call's result depends on its arguments only (not on call order or the
/// calling thread). The arrays are passed to METIS as-is (its API takes them
/// mutable). vwgt may be null. Return false on a METIS error or when METIS
/// is not compiled in. metis_node_nd: perm[new] = old, iperm[old] = new.
bool metis_node_nd(Int n, Int* xadj, Int* adjncy, Int* vwgt, Int* perm, Int* iperm);
/// metis_vertex_separator: part[v] in {0, 1, 2}, 2 = separator.
bool metis_vertex_separator(Int n, Int* xadj, Int* adjncy, Int* vwgt, Int* part, Int* sepsize);

} // namespace smf
