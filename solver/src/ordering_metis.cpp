#include "smf/ordering_metis.hpp"
#if defined(SMF_HAS_METIS) && SMF_HAS_METIS
#include <metis.h>
#endif
#include <vector>
#include <algorithm>
#include <memory>
#include <mutex>

namespace smf {

#if defined(SMF_HAS_METIS) && SMF_HAS_METIS
static_assert(sizeof(idx_t) == sizeof(Int), "METIS idx_t must match smf::Int");

namespace {
std::mutex& metis_mutex()
{
    static std::mutex m;
    return m;
}

void metis_options(idx_t* options)
{
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_SEED] = 0;
    options[METIS_OPTION_NUMBERING] = 0;
}
} // namespace
#endif

bool metis_node_nd(Int n, Int* xadj, Int* adjncy, Int* vwgt, Int* perm, Int* iperm)
{
#if defined(SMF_HAS_METIS) && SMF_HAS_METIS
    idx_t nv = n;
    idx_t options[METIS_NOPTIONS];
    metis_options(options);
    std::lock_guard<std::mutex> lock(metis_mutex());
    return METIS_NodeND(&nv, xadj, adjncy, vwgt, options, perm, iperm) == METIS_OK;
#else
    (void)n; (void)xadj; (void)adjncy; (void)vwgt; (void)perm; (void)iperm;
    return false;
#endif
}

bool metis_vertex_separator(Int n, Int* xadj, Int* adjncy, Int* vwgt, Int* part, Int* sepsize)
{
#if defined(SMF_HAS_METIS) && SMF_HAS_METIS
    idx_t nv = n;
    idx_t options[METIS_NOPTIONS];
    metis_options(options);
    std::lock_guard<std::mutex> lock(metis_mutex());
    return METIS_ComputeVertexSeparator(&nv, xadj, adjncy, vwgt, options, sepsize, part) ==
           METIS_OK;
#else
    (void)n; (void)xadj; (void)adjncy; (void)vwgt; (void)part; (void)sepsize;
    return false;
#endif
}

std::vector<Int> MetisOrdering::compute_ordering(
    Int n,
    const std::vector<Int>& xadj,
    const std::vector<Int>& adjncy,
    Info& info)
{
    if (n == 0) return {};

#if defined(SMF_HAS_METIS) && SMF_HAS_METIS
    // METIS_NodeND wants mutable arrays.
    std::vector<Int> xadj_m(xadj.begin(), xadj.end());
    std::vector<Int> adjncy_m(adjncy.begin(), adjncy.end());
    std::vector<Int> perm(static_cast<std::size_t>(n));
    std::vector<Int> iperm(static_cast<std::size_t>(n));
    if (!metis_node_nd(n, xadj_m.data(), adjncy_m.data(), nullptr, perm.data(), iperm.data())) {
        info.status = ErrorCode::OrderingFailed;
        return {};
    }
    return perm;
#else
    info.status = ErrorCode::Success;
    AmdOrdering fallback;
    return fallback.compute_ordering(n, xadj, adjncy);
#endif
}

// OrderingBackend interface override — delegates to the Info overload
std::vector<Int> MetisOrdering::compute_ordering(
    Int n,
    const std::vector<Int>& xadj,
    const std::vector<Int>& adjncy)
{
    Info info;
    return compute_ordering(n, xadj, adjncy, info);
}

// Factory — create the right ordering backend for the requested method.
// AutoParallel uses METIS nested dissection only at scale (n > 8192), where
// its tree parallelism actually pays; below that AMD is both cheaper and at
// least as good fill-wise on measured workloads.
std::unique_ptr<OrderingBackend> make_ordering_backend(
    OrderingMethod method, Int n, Info& info)
{
    (void)info;
    switch (method) {
#if defined(SMF_HAS_METIS) && SMF_HAS_METIS
        case OrderingMethod::AutoParallel:
            if (n > 8192)
                return std::make_unique<MetisOrdering>();
            return std::make_unique<AmdOrdering>();
        case OrderingMethod::METIS:
            return std::make_unique<MetisOrdering>();
#endif
        case OrderingMethod::AutoSerial:
            [[fallthrough]];
        case OrderingMethod::AMD:
            [[fallthrough]];
        default:
            return std::make_unique<AmdOrdering>();
    }
}

} // namespace smf
