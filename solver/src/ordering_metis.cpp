#include "smf/ordering_metis.hpp"
#include <metis.h>
#include <vector>
#include <algorithm>
#include <memory>

namespace smf {

std::vector<Int> MetisOrdering::compute_ordering(
    Int n,
    const std::vector<Int>& xadj,
    const std::vector<Int>& adjncy,
    Info& info)
{
    if (n == 0) return {};

    // METIS uses idx_t (= int32_t or int64_t depending on build)
    // Our Int = int32_t; METIS default is idx_t = int32_t. Safe to cast.
    idx_t metis_n = static_cast<idx_t>(n);
    // METIS_NodeND wants mutable pointers
    std::vector<idx_t> xadj_m(xadj.begin(), xadj.end());
    std::vector<idx_t> adjncy_m(adjncy.begin(), adjncy.end());
    std::vector<idx_t> perm_m(static_cast<std::size_t>(n));
    std::vector<idx_t> iperm_m(static_cast<std::size_t>(n));

    int ret = METIS_NodeND(
        &metis_n,
        xadj_m.data(),
        adjncy_m.data(),
        nullptr,   // vwgt (no vertex weights)
        nullptr,   // options (use defaults)
        perm_m.data(),
        iperm_m.data()
    );

    if (ret != METIS_OK) {
        info.status = ErrorCode::OrderingFailed;
        return {};
    }

    std::vector<Int> perm(perm_m.begin(), perm_m.end());
    return perm;
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
std::unique_ptr<OrderingBackend> make_ordering_backend(
    OrderingMethod method, Int n, Info& info)
{
    (void)n;
    (void)info;
    switch (method) {
#if defined(SMF_HAS_METIS) && SMF_HAS_METIS
        case OrderingMethod::AutoParallel:
            [[fallthrough]];
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
