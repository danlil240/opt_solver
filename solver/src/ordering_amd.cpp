#include "smf/ordering.hpp"
#if defined(SMF_HAS_AMD) && SMF_HAS_AMD
#include <suitesparse/amd.h>
#endif
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace smf {

namespace {
#if !(defined(SMF_HAS_AMD) && SMF_HAS_AMD)
std::vector<Int> fallback_minimum_degree_ordering(
    Int n,
    const std::vector<Int>& xadj,
    const std::vector<Int>& adjncy)
{
    std::vector<Int> perm;
    perm.reserve(static_cast<std::size_t>(n));
    if (n <= 0) return perm;
    if (adjncy.empty() || static_cast<Int>(xadj.size()) < n + 1) {
        perm.resize(static_cast<std::size_t>(n));
        std::iota(perm.begin(), perm.end(), 0);
        return perm;
    }

    std::vector<std::vector<Int>> adj(static_cast<std::size_t>(n));
    std::vector<uint8_t> eliminated(static_cast<std::size_t>(n), 0);
    std::vector<Int> deg(static_cast<std::size_t>(n), 0);
    for (Int i = 0; i < n; ++i) {
        const Int start = xadj[static_cast<std::size_t>(i)];
        const Int end   = xadj[static_cast<std::size_t>(i) + 1];
        adj[static_cast<std::size_t>(i)].assign(
            adjncy.begin() + start, adjncy.begin() + end);
        deg[static_cast<std::size_t>(i)] = end - start;
    }

    for (Int step = 0; step < n; ++step) {
        Int pivot = -1;
        Int best_deg = n + 1;
        for (Int v = 0; v < n; ++v) {
            if (!eliminated[static_cast<std::size_t>(v)] &&
                deg[static_cast<std::size_t>(v)] < best_deg) {
                best_deg = deg[static_cast<std::size_t>(v)];
                pivot = v;
                if (best_deg == 0) break;
            }
        }
        if (pivot < 0) break;
        eliminated[static_cast<std::size_t>(pivot)] = 1;
        perm.push_back(pivot);

        std::vector<Int> nbrs;
        for (Int u : adj[static_cast<std::size_t>(pivot)]) {
            if (!eliminated[static_cast<std::size_t>(u)]) {
                nbrs.push_back(u);
            }
        }
        adj[static_cast<std::size_t>(pivot)].clear();

        for (std::size_t a = 0; a < nbrs.size(); ++a) {
            const Int u = nbrs[a];
            auto& list_u = adj[static_cast<std::size_t>(u)];
            list_u.erase(
                std::remove_if(list_u.begin(), list_u.end(), [&](Int w) {
                    return eliminated[static_cast<std::size_t>(w)] != 0;
                }),
                list_u.end());
            for (std::size_t b = 0; b < nbrs.size(); ++b) {
                if (a == b) continue;
                const Int w = nbrs[b];
                if (!std::binary_search(list_u.begin(), list_u.end(), w)) {
                    list_u.insert(
                        std::lower_bound(list_u.begin(), list_u.end(), w), w);
                }
            }
            deg[static_cast<std::size_t>(u)] = static_cast<Int>(list_u.size());
        }
    }
    return perm;
}
#endif
} // anonymous namespace

// Build symmetric CSR (no self-loops) from lower-CSC.
// For each off-diagonal entry (i,j) with i>j in the lower CSC:
//   add edge j->i and edge i->j to the symmetric graph.
// Sort each row's adjacency.
std::pair<std::vector<Int>, std::vector<Int>>
build_symmetric_csr(Int n,
                    const std::vector<Int>& col_ptr,
                    const std::vector<Int>& row_idx)
{
    // Count degree of each node (excluding self-loops)
    std::vector<Int> deg(static_cast<std::size_t>(n), 0);
    for (Int j = 0; j < n; ++j) {
        for (Int k = col_ptr[static_cast<std::size_t>(j)];
             k < col_ptr[static_cast<std::size_t>(j) + 1]; ++k) {
            Int i = row_idx[static_cast<std::size_t>(k)];
            if (i != j) {
                ++deg[static_cast<std::size_t>(j)];
                ++deg[static_cast<std::size_t>(i)];
            }
        }
    }
    // Build xadj
    std::vector<Int> xadj(static_cast<std::size_t>(n) + 1, 0);
    for (Int i = 0; i < n; ++i)
        xadj[static_cast<std::size_t>(i) + 1] =
            xadj[static_cast<std::size_t>(i)] + deg[static_cast<std::size_t>(i)];
    // Fill adjncy
    std::vector<Int> adjncy(static_cast<std::size_t>(xadj[static_cast<std::size_t>(n)]));
    // current fill position per row
    std::vector<Int> pos(xadj.begin(), xadj.begin() + n);
    for (Int j = 0; j < n; ++j) {
        for (Int k = col_ptr[static_cast<std::size_t>(j)];
             k < col_ptr[static_cast<std::size_t>(j) + 1]; ++k) {
            Int i = row_idx[static_cast<std::size_t>(k)];
            if (i != j) {
                adjncy[static_cast<std::size_t>(pos[static_cast<std::size_t>(j)]++)] = i;
                adjncy[static_cast<std::size_t>(pos[static_cast<std::size_t>(i)]++)] = j;
            }
        }
    }
    // Sort each row's adjacency
    for (Int i = 0; i < n; ++i)
        std::sort(adjncy.begin() + xadj[static_cast<std::size_t>(i)],
                  adjncy.begin() + xadj[static_cast<std::size_t>(i) + 1]);
    return {xadj, adjncy};
}

bool amd_order_lower_csc(Int n,
                         const std::vector<Int>& col_ptr,
                         const std::vector<Int>& row_idx,
                         std::vector<Int>& perm,
                         std::vector<Int>& work)
{
#if defined(SMF_HAS_AMD) && SMF_HAS_AMD
    const auto sn = static_cast<std::size_t>(n);
    perm.resize(sn);
    if (n == 0) return true;

    // amd_2 consumes the symmetric pattern (no diagonal) in its own Pe/Len/Iw
    // form with elbow room, plus nine size-n work arrays. Building that form
    // directly, in a caller-owned buffer, skips amd_order's validation,
    // A+A^T reconstruction and per-call allocation.
    std::size_t nz = 0;
    for (Int j = 0; j < n; ++j)
        for (Int k = col_ptr[static_cast<std::size_t>(j)];
             k < col_ptr[static_cast<std::size_t>(j) + 1]; ++k)
            if (row_idx[static_cast<std::size_t>(k)] != j) nz += 2;
    if (nz == 0) {
        std::iota(perm.begin(), perm.end(), 0);
        return true;
    }
    const std::size_t iwlen = nz + nz / 5 + sn;
    if (work.size() < 9 * sn + iwlen) work.resize(9 * sn + iwlen);
    Int *Pe = work.data();
    Int *Len = Pe + sn;
    Int *Nv = Len + sn;
    Int *Next = Nv + sn;
    Int *Last = Next + sn;
    Int *Head = Last + sn;
    Int *Elen = Head + sn;
    Int *Degree = Elen + sn;
    Int *W = Degree + sn;
    Int *Iw = W + sn;

    std::fill(Len, Len + sn, 0);
    for (Int j = 0; j < n; ++j)
        for (Int k = col_ptr[static_cast<std::size_t>(j)];
             k < col_ptr[static_cast<std::size_t>(j) + 1]; ++k) {
            const Int i = row_idx[static_cast<std::size_t>(k)];
            if (i != j) {
                ++Len[j];
                ++Len[i];
            }
        }
    Int p = 0;
    for (Int i = 0; i < n; ++i) {
        Pe[i] = p;
        W[i] = p;
        p += Len[i];
    }
    for (Int j = 0; j < n; ++j)
        for (Int k = col_ptr[static_cast<std::size_t>(j)];
             k < col_ptr[static_cast<std::size_t>(j) + 1]; ++k) {
            const Int i = row_idx[static_cast<std::size_t>(k)];
            if (i != j) {
                Iw[W[j]++] = i;
                Iw[W[i]++] = j;
            }
        }
    amd_2(static_cast<int>(n), Pe, Iw, Len, static_cast<int>(iwlen),
          static_cast<int>(nz), Nv, Next, Last, Head, Elen, Degree, W,
          nullptr, nullptr);
    std::copy(Last, Last + sn, perm.begin());
    return true;
#else
    (void)n; (void)col_ptr; (void)row_idx; (void)perm; (void)work;
    return false;
#endif
}

std::vector<Int> AmdOrdering::compute_ordering(
    Int n,
    const std::vector<Int>& xadj,
    const std::vector<Int>& adjncy)
{
    std::vector<Int> perm(static_cast<std::size_t>(n));
    // Handle empty adjacency (diagonal-only matrix): AMD returns AMD_INVALID when
    // adjncy.data() is nullptr (empty vector). Return identity permutation instead.
    if (adjncy.empty()) {
        std::iota(perm.begin(), perm.end(), 0);
        return perm;
    }
#if defined(SMF_HAS_AMD) && SMF_HAS_AMD
    // amd_order takes (n, Ap, Ai, P, Control, Info)
    // Ap and Ai are const int* — our Int is int32_t = int, so direct cast is fine.
    int result = amd_order(
        static_cast<int>(n),
        xadj.data(),
        adjncy.data(),
        perm.data(),
        nullptr,   // use default AMD control
        nullptr    // don't capture AMD info
    );
    if (result != AMD_OK && result != AMD_OK_BUT_JUMBLED) {
        throw std::runtime_error("amd_order failed with code " + std::to_string(result));
    }
    return perm;
#else
    return fallback_minimum_degree_ordering(n, xadj, adjncy);
#endif
}

} // namespace smf
