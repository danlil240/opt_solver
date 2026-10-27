#pragma once
#include "smf/types.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/etree.hpp"
#include "smf/supernode.hpp"
#include "smf/assembly_tree.hpp"
#include <cstddef>
#include <span>
#include <vector>

namespace smf {

/// All symbolic analysis outputs needed by the factorization phase.
struct AnalysisKeep {
    CscLower                 cleaned;           ///< cleaned input (from check_matrix)
    std::vector<Int>         orig_col_ptr;      ///< original input col_ptr for pattern validation
    std::vector<Int>         orig_row_idx;      ///< original input row_idx for pattern validation
    std::vector<Int>         orig_to_clean_idx; ///< map: original entry k -> slot in cleaned (-1 if out of range)
    std::vector<Int>         orig_to_perm_direct; ///< map: original entry k -> slot in permuted matrix (-1 if out of range)
    bool                     input_clean = false; ///< input had sorted, unique, in-range rows (re-analyses of this pattern may pass Control::check_input = false)
    std::vector<Int>         perm_col_ptr;      ///< col_ptr of permuted matrix (pattern only, for fast value permutation); empty when ordering is identity
    std::vector<Int>         perm_row_idx;      ///< row_idx of permuted matrix (pattern only, for fast value permutation); empty when ordering is identity
    std::vector<Int>         orig_to_perm_idx;  ///< scatter map: clean entry k → slot in permuted matrix (O(nnz) value permutation); empty when ordering is identity
    std::vector<Int>         perm;              ///< perm[new] = old (fill-reducing)
    std::vector<Int>         iperm;             ///< iperm[old] = new
    EliminationTree          etree;             ///< elimination tree of the permuted matrix
    std::vector<Supernode>   supernodes;        ///< amalgamated supernodes (postorder; tree via parent and children())
    // Supernode tree in flat CSR form (no per-supernode heap storage).
    std::vector<Int>         child_ptr;         ///< nsn + 1 offsets into child_idx
    std::vector<Int>         child_idx;         ///< children of each supernode, descending postorder
    std::vector<Int>         front_row_ptr;     ///< nsn + 1 offsets into front_row_idx
    std::vector<Int>         front_row_idx;     ///< rows of each front, sorted ascending
    /// Optional parent-row scatter maps, one per child slot t = child_ptr[s] + k:
    /// cpr_idx[cpr_ptr[t] .. cpr_ptr[t+1]) gives, for child k's i-th extension
    /// row, its position in front s. Both empty when not stored (analyse does
    /// not store them; factorization computes the map on demand); otherwise
    /// cpr_ptr has child_idx.size() + 1 entries.
    std::vector<Int>         cpr_ptr;
    std::vector<Int>         cpr_idx;
    std::vector<Int>         postorder;         ///< supernode postorder: postorder[i] = s means supernode s is i-th in children-before-parents order
    std::vector<Int>         solve_postorder;   ///< supernode postorder reused by triangular solves
    std::vector<double>      matching_scale;    ///< Matching ordering: its matching's scaling of the analysed values (else empty)
    Int                      n = 0;             ///< matrix order (= cleaned.n)

    /// Number of supernodes (= supernodes.size()).
    Int num_supernodes() const { return static_cast<Int>(supernodes.size()); }

    /// Children of supernode s in descending postorder.
    std::span<const Int> children(Int s) const {
        const auto i = static_cast<std::size_t>(s);
        return std::span<const Int>(child_idx).subspan(
            static_cast<std::size_t>(child_ptr[i]),
            static_cast<std::size_t>(child_ptr[i + 1] - child_ptr[i]));
    }

    /// Row indices of supernode s's front, sorted ascending (the first
    /// width() of them are its pivot columns).
    std::span<const Int> front_rows(Int s) const {
        const auto i = static_cast<std::size_t>(s);
        return std::span<const Int>(front_row_idx).subspan(
            static_cast<std::size_t>(front_row_ptr[i]),
            static_cast<std::size_t>(front_row_ptr[i + 1] - front_row_ptr[i]));
    }

    /// Number of rows of supernode s's front.
    Int front_size(Int s) const {
        const auto i = static_cast<std::size_t>(s);
        return front_row_ptr[i + 1] - front_row_ptr[i];
    }

    /// True when the per-child parent-row maps are stored (cpr_ptr non-empty).
    bool has_child_parent_rows() const { return !cpr_ptr.empty(); }

    /// Positions in front s of child k's extension rows (k indexes children(s)).
    /// Empty when has_child_parent_rows() is false: callers then compute the
    /// map themselves.
    std::span<const Int> child_parent_rows(Int s, Int k) const {
        if (cpr_ptr.empty()) return {};
        const auto t = static_cast<std::size_t>(child_ptr[static_cast<std::size_t>(s)] + k);
        return std::span<const Int>(cpr_idx).subspan(
            static_cast<std::size_t>(cpr_ptr[t]),
            static_cast<std::size_t>(cpr_ptr[t + 1] - cpr_ptr[t]));
    }
};

} // namespace smf
