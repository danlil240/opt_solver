#pragma once
#include "smf/types.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/etree.hpp"
#include "smf/supernode.hpp"
#include "smf/assembly_tree.hpp"
#include <vector>

namespace smf {

/// All symbolic analysis outputs needed by the factorization phase.
struct AnalysisKeep {
    CscLower                 cleaned;           ///< cleaned input (from check_matrix)
    std::vector<Int>         orig_col_ptr;      ///< original input col_ptr for pattern validation
    std::vector<Int>         orig_row_idx;      ///< original input row_idx for pattern validation
    std::vector<Int>         orig_to_clean_idx; ///< map: original entry k -> slot in cleaned (-1 if out of range)
    std::vector<Int>         orig_to_perm_direct; ///< map: original entry k -> slot in permuted matrix (-1 if out of range)
    std::vector<Int>         perm_col_ptr;      ///< col_ptr of permuted matrix (pattern only, for fast value permutation); empty when ordering is identity
    std::vector<Int>         perm_row_idx;      ///< row_idx of permuted matrix (pattern only, for fast value permutation); empty when ordering is identity
    std::vector<Int>         orig_to_perm_idx;  ///< scatter map: clean entry k → slot in permuted matrix (O(nnz) value permutation); empty when ordering is identity
    std::vector<Int>         perm;              ///< perm[new] = old (fill-reducing)
    std::vector<Int>         iperm;             ///< iperm[old] = new
    EliminationTree          etree;             ///< elimination tree of the permuted matrix
    std::vector<Supernode>   supernodes;        ///< amalgamated supernodes (children pre-sorted descending postorder)
    std::vector<FrontalInfo> fronts;            ///< per-supernode frontal info (child_parent_rows may be empty for large-n analyses)
    std::vector<Int>         postorder;         ///< supernode postorder: postorder[i] = s means supernode s is i-th in children-before-parents order
    std::vector<Int>         solve_postorder;   ///< supernode postorder reused by triangular solves
    std::vector<double>      matching_scale;    ///< Matching ordering: its matching's scaling of the analysed values (else empty)
    Int                      n = 0;             ///< matrix order (= cleaned.n)
};

} // namespace smf
