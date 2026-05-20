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
    std::vector<Int>         perm_col_ptr;      ///< col_ptr of permuted matrix (pattern only, for fast value permutation)
    std::vector<Int>         perm_row_idx;      ///< row_idx of permuted matrix (pattern only, for fast value permutation)
    std::vector<Int>         orig_to_perm_idx;  ///< scatter map: orig entry k → slot in permuted matrix (O(nnz) value permutation)
    std::vector<Int>         perm;              ///< perm[new] = old (fill-reducing)
    std::vector<Int>         iperm;             ///< iperm[old] = new
    EliminationTree          etree;             ///< elimination tree of the permuted matrix
    std::vector<Supernode>   supernodes;        ///< amalgamated supernodes
    std::vector<FrontalInfo> fronts;            ///< per-supernode frontal info
    std::vector<Int>         solve_postorder;   ///< supernode postorder reused by triangular solves
    Int                      n = 0;             ///< matrix order (= cleaned.n)
};

} // namespace smf
