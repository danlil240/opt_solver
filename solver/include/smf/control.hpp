#pragma once
#include "smf/types.hpp"
#include <cstdint>

namespace smf {

struct Control {
    // Matrix type
    MatrixType     matrix_type   = MatrixType::RealSymmetricIndefinite;

    // Ordering
    OrderingMethod ordering      = OrderingMethod::AutoSerial;
    int            nemin         = 8;       // min columns for supernode amalgamation
    // OrderingMethod::UserSupplied: user_perm[new] = old, length n, in
    // original index order; read by analyse() (null or not a permutation
    // gives ErrorCode::IllegalValue). It is used as given: no identity-band
    // detection, matching or METIS, and no matching scaling is computed.
    const Int*     user_perm     = nullptr;
    // UserSupplied only: 2x2 pivot pairs (user_pair_first[2k],
    // user_pair_first[2k+1]) in original indices, k < user_num_pairs. The
    // two of a pair must be adjacent in user_perm (else IllegalValue); they
    // are kept in one supernode when the later one is the etree parent of
    // the earlier, i.e. L(second, first) is structurally nonzero. Other
    // pairs cannot be forced together and are counted in Info::pairs_split.
    const Int*     user_pair_first = nullptr;
    Int            user_num_pairs  = 0;
    // Auto orderings of indefinite matrices: when to switch to the matching
    // ordering (see MatchingTrigger). RealizedDelays makes factor() mutate
    // the analysis it is given (one re-analysis, stored inside it): do not
    // factor the same AnalysisKeep from two threads at once.
    MatchingTrigger matching_trigger = MatchingTrigger::Predicted;

    // Scaling
    ScalingMethod  scaling       = ScalingMethod::Auto;
    // ScalingMethod::UserSupplied: n positive factors in original index
    // order, read during factor() (MA97's user `scale` argument).
    const double*  user_scale    = nullptr;

    // Pivoting
    double         pivot_u       = 0.01;    // threshold pivoting parameter
    double         small_pivot   = 1e-20;   // zero-pivot threshold
    // Static pivoting (PARDISO-style), off at 0: indefinite fronts never
    // delay; a column with no acceptable 1x1/2x2 pivot is taken as a 1x1
    // pivot, its |d| raised to static_pivot_tol * max|A| if smaller. The
    // factor is then of a perturbed matrix: solve with iterative refinement.
    double         static_pivot_tol = 0.0;
    // Fallback static pivoting, off at 0. Threshold pivoting with delays as
    // usual, except in a root front (no parent to delay to) and in a front
    // that receives more delayed columns from its children than its own
    // supernode width: there a column with no acceptable pivot is static
    // pivoted, |d| raised to fallback_static_pivot_tol * max|A|, instead of
    // becoming a zero pivot (root) or being delayed again. Such a front uses
    // this tolerance in place of static_pivot_tol. Its inertia is hybrid: a
    // pivot raised from an exact zero (|d| below the zero-pivot threshold)
    // still counts as a zero eigenvalue; one raised from a nonzero |d|
    // counts by its kept sign.
    double         fallback_static_pivot_tol = 0.0;
    // Every pivot whose |d| static pivoting raised (static_pivot_tol or the
    // fallback; Info::static_pivots) counts as a zero eigenvalue
    // (Info::num_zero, so the factor reports Singular) instead of by its sign.
    // Without it, static_pivot_tol fronts count raised pivots by sign
    // (exact zeros as +), fallback fronts as above.
    bool           static_pivots_as_zero = false;
    // static_pivot_tol fronts: hybrid accounting, as on fallback fronts (a
    // pivot raised from an exact zero counts as a zero eigenvalue, one raised
    // from a nonzero |d| by its kept sign). Off: by sign, exact zeros as +.
    bool           static_zero_stays_zero = false;
    // static_pivot_tol > 0 only: no pivot search. Every eligible column is
    // taken in order as a 1x1 pivot, and one with |d| below
    // static_pivot_tol * max|A| (or not finite) is raised to it, keeping its
    // sign (the quasidefinite mode). No 2x2 pivots, no delays.
    bool           static_pivot_in_order = false;
    // Diagonal regularization, off while reg_diag_sign is null (indefinite
    // only): after scaling, diagonal entry i gets
    // reg_diag_sign[i] * reg_diag_tol * max|A| added, max|A| taken after
    // scaling and before the regularization. reg_diag_sign: n values in
    // original index order, +1 / -1 / 0 (none). A row without a structural
    // diagonal entry is not regularized.
    const int8_t*  reg_diag_sign = nullptr;
    double         reg_diag_tol  = 0.0;
    int            indef_panel_size = 16;   // blocked indefinite panel width
    bool           indef_use_reference_kernel = false; // force scalar reference front kernel

    // Factorization behaviour
    bool           continue_on_singular = false;
    // Indefinite only (ignored for SPD), -1 = off. Once more negative
    // pivots than this have been accepted the factorization stops with
    // FactorStatus::InertiaExceeded: an interior-point method that expects
    // this many negatives would reject the factor anyway. A factorization
    // that stays within the limit is bit-identical to an unarmed one on the
    // serial engine (num_threads = 1). Multi-threaded factorizations vary
    // run to run at ~1e-14, armed or not: there only the inertia, delays and
    // factor size are identical.
    int            max_negative_pivots = -1;
    double         factor_memory_multiplier = 1.1;
    int            expected_refactors = 1;

    // Parallelism
    // num_threads > 1: independent subtrees of the assembly tree are
    // factored concurrently (serial BLAS each), then the top of the tree
    // with num_threads BLAS threads. tree_parallel opts into the older
    // task-parallel engine instead, which measured slower everywhere.
    int            num_threads   = 1;
    bool           tree_parallel = false;
    bool           deterministic = false;
    uint64_t       memory_budget_bytes = 0; // 0 = unbounded
    int64_t        factor_parallel_min_flops = 10000;
    int64_t        solve_parallel_min_entries = 100000;

    // Solve options
    bool           solve_use_blas3_single_rhs    = false;
    bool           solve_multifrontal_forward     = false;
    int            refinement_max_iters          = 5;

    // Input checking. True (the default): analyse() scans the pattern for
    // unsorted, duplicate and out-of-range entries and cleans them up.
    // False skips that O(nnz) scan for callers that guarantee clean input:
    // every column's rows sorted, unique and in [j, n) (AnalysisKeep::
    // input_clean of an earlier analyse of the same pattern says so); only
    // the O(n) shape checks and the missing-diagonal count remain. Unclean
    // input with false gives an undefined analysis.
    bool           check_input   = true;

    // Diagnostics / logging
    int            print_level   = 0;
};

} // namespace smf
