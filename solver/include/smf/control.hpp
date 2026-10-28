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
    Amalgamation   amalgamation  = Amalgamation::Smf;  // merge rule (see Amalgamation)
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
    // Work budget of the Predicted trial's matching searches, in multiples of
    // the matrix's symmetric nonzeros (2 x lower entries); negative: none.
    // Past it the trial first decides on the matching's initial phase (see
    // MatchingTrial). The decisions are the same; the budget only spares the
    // searches when the matching ordering loses anyway.
    double         matching_search_budget = 2.0;

    // Scaling
    ScalingMethod  scaling       = ScalingMethod::Auto;
    // ScalingMethod::UserSupplied: n positive factors in original index
    // order, read during factor() (MA97's user `scale` argument).
    const double*  user_scale    = nullptr;
    // Scaling on delay (negative: off). Under ScalingMethod::Auto, an
    // indefinite factorization with plain threshold pivoting at pivot_u >=
    // 1e-6 (kWeakPivotRatio; never IPOPT's u = 1e-8), no static pivoting
    // (static_pivot_tol, fallback_static_pivot_tol), no delay_budget, no
    // max_negative_pivots and not on the legacy tree_parallel engine stops
    // as soon as its front work (Info::front_work: fully summed columns x
    // rows^2 per front, as factored) exceeds scale_on_delay_ratio times that
    // of the same factorization without delays (Info::predicted_flops), and
    // is redone with the maximum-product matching scaling of its own values
    // (that of ScalingMethod::Matching). What the delays cost decides, not
    // how many there are: delays that inflate the fronts a few times over (a
    // dense KKT, a cheap one) are not worth a matching; a cascade, or the
    // badly scaled KKTs it marks, are (16: on the tuning split the easy
    // cases stay below 9, the ones the matching rescues start at 28). This
    // covers both what Auto applies: no scaling (the analysis kept AMD or
    // METIS) and the analysis' matching scale, which is of the analysed
    // values and can be stale for later ones (factoring the analysed values
    // themselves it never stops). The FactorKeep keeps the scale for that
    // analysis: a later factor() of it into the same FactorKeep starts from
    // the kept scale while it still fits the new values (see
    // scale_on_delay_reuse_tol), under the same work limit, and computes a
    // new matching scaling otherwise or when the limit trips. Only a
    // factorization that ran to the end updates the FactorKeep.
    // Info::delay_rescales and Info::scaling_used report it. Work is only
    // ever added, so the decision is the one the finished factorization's
    // count would give: identical at every num_threads in deterministic
    // mode.
    double         scale_on_delay_ratio = 16.0;
    // Fit test of a kept scale s on new values: the 90th percentile over
    // rows of |log10 max_j |s_i a_ij s_j|| (0 for the values' own matching
    // scaling, whose rows all peak at exactly 1) may be at most this:
    // 0.1, i.e. 90% of the rows within 26% of the matching property. On
    // the tuning split a kept scale that cost accuracy (cops_pinene_400,
    // omega 1e-8 against 1e-12 fresh) measured 0.18, kept PGLib scales that
    // stayed accurate 0.01-0.15. A factor with a kept scale also has to pass
    // an accuracy probe (one solve with a fixed right-hand side: backward
    // error within 10x of the fresh scale's and at most 1e-9), else the
    // matching scaling is computed anew. Negative: a new matching scaling
    // for every factorization once the trigger has fired (no probes).
    double         scale_on_delay_reuse_tol = 0.1;

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
    // Delay budget (MA57's approach), -1 = unlimited. Indefinite, the
    // multifrontal engine (not tree_parallel), threshold pivoting only
    // (ignored with static_pivot_tol > 0). Once delay_budget delayed pivots
    // (Info::delayed_pivots) have been counted, every remaining front of the
    // counting region factors with the static path at static_pivot_tol =
    // 1e-8 (relative to max|A| after scaling, inertia accounted as for
    // static_pivot_tol). A front whose own delays would take the count past
    // the budget is factored again with the static path
    // (Info::delay_budget_retries). Any static pivot makes
    // Info::inertia_exact false.
    // Counting regions are fixed by the tree, never by num_threads: the tree
    // is partitioned as the tree-parallel engine would partition it for 16
    // threads. A tree too light to partition is one region (one running
    // count over the postorder): there delayed_pivots <= delay_budget.
    // Otherwise each subtree of the partition counts from zero over its own
    // postorder and a front above them starts from the sum of its
    // children's end counts (spent if any child's budget was). The regions
    // do not depend on num_threads, but the delays counted in them do
    // whenever the front kernel does: with top_tiles on (pivot_u >= 1e-6),
    // the large top fronts use the tile kernel at num_threads > 1 and TPP at
    // 1 thread, so "the same choice on every front at 1, 2 or 8 threads"
    // holds only in deterministic mode (tiles at every thread count) or when
    // no front is tiled. delayed_pivots can exceed delay_budget (each
    // subtree, and each branch above them, has its own budget), with any
    // num_threads, 1 included.
    int            delay_budget  = -1;
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
    // Fronts of at most 32 rows on the TPP kernel are assembled and factored
    // in an aligned per-thread buffer with AVX2 rank-1 / rank-2 microkernels
    // (ldlt_factor_small_front), and a front whose parent is factored next
    // hands its contribution block over in that buffer instead of the stack.
    // Same pivots and factor values as the generic path (false).
    bool           small_front_kernels = true;

    // Front kernel of plain threshold-pivoting indefinite fronts (see
    // FrontKernel). Aptp = APTP on all fronts; Auto = APTP on fronts with at
    // least aptp_min_elig fully summed columns, and with aptp_min_elig = 0
    // (the default) never, i.e. Tpp. aptp_nb / aptp_ib: APTP outer (tile)
    // and inner block widths; nb = 64 measured ~3% faster than 128 on the
    // tuning split, ib 16 vs 32 made no difference.
    FrontKernel    front_kernel  = FrontKernel::Auto;
    int            aptp_nb       = 64;
    int            aptp_ib       = 32;
    int            aptp_min_elig = 0;

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
    // num_threads > 1 and pivot_u >= 1e-6 (kWeakPivotRatio; below it the
    // tile kernel would lack TPP's weak-1x1 rule, so IPOPT's u = 1e-8 keeps
    // TPP at every thread count): fronts of at least 512 rows at the top of
    // the tree that pivot with plain threshold pivoting (indefinite, no static or
    // in-order pivoting, not a fallback / delay-budget static front) are
    // factored as 2D tile tasks on the tree-parallel thread pool, serial
    // BLAS per task (APTP block-column structure, whatever front_kernel
    // says), instead of one threaded-BLAS kernel call. top_tile_nb: their
    // block-column / tile width.
    bool           top_tiles     = true;
    int            top_tile_nb   = 256;
    bool           tree_parallel = false;
    // Deterministic mode (determinism.hpp; also always on in an
    // SMF_DETERMINISTIC build): bitwise-identical ordering, factor, inertia
    // and solutions across runs and across num_threads. Costs serial BLAS
    // per front, tiled large fronts at every thread count (pivot_u >= 1e-6;
    // below it no front is tiled) and a serial solve; tree_parallel is
    // ignored. Exception: on a factorization aborted early
    // (max_negative_pivots, InertiaExceeded) the status is deterministic but
    // the partial inertia reported can depend on thread timing.
    bool           deterministic = false;
    uint64_t       memory_budget_bytes = 0; // 0 = unbounded
    int64_t        factor_parallel_min_flops = 10000;
    int64_t        solve_parallel_min_entries = 100000;

    // Solve options
    // Factor time also stores L11^-1 of every front with at least 16 pivots
    // (FactorKeep::inv_diag, solve_diag.hpp) so the solve applies the pivot
    // block with a GEMV/GEMM instead of a TRSV/TRSM. A block whose
    // kappa_1(L11) exceeds kInvDiagMaxCond keeps its TRSV (accuracy). Costs
    // about p^3/3 flops per front at factor time and p^2 doubles per front.
    // Off by default: on the 7 large_kkt IPOPT replays (1 thread) it saved
    // no solve time (the blocks with p >= 16 are few and small; GEMV on the
    // full p x p inverse does twice TRSV's flops) and cost 1-8% factor time.
    bool           cache_inverse_diag            = false;
    bool           solve_use_blas3_single_rhs    = false;
    bool           solve_multifrontal_forward     = false;
    // Classical refinement steps (refine(), factor_solve); 0 disables
    // refinement for every refine_method.
    int            refinement_max_iters          = 5;
    // Refinement method (see RefineMethod and refinement.hpp). Fgmres / Auto:
    // classical steps plus FGMRES iterations together never exceed
    // refine_max_steps; FGMRES restarts every fgmres_restart iterations.
    RefineMethod   refine_method                 = RefineMethod::Auto;
    int            refine_max_steps              = 10;
    int            fgmres_restart                = 20;

    // Input checking. True (the default): analyse() scans the pattern for
    // unsorted, duplicate and out-of-range entries and cleans them up.
    // False skips that O(nnz) scan for callers that guarantee clean input:
    // every column's rows sorted, unique and in [j, n) (AnalysisKeep::
    // input_clean of an earlier analyse of the same pattern says so); only
    // the O(n) shape checks and the missing-diagonal count remain. Unclean
    // input with false is undefined behaviour: unsorted, duplicate or
    // out-of-range rows lead to out-of-bounds writes and may corrupt memory.
    bool           check_input   = true;

    // Diagnostics / logging
    int            print_level   = 0;
};

} // namespace smf
