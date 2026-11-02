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
    Amalgamation   amalgamation  = Amalgamation::CostModel;  // merge rule (see Amalgamation)
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
    // Work budget of the Predicted trial's matching searches. It is a
    // multiple of the lower-triangle entries of the cleaned matrix, diagonal
    // included: the searches may scan budget x 2 x that many edges (2 x the
    // lower entries is about the edge count of the matcher's symmetric graph,
    // so the budget reads as passes over it). The check runs before each
    // search, so the last search may overshoot, and a search is charged its
    // columns' full lengths even when it stops early in a dense column:
    // matrices with dense rows (OPF, KKT) reach the budget sooner. Negative,
    // NaN or infinite: no budget (the exact trial). 0: whenever a search is
    // left after the initial phase, decide on that phase first. Past the
    // budget the trial first decides on the matching's initial phase (see
    // MatchingTrial): a clear loss (initial flop ratio past 2.4 = 1.6 x the
    // 1.5 rule, the guard band) ends it; otherwise the searches resume and
    // the exact matching decides, so the budget only spares the searches of
    // clear losers. Why 2: on the 46 corpus matrices with a trial (n >=
    // 10000) the full searches cost at most 1.61 passes on 15 (no pause, 24%
    // headroom), 2.2 to 5.2 on 6 and 10 to 263 on the other 25 (PGLib OPF,
    // TSOPF, ncvxqp: the costly ones it is for). A value from 5.2 to 10
    // would pause only those 25; 2 also pauses 5 of the 6 middle ones
    // (darcy003, mario002, turon_m, stokes128, boyd1), where the outcome is
    // the same and the saving small (the sixth, aug3dcqp at 2.22, finishes
    // inside its last search).
    double         matching_search_budget = 2.0;
    // The sparse-part engine of analyse()'s auto matching trial (its
    // Degree-order SymmetricMatcher, serial and ConcurrentTrial) and of
    // nothing else: compute_symmetric_matching, compute_matching_scale and
    // the scale-on-delay rescue stay ShortestPath. Hybrid runs the
    // shortest-path searches and, past 40 passes of search work, a warm
    // auction on the remaining tail (scaling_matching.hpp); up to that point
    // it is ShortestPath bitwise, so within the 2-pass budget above no
    // verdict moves. The default stays ShortestPath. Re-gate (12900K,
    // interleaved A/B, indefinite tier, 3 rounds; f7dc144, canonical duals):
    // accuracy passed (no case-track lost, ncvxqp1 rec omega 5.7e-12),
    // verdicts passed (0 changes on 50 matrices), no case-track above 1.10x
    // after a 5-round rerun of the flagged sub-ms cases, ncvxqp3 analysis
    // 2.34x (1T) / 2.49x (8T) faster (wall 0.643x / 0.511x). It missed
    // ncvxqp7's 1.8x analysis target at 1T: 1.70-1.75x on the board,
    // 1.76-1.77x on a 5-round rerun (8T 1.83-1.85x). All-case wall geomean
    // 0.990 (1T) / 0.983 (8T).
    MatchingAlgorithm matching_algorithm = MatchingAlgorithm::ShortestPath;

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
    // A factorization that the work limit lets through as Auto resolves it,
    // but whose front work still exceeds scale_on_delay_next_ratio times the
    // delay-free one, leaves the next factor() of that analysis into that
    // FactorKeep to run under scale_on_delay_next_ratio as its limit
    // instead: delays that heavy often grow into a cascade over an IPOPT
    // sequence, and a try under the full ratio costs up to that many times
    // the delay-free work before it stops (pde_control_250 on the cost-model
    // tree: 1.4 s, against a 0.1 s factorization). A one-off spike
    // (cops_elec_200: 2.2, 8.5, 2.2) passes the tighter limit unscaled and
    // pays no matching. Not after a factor with the analysed values' own
    // matching scale (a new one would be the same). Negative: off.
    double         scale_on_delay_next_ratio = 4.0;
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
    // under a budget of as many BLAS threads in total as the factorization
    // runs on (at most num_threads; auto_threads below picks how many).
    // tree_parallel opts into the older task-parallel engine instead, which
    // measured slower everywhere.
    int            num_threads   = 1;
    // auto_threads (num_threads > 1): the factorization runs on as many of
    // the num_threads threads as its assembly tree repays, by a cost model
    // (src/parallel/thread_plan.hpp) of the engine's two phases at T threads.
    // The subtree phase runs on serial BLAS (BlasSerialGuard); the top
    // fronts run one after another or side by side, each under a
    // BlasThreadGuard with its share of the threads, or as tile tasks.
    // Both models (this and the solve's, solve_cost_* below) price work in
    // flops plus a per-thread and per-front term. The factor model: the
    // subtrees (the larger of the heaviest subtree and their work / T), then
    // the top part (the larger of its heaviest path, fronts run as tile
    // tasks counting 1/T, and its work, whose tile-split share speeds up to
    // thread_split_scaling_cap threads only), plus thread_cost_per_thread per
    // thread, thread_cost_per_top_front per top-part front and
    // thread_cost_per_split_front_thread per split front and thread;
    // candidates 1, 2, 4, 8, ... and num_threads. The choice is cached per
    // analysis in the NumericWorkspace. Fewer than 2 threads: the serial
    // engine (Info::threads_used = 1). The choice never changes deterministic
    // mode's results. Front-tier board, factor time at 16 threads against 1
    // with all threads used: robot_arm_8k 7.7x slower (11.8k of its 22.7k
    // fronts in the top part), cops_elec_200 / cops_polygon_200 slower (work
    // bound total / heaviest path 1.00 / 1.04); trees from 1.9e8 front work
    // up (TSOPF_FS_b39_c7) faster; ncvxqp3 ipopt (bound 1.54, root
    // dominated) gains nothing past 2-4 threads. With the model (acceptance
    // run, docs/performance/2026-10-30-thread-scheduling.md): robot_arm_8k
    // 10.5 ms at 1, 8 and 16 threads; factor time at 8 / 16 threads over the
    // trees of at least 1e9 front work 0.90 / 0.87x the engine without it.
    // auto_threads also covers the solve: on, the solve's thread count comes
    // from its own cost model (solve_cost_* below).
    // auto_threads = false: the tree is partitioned for num_threads threads
    // whenever its front work reaches 2e7, and the solve runs on the
    // factorization's thread count (FactorKeep::solve_threads; at most the
    // solve call's num_threads and the solve subtrees); neither cost
    // model chooses a count (both are evaluated for the plan trace alone,
    // SMF_BENCH_PLAN_TRACE), so thread counts are fixed for scaling
    // measurements.
    bool           auto_threads  = true;
    // The factor cost model's terms in front-work units (f^2 p). Fitted
    // offline on the Ryzen 9950X3D (2 CCDs of 8 cores) against measured
    // factor times at 1/2/4/8/16 threads with auto_threads off (truth data:
    // benchmarks/results/thread_sched_truth2, fit summary fit.md there,
    // tools/bench/fit_thread_costs.py): loss 3.41 against 4.15 for the
    // previous defaults (per_thread 1e5, no split terms); picks slower than
    // 1 thread by more than 5%: 0; 16-thread-max picks slower than the
    // 8-thread-max pick by more than 5%: 1 (ncvxqp7 ipopt, 1.051, noise).
    // per_thread 5e5-7e5 with per_split_front_thread 2e5-5e5 at a cap of
    // 7-8 all lie within about 3% of that loss.
    double         thread_cost_per_thread    = 7e5;
    double         thread_cost_per_top_front = 5e3;
    // Per tile-split top front and thread (team set-up and barriers). 5e5,
    // not the fit's least-loss 3e5 (loss 3.36): at 3e5 cops_elec_200
    // (recommended) picks 2 threads, 0.07% outside the 2% tie band, and its
    // delay-heavy split fronts make that 20% slower than 1 thread.
    double         thread_cost_per_split_front_thread = 5e5;
    // Threads past which a tile-split front's work stops speeding up. 8 is
    // the core count of one CCD on the 9950X3D where it was fitted: tile
    // teams spanning both CCDs gained nothing, and aug3dcqp, bratu3d and
    // TSOPF_FS_b300_c3 (recommended track) and ncvxqp7 (ipopt) ran 5-26%
    // slower at 16 threads than at 8. Open follow-up: detect the size of
    // the L3-sharing cluster at run time. Values below 1 count as 1.
    int            thread_split_scaling_cap  = 8;
    // The solve's thread count, chosen at factorization time from its own
    // cost model (thread_plan.hpp, plan_solve_threads; FactorKeep::
    // solve_threads) when the factorization leaves a parallel solve layout
    // and auto_threads is on:
    // the subtrees (the larger of the heaviest and their work / T), then
    // the top part's heaviest path, plus these terms per thread, per top
    // step and per top row and thread (the zeroed and summed buffers), in
    // step-work units (p f). Fitted like the factor terms above (same truth
    // data, fit.md): loss 2.46 against 8.08 for the previous defaults
    // (per_top_step 2e3, per_top_row 2.0); picks slower than 1 thread by
    // more than 5%: 0.
    double         solve_cost_per_thread   = 2e4;
    double         solve_cost_per_top_step = 100;
    double         solve_cost_per_top_row  = 1.0;
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
