#pragma once
#include "smf/types.hpp"
#include <cstdint>
#include <limits>

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
    // the scale-on-delay rescue's own matcher stay ShortestPath (Column);
    // when the rescue reuses the analysis' matching scale (unchanged values)
    // it gets the trial's own scale (by default Hybrid's: ShortestPath's to
    // about 1e-14 thanks to the canonical duals). Hybrid runs the
    // shortest-path searches and, past 4 passes of search work, a warm
    // auction on the remaining tail (scaling_matching.hpp); up to that point
    // it is ShortestPath bitwise, so within the 2-pass budget above no
    // verdict moves. Hybrid by default. Re-gate (12900K, interleaved A/B,
    // indefinite front tier, 3 rounds; f7dc144, canonical duals): no
    // accuracy loss (no case-track lost, ncvxqp1 rec omega 5.7e-12), 0 trial
    // verdict changes on 50 matrices, no case-track above 1.10x after a
    // 5-round rerun of the flagged sub-ms cases, ncvxqp3 analysis 2.34x (1T)
    // / 2.49x (8T) faster (wall 0.643x / 0.511x), ncvxqp7 analysis 1.76x
    // (1T) / 1.83-1.85x (8T), all-case wall geomean 0.990 (1T) / 0.983 (8T).
    // Accepted miss: ncvxqp7's 1T bar of 1.8x was missed by about 2%,
    // because about 0.6 s of its analysis lies outside the matcher; the
    // owner accepted it (2026-11-02). ShortestPath reproduces the old trial
    // (and is what ShortestPath-specific tests set).
    MatchingAlgorithm matching_algorithm = MatchingAlgorithm::Hybrid;
    // The auto matching trial's candidate ordering (compressed AMD and,
    // under the Auto rule, METIS on the pair-compressed graph) at
    // num_threads > 1, or in deterministic mode: its METIS is parallel
    // nested dissection (parallel_nd_order, as the base ordering's
    // AutoParallel METIS). Once the base ordering is done (any method) and
    // its AMD flops pass the kMetisFlopsPerRow rule, the candidate's METIS
    // and its flop prediction start speculatively on a helper thread
    // alongside its compressed AMD and AMD's flop prediction (same outputs
    // as the serial sequence; discarded when the candidate's own rule does
    // not try METIS, at the cost of waiting for it). While the base ordering
    // may still run its own parallel ND, the candidate takes half the
    // threads (about 1.5x num_threads at the overlap). The
    // order does not depend on the thread count (> 1, or any in
    // deterministic mode); at 1 thread outside deterministic mode nothing
    // changes. false: serial METIS_NodeND, no overlap.
    bool candidate_parallel_nd = true;
    // Skip-trial gate (on by default). On IPOPT-style
    // factorizations (pivot_u <= skip_trial_max_pivot_u: IPOPT drives HSL
    // solvers at u = 1e-8) whose base AMD factor is cheap, the auto matching
    // trial costs more analysis time than its verdict saves: analyse()
    // skips it when the base AMD ordering's predicted flops per row (sum of
    // squared exact column counts / n, not times expected_refactors) lie in
    // [skip_trial_min_flops_per_row, skip_trial_max_flops_per_row]. The
    // matcher's initial phase still runs (its diagonal verdict and partial
    // scale, MatchingTrial::Diagonal, are kept); only a non-diagonal trial
    // is skipped, and the analysis then keeps AMD with no matching scale
    // (AnalysisKeep::matching_rejected set, as for a rejected trial).
    // Info::matching_trial_skipped and Info::matching_gate_flops_per_row
    // report it. The decision uses AMD's ordering, which does not depend on
    // num_threads. Fit (skipab probe, 12900K 1T, ipopt track, tuning split
    // only): band [21.1, 8205] between boyd2 (8.7, 21.7x slower without the
    // trial) and pde_control_250 (8862, 1.26x); 9 tuning skips, 0 false
    // positives; held-out 8 skips, 0 false positives. Rounded to [20, 8e3].
    // Gate (12900K, interleaved A/B against the gate off, the front tier's
    // 66 ipopt-track KKT cases, 3 rounds; 5-round reruns of the flagged
    // ones): 17 skips at 1T and 8T on the ipopt track and on the IPOPT
    // plugin track (8T), no accuracy or inertia change, no case-track above
    // 1.10, the safety net never fired; wall geomean ipopt 1T 0.921 tuning /
    // 0.913 held-out, 8T 0.921 / 0.931, plugin 8T 0.908 / 0.935
    // (cops_pinene_400 0.32-0.36, stokes128 0.36-0.66, darcy003 and
    // mario002 0.63-0.67, PGLib OPF 0.77-0.88).
    bool           skip_trial_gate = true;
    double         skip_trial_max_pivot_u = 1e-6;
    double         skip_trial_min_flops_per_row = 20.0;
    double         skip_trial_max_flops_per_row = 8e3;
    // Safety net of a skipped trial: the scale-on-delay rescue
    // (scale_on_delay_ratio) is armed on an analysis whose trial the gate
    // skipped even below kWeakPivotRatio (where it is otherwise never
    // armed), with skip_trial_rescue_ratio as its work limit and without the
    // early stop or the tightened next limit: only a factorization whose
    // delays inflate the front work past that ratio stops and is redone
    // with the matching scaling of its values (boyd2 at u = 0.01: 0 -> 8652
    // delays and 1.5e7 -> 2.2e11 flops without the trial; the rescue
    // restored 1.5e7). Its other conditions (Auto scaling, no static
    // pivoting, delay budget or max_negative_pivots) still apply, so it
    // never arms on the IPOPT plugin (MA97 scaling 0 = None).
    // Info::skip_trial_rescues reports a firing. Negative ratio: off.
    // At pivot_u = 1e-6 exactly (kWeakPivotRatio, the gate's default bound)
    // the plain rescue is armed too and takes precedence, with its own
    // limits. A skipped analysis is marked matching_rejected: through the
    // IPOPT plugin an MC64 request is then answered with the unit scale, as
    // after a trial that rejected the matching (the plugin has no other
    // safety net for a skip).
    double         skip_trial_rescue_ratio = 16.0;

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
    // pays no matching (with scale_on_delay_carry off; on, the next one is
    // rescaled without a try). Not after a factor with the analysed values' own
    // matching scale (a new one would be the same). Negative: off.
    double         scale_on_delay_next_ratio = 4.0;
    // Conditional early stop of a first try whose limit is above
    // scale_on_delay_early_ratio times the delay-free front work F (the
    // full-ratio try; one under scale_on_delay_next_ratio is untouched): at
    // the front whose work takes the count past early_ratio x F, the try
    // stops, as a work-limit trip (the matching scaling follows), when its
    // front work so far, that front included, exceeds
    // scale_on_delay_early_proj times the delay-free work of the fronts
    // reached so far (that front included): the try's inflation
    // extrapolated. Otherwise it runs on to the full limit. The fronts are
    // taken in postorder whatever the engine, so the decision is the same
    // at every num_threads in deterministic mode. Probe (2026-11-02,
    // fronttier recommended track, 1T, full limit 16): tries that ended
    // past 16x measured 10.9-84.7 at the 4x crossing, tries let through
    // between 4x and 16x 4.0-6.4 (case500's first step, 6.25 and ending at
    // 16.6, is left to the full limit as before). Local 1T wall against the
    // full limit alone: pde_control_120 rec 0.66, cops_glider_400 0.83,
    // ncvxqp9 0.89; pglib300, pglib500 and cops_rocket_800 unchanged, with
    // the full limit's rescale counts (a fixed limit of 4 cost them extra
    // rescales, 1.06-1.20 wall). i9-12900K gate (interleaved A/B against the
    // rule off, small_kkt + large_kkt + hard_indef, 3 rounds, all splits; the
    // threshold was picked on all splits, so heldout is no independent
    // check): 1T recommended geomean 0.987 (tuning 0.995, heldout 0.965),
    // ipopt 0.999 (never armed); proj 7, 8 and 10 decide the same on the same
    // 12 case-tracks, no rescale count or accuracy changes. 1T wins (rec):
    // pde_control_120 0.61, cops_glider_400 0.83, ncvxqp9 0.85,
    // cops_chain_800 0.93, pglib case4661 0.94, case1354/2869 0.96. 8T:
    // recommended 0.993, ipopt 1.002 (pde_control_120 0.84, glider 0.81,
    // ncvxqp9 0.87; worst 1.09, cont-300 ipopt, a path never armed).
    // Negative early_ratio: off.
    double         scale_on_delay_early_ratio = 4.0;
    double         scale_on_delay_early_proj = 8.0;
    // The same conditional early stop on a next try (the one under
    // scale_on_delay_next_ratio after a costly let-through), at its own
    // crossing ratio and projection; only used with scale_on_delay_carry
    // off (a carried rescale runs no next try). Negative ratio: a next try
    // runs to its limit. Gated (2026-11-04, ratio 2, proj 4) and left off:
    // carry won every cell it changed.
    double         scale_on_delay_next_early_ratio = -1.0;
    double         scale_on_delay_next_early_proj = 4.0;
    // Carry the rescale forward: the factorization after a costly
    // let-through (the one that would run under scale_on_delay_next_ratio)
    // skips its unscaled try and starts from the matching scaling of its
    // own values, as after a limit trip; later ones go on from that kept
    // scale (scale_on_delay_reuse_tol) and never return to unscaled tries
    // for that analysis and FactorKeep. A sequence that never lets a factor
    // through past scale_on_delay_next_ratio is untouched. On the front
    // tier every next try tripped its limit (pglib case300 at 4.0x, cops
    // rocket 4.3x, pde_control_250 4.2x at 1T / 5.8x at 8T: a second
    // cascade before the matching). i9-12900K A/B (warm sequences, 3
    // rounds): pde_control_250 rec wall 0.78 (1T) / 0.87 (8T), case300 rec
    // 0.92 / 0.94, rocket 0.96 / 0.98, no outcome, inertia, rescale-count or
    // delay changes. Caveat: a costly let-through followed by values that
    // factor cheaply unscaled (cops_elec_200 once measured 2.2, 8.5, 2.2)
    // now pays a matching and a probe instead of the cheap try; elec_200
    // no longer lets a factor through past 4x (2.4x). The let-through test
    // reads the realized front work, which can differ with the thread
    // count outside deterministic mode, so near the next ratio one thread
    // count may carry and another not (before, only the next limit
    // differed). Needs scale_on_delay_next_ratio >= 0. False: the next try.
    bool           scale_on_delay_carry = true;
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
    // Static fallback after a cascade: where the scale-on-delay rescue is
    // armed (the recommended track: pivot_u >= kWeakPivotRatio, Auto
    // scaling; never IPOPT's u = 1e-8 or the IPOPT plugin), the first
    // factorization of an analysis into a FactorKeep that either trips the
    // work limit (or its early stop) or is let through past
    // scale_on_delay_next_ratio switches that analysis and FactorKeep to the
    // static path: the factor that tripped is redone on it instead of with a
    // matching scaling, and every later factorization runs on it directly
    // (no tries, fit tests, probes or matchings). The static path: no
    // delays, the best available pivot taken, |d| below 1e-8 * max|A|
    // raised (the delay-budget-0 path), scaled as Auto resolves it (never
    // with a rescue's matching scale, which cost accuracy there). A static
    // factor is kept only when it perturbed no pivot (and its growth stays
    // under 1e10): otherwise, as on badly scaled KKTs that only the
    // matching makes factorable, that work is spent, the rescue follows and
    // the sequence never returns to the static path (Info::static_rejects).
    // A factor let through stays as it is. A static-path factor sets
    // FactorKeep::static_path, which arms the backward-error refinement
    // trigger (refine_omega_target, refine_growth_threshold).
    // Info::static_fallback reports a static-path factor. The switch reads
    // the same work counts as the rescue's own decisions, so it is identical
    // at every num_threads in deterministic mode. False: the rescue as is.
    // OFF by default: over 33 warm rec sequences the net cost was 1.003 (1T)
    // / 1.014 (8T); only pde_control-like matrices gain, and every other
    // cascading sequence pays one rejected static try (+2-9%). With it off
    // the factorization path is bitwise the one before the fallback existed
    // (1T, 33/33 sequences). The guard and the backward-error refinement
    // trigger (FactorKeep::static_path) stay as machinery for an opt-in
    // fast mode.
    bool           static_fallback_after_cascade = false;
    // Pivoting policy of indefinite factorizations. Threshold (the default)
    // is the accurate mode documented everywhere else in this struct:
    // threshold pivoting with delays, the scale-on-delay rescue, exact
    // inertia.
    //
    // Pivoting::Static is an opt-in FAST MODE: PARDISO-like static
    // pivoting. Every indefinite factorization runs on the static path
    // described above: no pivot is ever delayed; a column with no pivot
    // that passes pivot_u is taken as the best available 1x1 pivot, and a
    // |d| below 1e-8 * max|A| (after scaling) is raised to that value (a
    // perturbed pivot, Info::static_pivots). It is faster where threshold
    // pivoting cascades delays (pde_control_250 / 120 rec at 8 threads:
    // 0.82 / 0.89 of PARDISO's time), at the cost of the guarantees:
    //  - Accuracy is NOT guaranteed on hard indefinite systems. The factor
    //    is of a perturbed matrix (perturbed pivots) and its growth is not
    //    bounded by 1 / pivot_u, so a solve may miss the backward-error gate
    //    even after refinement. Badly scaled KKT systems (OPF, COPS-like)
    //    may go Inaccurate where threshold pivoting succeeds. A singular
    //    matrix is not reported Singular for its raised zeros: the factor
    //    is Success with num_zero 0, and only inertia_exact = false (and
    //    static_pivots > 0) tells.
    //  - The reported inertia (Info::num_positive / num_negative / num_zero)
    //    is that of the PERTURBED matrix when a pivot was perturbed
    //    (Info::inertia_exact = false): every raised pivot counts by the
    //    sign of its perturbed value, one raised from an exact zero as +
    //    (as PARDISO counts it), so a zero-diagonal KKT is not reported
    //    Singular for its raised zeros (unless the caller set
    //    static_zero_stays_zero or static_pivots_as_zero, or on a fallback
    //    front: fallback_static_pivot_tol > 0). An interior-point
    //    method's inertia correction can be misled by it. (The IPOPT plugin keeps its honest-
    //    inertia guard in this mode: a factor with a perturbed pivot is
    //    reported to IPOPT as singular, MA97 flag 7, so IPOPT regularizes;
    //    on a KKT that stays perturbed whatever the regularization, IPOPT
    //    may then give up where threshold pivoting succeeds.)
    //  - The delay machinery and the scale-on-delay rescue (scale_on_delay_*,
    //    static_fallback_after_cascade, skip_trial_rescue_ratio) are
    //    disabled: there are no delays to rescue. A matching re-analysis the
    //    analysis already holds is still used.
    //  - tree_parallel is ignored (the legacy task-parallel engine has no
    //    static path): the default subtree engine at the same num_threads.
    //  - The backward-error refinement trigger: every static factor sets
    //    FactorKeep::static_path, and one that forced or perturbed a pivot
    //    (pivot_growth > 0 or static_pivots > 0) is refinement_advised():
    //    refine() refines it to omega <= refine_omega_target (1e-10).
    //    Solver::factor_solve, IpmSession (full solves), the C API's
    //    smf_solve (SMF_SOLVE_FULL) and the IPOPT plugin refine such a
    //    factor; after Solver::factor + Solver::solve, call refine() when
    //    refinement_advised(fkeep). One that forced no pivot is a threshold
    //    factorization and is solved without refinement.
    // Positive definite factorizations ignore it. Set it from a string with
    // parse_pivoting(); the IPOPT plugin reads SMF_PIVOTING=threshold|static
    // and the C API has smf_set_pivoting(). Fast mode is reported apart from
    // the headline benchmark numbers.
    Pivoting       pivoting = Pivoting::Threshold;

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
    // Front kernel of SPD fronts (see PosdefFrontKernel): Lapack = dpotrf +
    // dtrsm + dsyrk on fronts of >= posdef_lapack_min_front rows. Ignored in deterministic mode
    // (Ldlt there) and by the legacy tree_parallel engine. Lapack measured
    // (12900K, fronttier spd class, 2026-11-04) factor 0.896x at 1 thread,
    // 0.840x at 8 (G3_circuit 0.78x, nd24k 0.56x), every case <= 1.003x.
    PosdefFrontKernel posdef_front_kernel = PosdefFrontKernel::Lapack;
    // Smallest front (rows) on the Lapack SPD kernel; smaller fronts keep
    // Ldlt (the unblocked / small-front kernels). Values < 1 act as 1.
    // 64 (was 96 = LdltKernelParams::blocked_min_front): 12900K spd class,
    // 2026-11-04, factor 0.996x at 1 thread (every case <= 1.010x), 0.946x
    // at 8 (Dubcova3 0.81x, oilpan 0.81x); 8T noise is about +-8%.
    int            posdef_lapack_min_front = 64;

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
    // Slack-pair elimination (slack_elimination.hpp): at pivot_u <= 1e-6
    // (kSlackMaxPivotU) and an auto ordering, analyse() finds IPOPT's
    // inequality slack/multiplier pairs and, when they are at least
    // slack_min_share of the rows (Auto; On: any), analyses the reduced
    // matrix, and the full matrix pairs first after it; factor() eliminates
    // the pairs as 2x2 blocks in one pass (the condensed barrier step) and
    // factors the reduced matrix, solve() substitutes them back. A
    // factorization takes the full matrix instead when a pair's update
    // exceeds slack_update_limit times the state diagonal it lands on (<= 0:
    // no limit), and the reduction never runs with static pivoting,
    // reg_diag_sign, user-supplied scaling or max_negative_pivots. Above
    // kSlackMaxPivotU, or Off, nothing changes.
    // Opt-in (default Off): it pays on IPOPT problems with many inequality
    // slacks over long runs (trajectory optimization: 2x on a truck-trailer
    // planner), but its analysis costs more on short sequences and on
    // matrices whose reduced matrix factors no cheaper (PGLib AC-OPF).
    SlackElimination slack_elimination = SlackElimination::Off;
    double         slack_min_share = 0.2;
    double         slack_update_limit = 1e15;
    // A full solve through the plan is checked on the full matrix: when its
    // componentwise backward error exceeds slack_omega_target, up to
    // kSlackRefineSteps refinement steps with the reduced factor follow, and
    // if it still exceeds the target the factorization is redone on the full
    // matrix for that solve and the factor's later ones (Info::
    // slack_solve_fallbacks). <= 0: no check. 1e-8 (the board's omega gate)
    // catches the gross losses (Codex's huge-slack case: 1e-3); a tighter target
    // refines solves IPOPT's own refinement handles and changes its path.
    double         slack_omega_target = 1e-8;
    double         factor_memory_multiplier = 1.1;
    int            expected_refactors = 1;

    // Parallelism
    // num_threads > 1: independent subtrees of the assembly tree are
    // factored concurrently (serial BLAS each), then the top of the tree
    // under a budget of as many BLAS threads in total as the factorization
    // runs on (at most num_threads; auto_threads below picks how many).
    // tree_parallel opts into the older task-parallel engine instead, which
    // measured slower everywhere.
    // BLAS: every factor and solve call (Solver, IpmSession, the C API, the
    // MA97 plugin) holds the calling thread's BLAS at num_threads threads
    // for the whole call (CallBlasGuard, blas_thread_guard.hpp), the
    // caller's setting restored on return; per-front shares and serial BLAS
    // in subtrees and tiles nest inside. So num_threads = 1 is one BLAS
    // thread in every mode, also with threaded MKL whose global count is
    // every core (deterministic mode: one at every num_threads).
    // num_threads <= 0: no guard, one tree thread on the caller's own BLAS
    // setting (e.g. threaded MKL's global count). Not "auto": the IPOPT
    // plugin's helpers (ipopt_plugin_control.hpp) read their own
    // num_threads argument <= 0 as "the plugin default" and always pass a
    // count >= 1 here. OpenBLAS builds: its thread count is process-wide,
    // so concurrent calls with different num_threads (or a caller on a
    // different OpenBLAS count) race on it; MKL's setting is per thread.
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
    // tasks and untiled fronts of 1024 rows and more (their BLAS on the
    // threads) counting 1/T, and its work, whose tile-split share speeds up to
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
    // thread_plan_front_costs: the thread plan prices each front as
    // f^2 p + thread_cost_per_front_entry f^2 + thread_cost_per_front (work
    // units of f^2 p: assembly and fixed overhead on top of the flops), and
    // has no work floor (kBudgetRegionMinWork stays the delay-budget regions'
    // floor, which decide numerics). Off: flops alone above a 2e7 floor.
    // Fitted 2026-11-09 on 1-thread per-front traces (12900K, probe build of
    // probe/front-trace-8t, every front): the 8 PR #64 cases plus the board's
    // 66 ipopt-track cases, 25.0M fronts (results/remote/e8t_fronts_8,
    // e8t_fronts_ipopt; tools/bench/fit_front_costs.py, relative error, R2
    // 0.947): t = a f^2 p + b f^2 + c gives b / a = 45. The fixed cost per
    // front (c) is below what the traces resolve (their own print cost is
    // ~1.4 us per front), so 0. PGLib's tiny fronts cost ~6x the big fronts'
    // time per f^2 p unit: the f^2 term prices it.
    bool           thread_plan_front_costs   = true;
    // parallel_value_prep: with more than one thread and at least
    // parallel_value_prep_min_entries matrix entries, mapping the values to the
    // permuted pattern and scaling them run over ranges on the factorization's
    // pool (bit-for-bit the serial result).
    bool           parallel_value_prep       = true;
    // factor_in_place: the tree-parallel factorization's threads write the
    // factor and its row ids straight into per-thread slices of the
    // FactorKeep's FactorStore (no gather copy); a slice that outgrows its
    // reservation falls back to the gather (Info::factor_store_regathers).
    bool           factor_in_place           = true;
    // parallel_dinv: after a tree-parallel factorization the solve's D^-1
    // table is built over ranges of solve steps on the factorization's pool
    // (bit-for-bit the serial build).
    bool           parallel_dinv             = false;
    // lazy_diag_entries: a tree-parallel factorization leaves FactorKeep's D
    // entries in its threads' lists; FactorKeep::diag() merges them into
    // elimination order on first use (Info::two_by_two_pivots counts the 2x2
    // blocks without it).
    bool           lazy_diag_entries         = false;
    std::size_t    parallel_value_prep_min_entries = 100000;
    double         thread_cost_per_front_entry = 45.0;
    double         thread_cost_per_front     = 0.0;
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
    // TPP at every thread count): fronts of at least 2048 rows (512 in
    // deterministic mode, which tiles them at every thread count) at the top of
    // the tree that pivot with plain threshold pivoting (indefinite, no static or
    // in-order pivoting, not a fallback / delay-budget static front) are
    // factored as 2D tile tasks on the tree-parallel thread pool, serial
    // BLAS per task (APTP block-column structure, whatever front_kernel
    // says), instead of one threaded-BLAS kernel call. top_tile_nb: their
    // block-column / tile width.
    bool           top_tiles     = true;
    int            top_tile_nb   = 256;
    // > 0: tile fronts of at least this many rows instead of the default
    // (2048, and 512 in deterministic mode; top_tile_min_front). For tests
    // and tuning. Deterministic mode stays deterministic with any value
    // (it applies at every thread count).
    int            top_tile_min_rows = 0;
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
    // The small-system solve (solve_interleaved.hpp): D^-1 precomputed per
    // factorization (FactorKeep::dinv), up to kInterleavedMaxRhs right-hand
    // sides of a serial solve interleaved row by row, and every serial solve
    // of a factor of small fronts only (FactorKeep::solve_small_fronts) on
    // those kernels, one right-hand side in place. Off: the front kernels of
    // solve_forward / solve_backward as before. On the trajectory KKTs (949
    // fronts of 2-4 pivots) IPOPT's solve time drops 12 -> 4 ms.
    bool           small_solve_kernels           = true;
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
    // Backward-error refinement trigger of a static-path factor
    // (FactorKeep::static_path: static_fallback_after_cascade, Pivoting::
    // Static). Such a factor took pivots that fail the pivot_u test, so its
    // growth is not bounded by 1/pivot_u: refine() then aims at a
    // componentwise backward error omega <= refine_omega_target (instead of
    // the usual 1e-8; the normwise gate is unchanged) within the usual step
    // caps. Callers that refine only after static pivots should also refine
    // such a factor when it forced a pivot (refinement_advised()).
    // refine_growth_threshold: opt-in, off (infinity) by default. A static-
    // path factor whose pivot growth (FactorKeep::pivot_growth: the largest
    // |l_ij| below a forced pivot) exceeds it takes at least one step even
    // when the first omega already meets the target. Off since the fast
    // mode counts raised exact zeros by sign: every zero-diagonal KKT then
    // has growth near 1e8 and above, and omega is measured on the true
    // residual, so a met target leaves the step nothing to improve (it cost
    // a solve and a residual per solve).
    double         refine_omega_target           = 1e-10;
    double         refine_growth_threshold       = std::numeric_limits<double>::infinity();

    // Input checking. True (the default): analyse() scans the pattern for
    // unsorted, duplicate and out-of-range entries and cleans them up.
    // False skips that O(nnz) scan for callers that guarantee clean input:
    // every column's rows sorted, unique and in [j, n) (AnalysisKeep::
    // input_clean of an earlier analyse of the same pattern says so); only
    // the O(n) shape checks and the missing-diagonal count remain. Unclean
    // input with false is undefined behaviour: unsorted, duplicate or
    // out-of-range rows lead to out-of-bounds writes and may corrupt memory.
    bool           check_input   = true;

    // Intel OpenMP runtime blocktime (KMP_BLOCKTIME) in ms for the threads
    // that drive MKL's thread teams: set on every thread that runs a front
    // under a BlasThreadGuard (pool workers, the calling thread) for the
    // guard's lifetime, the previous value restored after. Negative: leave
    // the runtime's own setting (KMP_BLOCKTIME or its default, 0 on the
    // 12900K) alone. Default 1 ms: libiomp5's team sleeps between the
    // hundreds of short MKL calls of a big front at 0 and wastes cores
    // spinning at 200; 1 ms keeps it hot within a front only (8T front tier,
    // 3 rounds: factor 0.98, wall 0.99 against unset; 1T unchanged). A no-op
    // unless the Intel runtime (libiomp5md.dll / libiomp5.so, MKL "intel"
    // threading) is already loaded; see blas_thread_guard.hpp. Never changes
    // results, only wake-up latency.
    int            mkl_blocktime_ms = 1;

    // Diagnostics / logging
    int            print_level   = 0;
};

} // namespace smf
