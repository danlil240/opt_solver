# Thread scheduling: more threads never lose

**Date:** 2026-10-30 to 2026-10-31
**Branch:** `feat/small-tree-serial` (on main `7ee12e4` + `feat/sod-next-step`)
**Plan:** `docs/superpowers/plans/2026-10-30-thread-scheduling.md`
**Machine:** Ryzen 9 9950X3D (2 CCDs of 8 cores), MKL, Windows 11.

## Result

With `Control::auto_threads` on (the default), factor and solve at 2-16 threads are no slower than at
1 thread beyond measurement noise. The large trees are faster than the old engine at 8 and 16
threads.

| case (track) | factor 1T → 8T → 16T, before | after | solve 1T → 8T → 16T, before | after |
|---|---|---|---|---|
| robot_arm_8k (ipopt) | 11.2 → 32.9 → 108 | 10.5 → 10.5 → 10.5 | 36 → 80 → 83 | 35.0 → 36.2 → 36.1 |
| cops_elec_200 (ipopt) | 5.0 → 7.9 → 10.2 | 4.4 → 4.5 → 4.5 | 7.3 → 8.0 → 10.1 | 7.2 → 7.3 → 7.4 |
| cops_polygon_200 (ipopt) | 3.3 → 7.1 → 7.9 | 3.1 → 3.1 → 3.2 | 8.4 → 14.3 → 14.4 | 7.5 → 7.7 → 8.6 ¹ |
| TSOPF_FS_b300_c3 (rec) | 1355 → 1012 → 1113 | 1294 → 969 → 970 | — | — |
| nd24k (SPD) | 30,101 → 16,895 → 16,925 | 29,819 → 16,740 → 16,768 | — | — |

- **Before:** the front-tier board, main `7ee12e4` + `feat/sod-next-step`
  (`benchmarks/results/fronttier_20261030-t816-merged`).
- **After:** acceptance run 3 (`benchmarks/results/thread_sched_final3`, `599aeba`, 5 trials, median
  per step).
- ¹ See the 11-trial recheck below.
- nd24k stays flat past 8 threads. Its root fronts are SPD and run serially inside; intra-front SPD
  parallelism is a follow-up.

## What changed

| commits | change |
|---|---|
| `d4ff5ef`, `0b1fee0` | A persistent worker pool replaces three rounds of `std::thread` creation per factorization. |
| `b354516` | Top-part scheduler: chain continuation and spin-then-sleep, instead of one mutex and `notify_all` per front. |
| `03da21b`, `91f2353`, `c65fe7e` | Thread count from a cost model of the tree (`thread_plan.hpp`), for the factor and, separately, for the solve; the plan is cached per analysis. |
| `ce27d5e`, `0bdcac0` | Tiny subtrees merged into grain-sized phase-1 tasks; partition kept per analysis. |
| `8c84238`, `b480de0`, `e1ec96e` | Calibrated cost constants and a capped tile-split term (below). |
| `ce29e83`, `68f7ec9` | Trees the model rules serial skip plan evaluation; one tree walk for plan and partition. |
| `4927d96`, `dcb56ef`, `72087fb`, `dd96520` | Buffer regrowth fix (TSOPF regression, below). |
| `9a08705`, `599aeba` | No matrix copy per factorization for user-supplied/Auto scaling; cache-sized factors gathered in pieces. |

### The TSOPF regression (`4927d96`..`dd96520`)

After the pool and the new scheduler, the TSOPF trees (hundreds of tile-split top fronts) were slower
than the `b70190b` engine at every thread count. The cause was buffer regrowth, not scheduling.

- Each pass keeps its factor-value and contribution buffers at their high-water mark across
  factorizations. With fresh threads started in order, the same pass took the same heavy work every
  time.
- With the persistent pool, which thread takes which work is a wake-up race. Heavy work landed on a
  pass that had never held it. The pass then grew: a reallocation, a copy, and the first touch of
  every new page. That cost 30-150 ms per factorization.

The fix:
- **Phase 2:** a chain of top fronts picks a pass with room (`PassLanes`), not its thread's pass.
- **Phase 1:** subtree roots' contribution blocks (≥ 4096 entries) go to a shared `ChunkArena`
  (16 MB chunks, own-size chunks reused best-fit), not the pass stacks. A block that no free
  own-size chunk holds reallocates the largest free one, so drifting root sizes over an IPOPT run
  add no chunks (`e846a9e`).
- **Phase 1:** the heaviest T tasks are dealt one per thread. A dealt task is claimed, not pinned:
  a thread that runs out of tasks takes one whose thread has not started it, so a late wake-up
  holds up nothing (`3fae7bf`).

### Small residuals (`9a08705`, `599aeba`)

- **Scaling copy:** user-supplied and Auto scaling copied the cleaned matrix on every factorization,
  only to read its order. The copy cost 15-450 µs, bimodally. It showed as a fixed T > 1 overhead
  on cops_catmix_800 and portfolio_20k.
- **Gather:** phase 3 copied each pass's factor with one large memmove, which streams past the cache.
  The first solve after a T > 1 factor then read the whole factor from DRAM (TSOPF_FS_b162_c1
  rec solve: 1.04-1.08 → 0.68-0.83 ms, against 0.81 ms at 1T). A factor of at most 32 MiB is now
  copied in 64 KiB pieces.

## The cost model

Defaults (`solver/include/smf/control.hpp`), fitted by `tools/bench/fit_thread_costs.py` against
truth data with `auto_threads` off at 1/2/4/8/16 threads (`benchmarks/results/thread_sched_truth2`,
`fit.md`):

| term | value |
|---|---|
| `thread_cost_per_thread` | 7e5 |
| `thread_cost_per_top_front` | 5e3 |
| `thread_cost_per_split_front_thread` | 5e5 |
| `thread_split_scaling_cap` | 8 |
| `solve_cost_per_thread` | 2e4 |
| `solve_cost_per_top_step` | 100 |
| `solve_cost_per_top_row` | 1.0 |

- **Factor:** loss 3.41 against 4.15 for the pre-fit defaults. A1 proxy 0, real-A2 1.
- **Solve:** loss 2.46 against 8.08. A1 proxy 0. Real-A2 at the fitted solve constants is 3
  (`fit.md`); solve A2 is not an acceptance criterion.
- **`per_split_front_thread`** is 5e5, not the least-loss 3e5 (loss 3.36). At 3e5, cops_elec_200 rec
  picks 2 threads, and its delay-heavy split fronts make that 20% slower than 1 thread.
- **`split_cap` 8** is one CCD of this machine. Tile teams spanning both CCDs gained nothing.
  Detecting the L3-sharing cluster at run time is a follow-up.
- The constants were fitted before the buffer fix. They were kept (ruling R28): the fix makes T > 1
  cheaper, which can only move picks up, and the acceptance runs measure the picks directly.

## Acceptance

| criterion | definition |
|---|---|
| A1 | no `FAC>1T` / `SOL>1T` flags (`tools/bench/thread_scaling.py`, ratio > 1.05, `--min-ms 1.0`) |
| A2 | no `16>8` flags |
| A3 | shifted geomean (10 ms) of median factor time at 8T and at 16T over the 58 case-tracks with total front work ≥ 1e9, against `thread_sched_baseline` (the `b70190b` engine), ≤ 1.00 |
| A4 | full ctest passes |

All runs are auto on, default costs, 1/2/4/8/16T, the large_kkt, hard_indef and spd classes.

| run | binary | trials | FAC>1T | SOL>1T | 16>8 | A3 8T | A3 16T | A4 |
|---|---|---|---|---|---|---|---|---|
| 1 (`thread_sched_final`) | `b480de0` | 3 | 13 | 1 | 6 | 1.016 ✗ | 0.971 | pass |
| 2 (`thread_sched_final2`) | `dd96520` | 5 | 2 | 3 | 2 | 0.913 | 0.875 | pass |
| 3 (`thread_sched_final3`) | `599aeba` | 5 | 4 | 2 | 3 | **0.901** | **0.868** | 72/72 |
| 3, 11-trial recheck of the flagged cases | `599aeba` | 11 | 1 ³ | 1 ³ | 0 | — | — | — |

A3 is computed over all outcomes, from the median `factor_ms` of each of the 58 case-tracks (one
script, `a3_all.py` in the session scratchpad, for every run). The run-1 values 1.016 / 0.971 are
that script's output; a Success-only variant gave 1.017 / 0.969.

Run 1 failed A1, A2 and A3. Two causes:
- a fixed overhead at T > 1 on trees the plan runs serially (fixed by `ce29e83`, `68f7ec9`);
- the TSOPF regression (fixed by `4927d96`..`dd96520`).

Run 2's 7 flags were rechecked with 11 trials (`thread_sched_final2_recheck`). Four were noise. Three
reproduced (cops_catmix_800, portfolio_20k rec, TSOPF_FS_b162_c1 solve), and `9a08705` and
`599aeba` fixed them.

**Run 3 flags.** Every one is on a case under 16 ms:

| case | track | flag | 5-trial ratio | 11-trial recheck |
|---|---|---|---|---|
| boyd1 | rec | FAC>1T | 2T 1.07 | 0.99-1.01, clear |
| cops_pinene_400 | rec | FAC>1T | 2T 1.07 | ≤ 1.03, clear |
| ncvxqp9 | ipopt | FAC>1T, 16>8 | 16T 1.08 | ≤ 1.02, clear |
| portfolio_20k | rec | FAC>1T, 16>8 | 16T 1.11 | ≤ 1.05 (8T), clear |
| cops_polygon_200 | ipopt | SOL>1T | 4T 1.10, 16T 1.14 | 2/8/16T 1.09-1.10, 4T 0.98: noise ³ |
| portfolio_20k | ipopt | SOL>1T | 2-16T 1.06-1.09 | solve clear; factor 4T 1.07: noise ³ |
| pde_control_120 | rec | 16>8 | factor 0.64 vs 0.61, solve 0.53 vs 0.43 | factor 0.61 vs 0.62, clear |

The recheck (`thread_sched_final3_recheck`, 11 trials) left 2 flags. Both are noise (ruling R32), so
A1 and A2 hold:

³ Notes on the two remaining flags:
- **portfolio_20k:** below the model's serial bound (total front work 4.26e6), so both tracks run the
  same serial engine at every T. The flagged cell moves between runs: ipopt factor 2T in run 2, ipopt
  solve in run 3, ipopt factor 4T in the recheck.
- **cops_polygon_200 ipopt:** the factor and the solve run on 1 thread at every T (`threads=1
  solve_threads=1`). Solve + refine is bimodal, about 7.5 or 8.5 ms per step at every T including
  1T; in run 2, every 1T trial was in the slow mode. An interleaved A/B, 12 rounds of T=1 against
  T=8 in alternating processes, measured 8.4 ms at 1T and 7.5 ms at 8T.

**Accuracy (not an acceptance item).** robot_arm_8k rec, d_pretok and turon_m are Inaccurate at
every thread count in every run, including the baseline. TSOPF_FS_b300_c3 rec sits at the tolerance:
omega is 2e-9 to 1.2e-8 against 1e-8 at T ≥ 2, depending on summation order. The baseline engine
failed 1 of 3 trials at 8T. Run 3 failed 1 of 5 at 16T (ruling R31).

## Front-tier board at 1, 8 and 16 threads

smf against the best accurate competitor, shifted geomean, by class/track. The competitor records
(`fronttier_20261029-ssidsfix/score_fm_ssidsfix_t816`) are the same in both columns.
- **Before:** `fronttier_20261030-t816-merged/score_merged`.
- **After:** `thread_sched_board`, with smf records from `thread_sched_final3` and
  `thread_sched_board_small` (`599aeba`).

Cells are tuning | held-out (case count in brackets when it differs).

| class / track | T | before | after |
|---|---|---|---|
| hard_indef / recommended | 1 | 0.915 \| 0.961 | **0.882 \| 0.931** |
| hard_indef / recommended | 8 | 0.612 \| 0.739 | **0.576 \| 0.709** |
| hard_indef / recommended | 16 | 0.613 \| 0.730 | **0.563 \| 0.645 (9)** ² |
| hard_indef / ipopt | 1 | 0.893 \| 0.996 | **0.861 \| 0.954** |
| hard_indef / ipopt | 8 | 0.657 \| 0.830 | **0.620 \| 0.799** |
| hard_indef / ipopt | 16 | 0.654 \| 0.827 | **0.604 \| 0.779** |
| large_kkt / recommended | 1 | 0.730 \| 0.726 | **0.676 \| 0.706** |
| large_kkt / recommended | 8 | 0.601 \| 0.415 | **0.533 \| 0.358** |
| large_kkt / recommended | 16 | 0.589 \| 0.421 | **0.501 \| 0.362** |
| large_kkt / ipopt | 1 | 0.811 \| 0.958 | **0.794 \| 0.943** |
| large_kkt / ipopt | 8 | 0.728 \| 0.616 | **0.618 \| 0.527** |
| large_kkt / ipopt | 16 | 0.776 \| 0.645 | **0.582 \| 0.524** |
| spd / recommended | 1 | 0.868 \| 0.944 | 0.864 \| 0.946 |
| spd / recommended | 8 | 0.512 \| 0.614 | **0.501 \| 0.593** |
| spd / recommended | 16 | 0.482 \| 0.611 | **0.461 \| 0.579** |
| small_kkt / recommended | 1 | 0.925 \| 0.989 | 0.910 \| 0.960 |
| small_kkt / ipopt | 1 | 0.533 \| 0.781 | 0.532 \| 0.774 |

² TSOPF_FS_b300_c3 rec drops out at 16T on the one Inaccurate trial (accuracy note above). Its time,
970 ms, is below the 8T time of the baseline engine.

The 1-thread cells move as well. These smf runs are from a different day, and `9a08705` removes a
matrix copy from every factorization.

## Follow-ups

- **CCD/L3 cluster size at run time** for `thread_split_scaling_cap` (8 = one CCD here).
- **Intra-front parallelism for SPD fronts and u < 1e-6 fronts:** tile Cholesky on the pool, and a
  parallel TPP trailing update. This is what lifts the 8 → 16 plateau on nd24k, c-big ipopt and
  ncvxqp3 ipopt.
- **TSOPF_FS_b162_c1 bimodality** (18.7 vs 23-24 ms at the same T, before the buffer fix). Which pool
  thread inherits the top chain decided whether a grown buffer was reused. `PassLanes` should have
  removed this; run 3 shows 18.0-18.4 ms at every T > 1.
- **cops_elec_200 delays on the tile path.** Split fronts with many delays lose to 1 thread. The cost
  model keeps this case serial (`per_split_front_thread` 5e5) instead of fixing the path.
- **Arena footprint:** the root-block arena holds about 1.9 GB on TSOPF_FS_b300_c3 between
  factorizations. It replaces larger pass-stack high-water marks, but is never trimmed.
- **`factor_posdef`'s user-supplied scaling path** still copies the matrix per factorization. The
  `perm_values` copy has not been examined.
- **The 32 MiB piecewise-gather threshold** was tuned on this machine's L3.
- **TSOPF_FS_b300_c3 rec accuracy** at the omega tolerance under non-deterministic summation order.
