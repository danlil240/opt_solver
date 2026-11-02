# Factor-side delays: TPP stalls, the delay guard, and the rescue's matching

**Date:** 2026-11-02
**Branch:** `feat/factor-delays` (on main `97b9f42`)
**Machines:** Ryzen 9 9950X3D (profiling, interleaved A/B), i9-12900K (full board, same pair of
binaries, sequential runs).

## Result

Three changes, one commit each:

1. **TPP kernel** (`ldlt_factor_front`): plain threshold fronts first pass over their columns
   once. A stalled panel pulls each failed column's largest partner in, or moves its failed
   columns behind the untried ones; the exhaustive loop retries what is left.
2. **Amalgamation**: CostModel's delay guard is gone.
3. **Scale-on-delay rescue**: its matching stops once at most 2% of the columns are free
   (`SymmetricMatcher::set_stop_free`), and scales by that partial matching's duals.

Board: smf only, the 126 indefinite case-tracks of the front-tier manifest, 1 thread, 3
interleaved rounds. Shifted geomean (10 ms) against main, lower is better:

| class | split | wall | factor |
|---|---|---:|---:|
| hard_indef | heldout | 0.969 | 0.955 |
| hard_indef | tuning | 0.986 | 0.976 |
| large_kkt | heldout | 0.933 | 0.920 |
| large_kkt | tuning | 0.952 | 0.948 |
| small_kkt | heldout | 0.925 | 0.909 |
| small_kkt | tuning | 0.953 | 0.945 |
| recommended track | | 0.933 | 0.914 |
| ipopt track | | 0.993 | 0.992 |
| all 126 | | **0.963** | **0.953** |

- No case-track loses accuracy. pglib_opf_case9241_pegase rec, inaccurate on main, passes
  (omega 1.3e-8 -> 9.6e-10).
- Backward error moves more than 10x on 6 case-tracks: 4 worse, all still at most 4.3e-11
  (cops_polygon_200 rec 5.5e-13 -> 4.3e-11, ncvxqp3 ipopt 2e-16 -> 2.6e-12), 2 better (cvxqp3
  ipopt 3.8e-11 -> 2.5e-16).
- One real loss: pglib_opf_case300_ieee rec 1.10x (one more rescue). Everything else is within
  the A/A noise floor of this setup (±5% per case, ±0.3% per class).
- 8 threads, 11 cases (22 case-tracks): 0.904, no accuracy change.

i9-12900K, full front-tier manifest (SPD included), 3 trials, wall:

| class | split | 1T | 8T |
|---|---|---:|---:|
| hard_indef | heldout / tuning | 0.975 / 0.984 | 0.971 / 0.984 |
| large_kkt | heldout / tuning | 0.945 / 0.933 | 0.938 / 0.947 |
| small_kkt | heldout / tuning | 0.928 / 0.955 | 0.948 / 0.956 |
| spd | heldout / tuning | 0.998 / 1.013 | 1.007 / 1.002 |
| recommended / ipopt track | | 0.938 / 0.998 | 0.946 / 0.993 |
| all 137 / 138 | | **0.965** | **0.968** |

The same accuracy picture: nothing lost, pglib 9241 rec fixed at both thread counts. Largest
gains: cops_elec_200 rec 0.44x (1T), pde_control_250 rec 0.59x, PGLib rec 0.62-0.73x. Single-case
losses are at most 1.11x on these sequential runs, with unchanged delays (cops_catmix_800 and
cops_polygon_200 ipopt at 8T; flat in the interleaved Ryzen A/B).

The two cases this started from, 1 thread (best rival from `thread_sched_board`, same machine):

| case | main wall / factor | now | best rival |
|---|---:|---:|---:|
| ncvxqp9 rec | 48.6 / 39.7 ms | 32.0 / 23.0 ms | MA97 16.0 / 10.3 ms |
| cops_elec_200 rec | 41.1 / 38.3 ms | 24.1 / 21.4 ms | MA97 21.9 / 17.4 ms |

Other large gains: pglib 1354 / 2869 / 4661 rec 0.60-0.69x (now well ahead of MA57),
pde_control_250 rec 0.66x, pde_control_120 rec 0.73x, cops_rocket_800 rec 0.75x.

## 1. The TPP kernel searched every failed column for every pivot

`process_front` puts the children's delayed columns first in the eligible block. In the blocked
kernel the first panel then holds only columns that already failed:

- it stalls;
- the full search scans every failing column, and its 2x2 partner's column, to find one pivot;
- that pivot updates the whole eligible block (rank 1 or 2);
- the next panel starts on the same failing columns, and stalls again.

A delay-heavy front cost a scan of each failed column per pivot, at BLAS-2 speed. ncvxqp9
unscaled at u = 0.01 factors 217 Mflop in 40 ms; MA97 factors 229 Mflop in 10 ms. Profile:
`select_pivot` 40%, `two_by_two_ok` 6%, MKL rank-1 kernels ~35%. Its top fronts carry 260-500
delayed columns against 7-109 own pivots (f = 646: 510 eligible, 99 accepted, 6.3 ms).

**Deferral.** A stalled panel moves its failed columns behind the untried ones, and the next
panel starts on fresh columns. The exhaustive loop afterwards tries the moved columns again, all
current. ncvxqp9 unscaled: 40 -> 19 ms. More deferral rounds before the exhaustive loop: no gain.

**Partners.** Deferral alone loses the cross-panel 2x2 pairs the full search found. A far
partner then pivots alone first (a 1x1 at u = 0.01 may grow entries 100x), and the failed
column later pivots on what is left:

- cops_polygon_200 rec, last step: omega 1.3e-14 -> 1.4e-8 (fails the gate);
- ncvxqp7 ipopt: 2e-16 -> 6e-11.

So a stalled panel first pulls in partners. Each failed column whose largest entry lies outside
the panel gets that column next to it, and the next panel starts on those pairs. The 2x2 is then
taken with in-panel updates, so no BLAS-2 step is needed. Pairing happens again only after a
pivot was taken, so the loop ends.

| | no pairing | pairing | full search per stall (BLAS-2) |
|---|---:|---:|---:|
| cops_polygon_200 rec factor / omega | 21.5 ms / 1.7e-9 | 19.5 ms / 4.3e-11 | 19.9 ms / 1.3e-12 |
| ncvxqp7 ipopt omega | 5.8e-11 | 2.1e-16 | 2.1e-16 |
| cops_elec_200 rec factor | 20.4 ms | 21.1 ms | 37.4 ms |
| pde_control_120 rec factor | 167 ms | 172 ms | 203 ms |

Static-pivot, in-order and SPD fronts are unchanged: they pivot or stop at the first failure, and
their order matters. Fronts below 96 rows take the unblocked path, untouched, so the small-front
kernels stay bitwise equal to it.

Tried, not kept:
- The APTP front kernel (`FrontKernel::Aptp`) is 10-20% slower on both cases. Its failed
  columns fall back to the same TPP search.
- Equilibration scaling: ncvxqp9 keeps 12.5k delays (matching: 3.6k).
- Letting the unscaled factor finish (no rescue): 40 ms with the old kernel, omega 1.9e-9.

## 2. The delay guard was protecting against the old kernel

CostModel did not merge into or out of a group that receives a certain delay (a width-1 leaf with
a zero diagonal). The guard came from cops_elec_200 at u = 1e-8 under the old kernel: merging its
chain gave flops +37% and factor +47%. That cost was the per-pivot search of section 1.

elec's tree under the guard is a chain of 25 fronts (8 pivots, ~600 rows each, a ~590² block to
copy and extend-add per link), fed by 200 zero-diagonal constraint leaves. Delays pile up the
chain: 8 -> 163 on the 13 ms step. Without the guard the chain is one front (171 Mflop, MA97's
count):

| cops_elec_200 factor | guard | no guard |
|---|---:|---:|
| rec, old kernel | 38.3 ms | 38.3 ms |
| rec, new kernel | 29.4 ms | 20.7 ms |
| ipopt, old kernel | 22.8 ms | 33.5 ms |
| ipopt, new kernel | 22.5 ms | 19.9 ms |

## 3. The rescue's matching: partial, not exact

ncvxqp9 rec at 1 thread, before:

| piece | ms |
|---|---:|
| first try, unscaled, stopped at 19x the delay-free work (6970 delays) | 8.0 |
| full matching for the rescue | 25.7 |
| rescaled factor (3580 delays) | 5.6 |
| probe solve | 0.5 |

The matcher is not slow by implementation. MA97 with MC64 scaling (`scaling = 1`) takes 29.7 ms to
factor ncvxqp9, about 25 ms of it MC64. MA97 wins by not scaling at all.

The PGLib rec rescues paid 67-370 ms per full matching. The analysis trial's budget snapshot is
too far from complete to reuse: 19.5% free on ncvxqp9, 6-7% on PGLib.

Stopping the searches at a free share (ncvxqp9 rec, factor time):

| free share | matching | rescaled factor delays | factor | omega |
|---|---:|---:|---:|---:|
| exact | 25.1 ms | 3588 | 35.5 ms | 2.8e-13 |
| 0.5% | 17.3 ms | 3770 | 28.0 ms | 9.6e-13 |
| 2% | 11.6 ms | 3989 | 22.7 ms | 1.7e-12 |
| 5% | 9.8 ms | 4502 | 21.6 ms | 7.9e-13 |

2% is the share up to which the analysis already keeps an initial matching's scale
(`kInitialScaleMaxFree`). The partial duals stay feasible, so every scaled entry is at most 1. An
all-zero column is still reported singular when the cost graph is built.

## Open

- **ncvxqp9 rec is still 2x MA97** (32 vs 16 ms): try 5.5 + matching 11.6 + factor 4.5 ms.
  - The matcher's construction and greedy phase are ~8 ms of the matching; the analysis already
    ran them on the same values and threw them away.
  - The stopped try is 5.5 ms.
- **cops_elec_200 rec at 8 threads is 36 ms against 24 ms at 1**. The merged root front takes the
  2D-tile APTP path, which handles its 200 zero-diagonal columns worse than the serial kernel:
  `bench_profile --top-tiles 0` at 8T gives 19.7 ms.
- pglib_opf_case300_ieee rec 1.10x: one more rescue under the partial scale.
