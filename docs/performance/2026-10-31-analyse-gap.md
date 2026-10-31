# Analyse gap: AMD's quasi-dense rows and diagonal matchings

**Date:** 2026-10-31
**Branch:** `feat/analyse-gap` (on `feat/small-tree-serial` `1cec5e0`)
**Machine:** Ryzen 9 9950X3D, MKL, Windows 11.

## Result

The analysis phase lost to MA97/MA57 on several indefinite cases. Two commits remove the parts of it that
bought nothing:

| commit | change | 1T wall, new / old (ipopt / recommended track) |
|---|---|---|
| `8bb06e7` | AMD dense-row threshold alpha 10 → 2 | cops_polygon_200 0.61 / 0.60, boyd2 0.78 / 0.79, case39 0.92 / 0.85, TSOPF_FS_b39_c7 0.90 / 0.82, b39_c19 0.84 / 0.77, b39_c30 0.93 / 0.98, b162_c4 0.88 / 1.00 |
| `0884a69` | `MatchingTrial::Diagonal`: skip the trial when the matching is diagonal | boyd1 0.38 / 0.35 (8T: 0.39 / 0.37), portfolio_20k 0.85 / 0.82 (8T: 0.83 / 0.82) |

Every case stays accurate, and the unaffected controls are unchanged (stokes128, boyd1 for `8bb06e7`;
case39, cops_pinene_400, blockqp1 for `0884a69`). ctest passes 72/72.

### 1T board

The board is the front-tier manifest with smf only (3 trials, `benchmarks/results/analyse_gap_board_diag_1t`).
It is scored against the competitor records of `thread_sched_board`: shifted geomean of
(smf + 10 ms) / (best rival + 10 ms).

| class | split | before (`thread_sched_board`) | after |
|---|---|---|---|
| hard_indef | heldout | 0.942 | **0.840** |
| hard_indef | tuning | 0.871 | 0.868 |
| large_kkt | heldout | 0.816 | **0.706** |
| large_kkt | tuning | 0.735 | 0.739 |
| small_kkt | heldout / tuning | 0.862 / 0.696 | 0.870 / 0.697 |

At the branch tip (`06126ff`, `analyse_gap_board_tip_1t`) the scores are: hard_indef 0.851 heldout,
0.854 tuning (stokes128 rec +0.875 → +0.080); large_kkt 0.712 / 0.743; small_kkt 0.872 / 0.700. Run to run,
a class score moves by about ±0.01.

The remaining small moves are machine noise. darcy003 and mario002 read 8% slower on the board, but an
interleaved A/B of the three binaries gives the same 300-345 ms.

### 8T and 16T boards

`analyse_gap_board_diag_8t` / `_16t`, scored against `thread_sched_board` at the same thread count:

| class | split | 8T before → after | 16T before → after |
|---|---|---|---|
| hard_indef | heldout | 0.753 → **0.685** | 0.727 → **0.663** |
| hard_indef | tuning | 0.598 → 0.598 | 0.583 → 0.586 |
| large_kkt | heldout | 0.434 → **0.397** | 0.435 → **0.390** |
| large_kkt | tuning | 0.576 → 0.581 | 0.542 → 0.547 |

The ±0.06 per-case drifts on untouched cases (helm3d01, Dubcova3, obstacle_surface_200) are the noise
level. TSOPF_FS_b300_c3 rec fails at 8T and 16T because it sits on the omega tolerance (see the
thread-scheduling write-up).

### Review fixes (`5f47dfd`)

- A diagonal trial's scale is a partial matching's: the dense columns are unmatched. Before the fix, Auto
  treated it as the analysed values' own matching scaling and disarmed the scale-on-delay rescue.
  `AnalysisKeep::matching_scale_partial` now keeps the rescue armed (it computes the full matching), and
  ScalingMethod::Matching never reuses a partial scale.
- AMD alpha 2 applies to graphs of 10000 vertices or more. Smaller graphs, such as the parallel nested
  dissection's leaves, keep the library default.
- Factors are identical to `0884a69` on the affected cases.

### A nearly complete initial matching keeps its scale (`06126ff`)

An initial rejection previously kept AMD with no scaling. Now, when the initial matching left at most 2%
of the columns free and the pivot threshold is strong (u >= 1e-6), its scale is applied as a partial
scale, with the rescue still armed.

- stokes128 recommended (1.0% free): 54k → 17k delays, wall 159 → 65 ms at 1T (0.40), 85 → 57 ms at 8T
  (0.67), accurate.
- Of the 108 matrices, stokes128 is the only initial rejecter that qualifies. PGLib OPF is 13.5-15% free,
  darcy003 and mario002 20%, ncvxqp9 31%, turon_m 38%.
- Without the free-share gate, PGLib paid 2-9% for a scale that leaves its delays in place. At
  u = 1e-8 the scale only costs time. Without the rescue armed, pglib_opf_case4661_sdet rec turned
  inaccurate.
- The rule was found on a tuning case (stokes128). The gate keeps it to initial matchings whose duals are
  nearly the full matching's.

## Where the analysis time went

`bench_matching_trial --pieces` splits the trial's matching phase into the matcher's constructor (cost
graph, greedy and length-2), the budgeted searches, the rest of the searches, and the candidate ordering
(compressed graph, compressed AMD and the candidate's exact flops).

- On initial rejections the candidate ordering is a third to a half of the phase. stokes128 spends
  4 + 7.6 + 17.5 ms; darcy003 spends 34 + 63 + 80 ms.
- AMD itself was slow on quasi-dense rows. cops_polygon_200 spent 15 ms of AMD on a 180k-entry factor:
  400 rows of degree 600, i.e. 3 sqrt(n), sit below the default 10 sqrt(n) threshold. Each elimination
  next to one of those rows recomputes its approximate degree. Its matching phase ran AMD again on the
  compressed graph (12 ms). With alpha = 2 those rows are ordered last, and AMD takes 0.5 ms for +2.8% exact
  flops. Across the 108 benchmark matrices only 8 orderings change.
- boyd1's matcher matched every sparse column to its own diagonal. Its 18 dense columns then ran 18
  phase-3 searches, each scanning the whole 1.2M-edge graph: about 70 of 98 ms, for an adopted ordering
  with flops ratio 1.002. Only boyd1 and portfolio_20k have a diagonal sparse matching. Every other KKT
  is almost entirely off-diagonal.

## Pitfalls (tried, not kept)

- **RealizedDelays as the default trigger.** It wins where the trial changes nothing (boyd1 0.32,
  blockqp1 0.66, portfolio 0.72, polygon 0.73). It loses 1.5-2x where the trial adopts (case39, TSOPF,
  darcy, stokes, acopf, pglib ipopt).
- **Forced METIS.** Our METIS path is slow on these graphs: stokes128 analyse 189 ms, darcy003 1211 ms,
  all inside METIS's coarsening and refinement. Delays explode on the hard KKTs.
- **Dense columns left unmatched with a fixed-up scale.** Skipping the dense searches entirely, and
  scaling a free column to put its largest entry at 1, breaks cops_pinene_400 on the recommended track.
  Its neighbours scale to about 1e122, so the free columns' scaled diagonals underflow and give 5 zero
  pivots. The diagonal verdict keeps the dense columns' phase-2 duals instead, and only fires when the
  sparse part is diagonal.
- **Diagonal verdict without a scale.** boyd1 on the recommended track (u = 0.01) then cascades delays
  and factors in 14 s instead of 13 ms. Auto must keep the matching's scaling.
- **A rejected matching's scale applied with AMD, ungated** (temporary `SMF_REJ_SCALE` experiment;
  the gated form is `06126ff` above).
  - The initial matching's scale on an initial rejection gives stokes128 rec 159 → 65 ms and pglib rec
    -23 to -26%.
  - But pglib_opf_case4661_sdet rec turns inaccurate, the ipopt tracks get 5-8% slower, and the full
    scale of an exact rejection breaks cops_pinene_400 rec.
  - A scaling applied up front has no fallback when it is wrong, so it was not kept.

## Open levers

- **Matcher search speed** is the common cost.
  - In the analysis: case39 runs 10k searches of about 4k edges each; ncvxqp3 scans 144M edges.
  - In the recommended track's scale-on-delay rescue, after an initial rejection left no matching: the
    first factor after analysis costs 38.6 ms on ncvxqp9 against 8.3 ms per refactor, about 25 ms of it
    the matching.
  - MA97 avoids the matching entirely and factors through its delays.
- **Delay-heavy factors cost more than their flops.** pglib2000 rec spends 237 ms on 23M flops; MA97
  does 335M in 160 ms. Part of that is the rescue above. Part is our kernel on fronts swollen by delays:
  stokes128 rec's cold first factor takes 113 ms with 54k delays against MA97's 38 ms with 30k.
- **Initial-rejection cost.** The candidate ordering and its exact flops are paid only to reject
  (darcy003 80 ms, stokes128 17.5 ms). No cheap predictor separated the rejecters from the adopters (free
  fraction, off-diagonal share and r_init all overlap).

## Tools

- `bench_matching_trial --list <file> [--pieces] [--case X --sample S [--lines]]`, from a /Zi build. The
  new verdict prints as `diagonal`.
- The A/B scripts and per-case scoring helpers are in the session scratchpad, not in the repo.
