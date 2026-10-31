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

The remaining small moves are machine noise. darcy003 and mario002 read 8% slower on the board, but an
interleaved A/B of the three binaries gives the same 300-345 ms.

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
- **A rejected matching's scale applied with AMD** (temporary `SMF_REJ_SCALE` experiment).
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
