# Matching trial: search order, and what does not shorten the searches

**Date:** 2026-11-01
**Branch:** `feat/matching-search` (on main `a2b32e2`)
**Machines:** Ryzen 9 9950X3D (profiling, surveys), i9-12900K (boards, same binary for A and B).

## Result

The analysis' matching trial now starts its shortest augmenting path searches from the free columns by
decreasing degree (`MatchingSearchOrder::Degree`) instead of column order. The matching is still a
maximum-product one; ties may resolve otherwise.

Front-tier board, smf only, against column order (12900K, shifted geomean of wall, lower is better):

| class | split | track | 1T | 8T |
|---|---|---|---:|---:|
| hard_indef | heldout | ipopt | 0.914 | 0.879 |
| hard_indef | heldout | recommended | 0.926 | 0.888 |
| hard_indef | tuning | ipopt | 0.959 | 0.940 |
| hard_indef | tuning | recommended | 0.964 | 0.933 |
| large_kkt | tuning | ipopt | 0.970 | 0.957 |
| large_kkt | tuning | recommended | 0.994 | 0.980 |
| small_kkt, spd | all | all | 0.995-1.004 | 0.981-1.002 |
| all 138 case-tracks | | | 0.972 | 0.952 |

- The gains are analysis time. The TSOPF_FS family is 0.80-0.91x at 1T and 0.69-0.83x at 8T,
  case39 0.66-0.69x, and cops_catmix_800 0.65-0.84x.
- Outcomes and accuracy are unchanged.
- **cvxqp3 is the one real loss** (1.06x at 1T, rec 1.15x at 8T). Its ordering changes and the
  factor takes a few more delays.
- pglib 9241 and 4661 rec at 8T move ±20-36% either way between runs. 9241 rec is inaccurate on both
  sides, and neither case runs a matching search that the order changes.

## Why the order matters

Edges scanned by the searches, as a multiple of the matrix's entries, for 50 indefinite matrices
(`bench_matching_trial`):

| matrix | column order | degree order |
|---|---:|---:|
| case39 | 39.0M | 2.5M |
| TSOPF_FS_b300_c3 | 253M | 47M |
| cops_catmix_800 | 1.8M | 0.28M |
| ncvxqp3 | 145M | 147M |
| total, 50 matrices | 1180M | 553M |

The matcher's time is 0.836x as a geomean over the 50.

In column order, a KKT system's constraint columns come last. By then the cheap rows are taken, so each
late search walks around them. Searched first, the high-degree columns find free rows close by.

## Scope: the analysis trial only

`compute_symmetric_matching` keeps column order. It serves `ScalingMethod::Matching` and the
scale-on-delay rescue on new values.

- With degree order there too, tie changes in the rescue's scale cost pglib_opf_case4661_sdet rec
  1.24-1.30x (one more rescale) and pglib500 rec 1.14x.
- They won robot_arm_8k rec 0.93x (delays 78k -> 33k).
- That is tie luck in both directions, so the change stays where its gains are.
- Reusing the analysis' scale for the analysed values stays valid: any maximum-product scaling serves.

## ncvxqp3/7: what the searches cost, and what did not help

ncvxqp3 runs 16,421 searches that scan 145M edges, 289x the graph. The last 10% of the searches do 83%
of that work, finalizing 11-29k rows each. They start from constraint columns after the free rows have
run out.

| tried | result |
|---|---|
| Ordering from the initial (greedy) matching, no exact searches | analyse 2.4 -> 0.6 s, but delays 17 -> 88k: ncvxqp3 rec 2.3x, ncvxqp7 rec 4.6x wall |
| No matching (`RealizedDelays`) | factor 34 s (ncvxqp3), 106 s (ncvxqp7) |
| Stop the searches at <= 2% / 0.5% free columns (partial scale) | ncvxqp3 wall 0.69 / 0.73-0.79 on the 5-step track, but delays 18 -> 800-4700 and +14-45% per refactorization: break-even near 35 factorizations; TSOPF 1.07-1.12x |
| Prefetch each column's rows before scanning it | none on ncvxqp; case39 +15% |
| Breadth-first relabelling for locality | none (1922 vs 1831 ms) |
| Reverse column order | pinene 5.6x faster, ncvxqp 20-35% slower |
| Tie-break by reverse index or by tight-edge count | reverse: 0.934 overall (only pinene gains); tight edges: within noise |

The exact matching is load-bearing on these matrices, and their tail searches are inherent to it.

## Measuring

- Compare edges scanned, not standalone matcher ms. On the Ryzen the latter drifts ±20% between runs
  (pde_control_250, which runs no search, included), and unpinned ncvxqp3 swings 1630-2200 ms with the
  same binary: L3-bound code depends on which CCD it lands on.
- Boards: A and B from one binary, on one machine.
