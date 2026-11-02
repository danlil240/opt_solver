# Auction matching for the analysis trial: killed at G2 (2026-11-02)

Plan: `docs/superpowers/plans/2026-11-02-auction-matching.md`. Tasks 1-4 ran; the Task 4 kill point tripped, so Tasks 5-6 were not run and the auction is not wired into `analyse()`.

## Result

| gate | criterion | result | pass |
|---|---|---|---|
| G1 | same optimum and status as Degree ShortestPath on all 50 | `prod_fail 0` in all 18 configurations run | yes |
| G2 | total `edges_auc` <= 0.6 x `edges_deg` (553.4M) | best configuration 2.51G = **4.5x** | **no** |
| G2 | ncvxqp3 `work_auc` <= 0.5 x `work_deg` | 0.22x (57.8 vs 267.0 passes) | yes |
| G2 | no matrix with `work_deg` > 1 above 1.5x | worst 91x (robot_arm_8k), stokes128 45x, c-big 36x | **no** |
| G2 | `mta_ms` geomean <= 0.8 x `mtd_ms` geomean | 1.76x | **no** |

Measured on the i9-12900K, 1 thread, pinned to the P-cores, with `bench_matching_trial --list list50.txt` at commit 9b732c4. The ShortestPath baseline (`ssp_survey`, 3 trials) is 553,411,365 edges. Its per-matrix values match the matching-search-order write-up's degree-order column: case39 2.33 passes, TSOPF_FS_b300_c3 3.55, ncvxqp3 267.

## Sweep

18 of the plan's 24 configurations ran (factor 4 and 7 complete, factor 10 at eps 1e-2). The sweep was stopped once it was clear that no configuration could reach 0.6x. Total `edges_auc`, in units of 1e9:

| eps_factor | eps_final | warm | cold |
|---|---|---|---|
| 4 | 1e-2 | 6.04 | 4.53 |
| 4 | 1e-3 | 6.25 | 4.70 |
| 4 | 1e-4 | 6.58 | 4.91 |
| 4 | 1e-6 | 7.27 | 5.48 |
| 7 | 1e-2 | 3.06 | 3.79 |
| 7 | 1e-3 | 3.37 | 4.12 |
| 7 | 1e-4 | 3.42 | 4.40 |
| 7 | 1e-6 | 3.88 | 4.94 |
| 10 | 1e-2 | **2.51** | 3.03 |

- A larger `eps_factor` and a larger `eps_final` both help, but the trend flattens long before 0.33G, which is 0.6x of 553M.
- Every configuration ended with `prod_fail 0`: the auction plus the exact repair always reached the ShortestPath optimum.

Selected matrices at the best configuration (factor 10, eps 1e-2, warm), in passes of 2 x lower entries:

| matrix | n | work_deg | work_auc | ratio | repair / n |
|---|---:|---:|---:|---:|---:|
| ncvxqp3 | 75,000 | 267.0 | 57.8 | **0.22** | 0.62 |
| cops_pinene_400 | 16,005 | 40.1 | 127.9 | 3.2 | 0.58 |
| TSOPF_FS_b300_c3 | 84,414 | 3.55 | 28.7 | 8.1 | 0.84 |
| case39 | 40,216 | 2.33 | 41.7 | 17.9 | 0.84 |
| darcy003 | 389,874 | 2.85 | 52.5 | 18.4 | 0.76 |
| stokes128 | 49,666 | 4.68 | 211.7 | 45.2 | 0.78 |
| pglib_opf_case9241_pegase | 400,511 | 19.4 | 45.3 | 2.3 | 0.51 |

The auction wins only where the shortest-path searches have a long tail: ncvxqp3 0.22x, ncvxqp7 0.32x, cvxqp3 0.54x. On everything else the trial needs only 2-5 passes, and the auction costs 3-90x that.

## Why

1. **The auction pays at least one full pass per scaling phase, but most of the corpus needs only a few passes in total.** Even with a warm start, eps-scaling runs about log(max cost / eps_final) / log(factor) phases of 7-10 rounds. Each phase restarts with a pass over the matched columns, and then bids. The trial's ShortestPath matcher finishes 40 of the 50 matrices in under 5 passes.
2. **The exact repair hands most of the work back to the searches.** A winning bid leaves its edge exactly eps above the column's second best. The repair recomputes v = min and keeps only tight edges, so it unmatches about every column that bid in the last phase: `repair` is 51-84% of n on the table above, and 55% summed over the corpus. The Degree-order searches then redo that work from feasible duals. So the auction's work is added on top of the searches rather than replacing them.
3. **Price wars on deficient sparse parts.** When the sparse part has no perfect matching, prices fall until the bid cap. Task 3 had to reset the duals on the rows reachable from the auction-free columns, otherwise duals near 8e4 reached the scaling. stokes128, robot_arm_8k and c-big are the worst ratios, and they are the cases with dense or deficient structure.

## What this leaves

- Matching time stays the analysis' largest piece on cops_pinene, case39, stokes128 and TSOPF. A replacement has to beat 2-5 passes on easy matrices, which a phase-based method cannot do.
- The one class the auction does help is the long search tail (ncvxqp3/7, cvxqp3). There, a hybrid could be worth a probe: run the ShortestPath searches, and switch to an auction on the still-free columns only once a search-work budget is exceeded. Run it as a probe first, with edges only and no code in `analyse()`.
- Goldberg-Kennedy cost-scaling (CSA), which the plan names as the follow-up, has the same per-phase floor as reason 1. It is not recommended without such a probe.
- Task 7 (the initial-rejection candidate-ordering cost on darcy003/mario002) does not depend on the auction and is still open.

## Code state

Branch `feat/auction-matching`:
- `124df5e`: survey baseline columns.
- `debd0c7`: auction kernel.
- `5e85cef`: `SymmetricMatcher` Auction engine and exact repair.
- `9b732c4`: prototype columns.
- This write-up.

Nothing calls `MatchingAlgorithm::Auction` outside the tests and `bench_matching_trial`. Every existing caller defaults to ShortestPath and is bitwise unchanged. The full suite is 72/73 on the 12900K; the one failure is the known Intel-only `ScaleOnDelay.ProbeRejectsAStaleReuseOnPglib`, which PR #19 fixes.
