# 8-thread engine, Stage 1 gate

**Date:** 2026-11-09 · **Build:** `perf/8t-p3-layout` (Parts 1-4: #66 -> #67 -> #68 -> #69), threaded MKL, cuDSS off ·
**Machine:** i9-12900K, P-cores · **Data:** `benchmarks/results/remote/e8t_gate1` (8 cases, smf + MKL PARDISO, same
day), `e8t_board_gate1` vs `e8t_board_base` (board, smf, 1/8/16T, 3 trials + 1 warmup) ·
**Spec:** `docs/superpowers/specs/2026-11-09-refactor-8t-engine-design.md`

## Verdict

**Fail, on one of the six target cases.** obstacle_surface_200 projects at 1.045 against PARDISO at 8 threads. The
other five beat PARDISO and the board improves at 8 and 16 threads. Per the spec's stop rule, Stage 2 waits for a
re-plan.

## 8 cases, projected IPOPT-run factorization time, smf / MKL PARDISO

| case | N | 8T before (e8t_base) | 8T gate | 1T gate | refactor 8T ms, base -> gate |
|---|---:|---:|---:|---:|---|
| pglib case1354_pegase | 46 | 0.895 | **0.368** | 0.451 | 5.90 -> 2.20 |
| pglib case2000_goc__sad | 75 | 1.440 | **0.463** | 0.501 | 12.94 -> 4.39 |
| portfolio_20k | 17 | 0.931 | **0.880** | 0.862 | 6.81 -> 5.57 |
| pde_control_250 | 12 | 1.030 | **0.922** | 0.851 | 31.55 -> 27.49 |
| pde_control_120 | 13 | 1.056 | **0.937** | 0.636 | 5.66 -> 4.99 |
| obstacle_surface_200 | 136 | 1.201 | **1.045** | 0.652 | 6.09 -> 5.24 |
| acopf_grid_80 | 25 | 1.467 | 1.174 | 0.829 | 8.86 -> 7.04 |
| acopf_grid_160 | 24 | 1.448 | 1.261 | 0.866 | 44.77 -> 38.41 |

PARDISO was re-measured the same day. obstacle_surface_200 needs about 4.5% more: 5.24 ms per refactorization
against PARDISO's ~5.0 ms; it is the case with the most factorizations (136), so its analyse is a small share.
acopf_grid_80 / _160 are Stage 2's targets (1.17 / 1.26 now; the spec projected 1.09 / 1.14 after Stage 1).

## Guards

| check | pass mark | result |
|---|---|---|
| board geomean, smf gate / base | each track <= 1.005 | 1T 0.999 / 0.998, 8T 0.930 / 0.962, 16T 0.935 / 0.965 — pass |
| board single cases > 5% slower | none on a 3-round rerun | 4 / 14 / 8 cells at 1 / 8 / 16T in this 3-trial board; not rerun (gate already failed) |
| inertia, validation | identical | inertia identical everywhere; one validation flip: pglib case500 recommended 16T, 2 of 3 trials "Inaccurate" with omega 9.74e-10 identical to the base (the threshold is borderline on that case) |
| peak RSS, 8 cases, 8T | <= base | lower on the five large cases (acopf_grid_160 611 -> 453 MB, pde_control_250 437 -> 390); higher on the three that now run in parallel (case1354 116 -> 128, case2000 277 -> 303, portfolio_20k 83 -> 117 MB) — fail as written |
| IPOPT end-to-end on the 8 cases | same termination and iterations | not run (gate failed on the main criterion) |

## What each part bought (8T refactor, from the per-part A/Bs)

- Part 1, thread plan priced in time: pglib 0.44-0.46x, portfolio 0.83x.
- Part 4, parallel value prep and setup reuse: acopf_grid_160 / obstacle 0.93x, pde_control_250 0.97x.
- Part 2, factor written in place: obstacle 0.84x, pde_control_250 / acopf_grid_80 0.89x, peak RSS 0.84-0.97x.
- Part 3, solve preparation off the serial path: pglib case2000 0.89x, acopf_grid_160 / obstacle 0.95x.

## Open items for the re-plan

- obstacle_surface_200: 4.5% short. Remaining serial work at 8T: the symbolic plan (~0.1 ms), the top-of-tree chain
  (Stage 2's target), value prep's scaling selection.
- Peak memory on the trees that now run in parallel (per-thread pass buffers): +10-40% on small KKTs.
- Cold first factorization 11-25% slower on a few large board cases on the 12900K only (Part 2's commit granularity).
- `SolveEndToEnd` pde_control_250 4T inverse-vs-trsv tolerance flaky (~1 in 3, pre-existing path).
