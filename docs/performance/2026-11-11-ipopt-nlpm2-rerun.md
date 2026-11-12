# Mittelmann AMPL-NLP rerun with #74 and #75 (run `nlpm2`, 2026-11-11)

**Question:** PR #75 (plugin honours IPOPT's MC64 request after a delayed-pivot cascade) and PR #74 (factor store
reserves from the realized size) were checked on two qcqp problems only. Does the full set agree, and where does
the number against POUNCE land?

**Setup:** 12900K, smf8 only (8 threads), 45 problems, 1800 s per-problem limit, plugin `smf_ma97.dll` built from
main `93ff3d5`. Plugin trace on (`SMF_PLUGIN_STATS=2`, one log line per factorization), which `nlpm1` did not
have. Records: `benchmarks/results/remote/nlp_mittelmann/nlpm2/`. POUNCE, MA57 and the 11-08 smf8 times come from
run `nlp1`, so the comparison with POUNCE is cross-day.

**Note added 2026-11-12:** this run used stock IPOPT, whose MA97 interface (the one smf is loaded through) can
pick its strategy at the start of a solve on an uninitialised value (`docs/ipopt-ma97-interface-numneg.md`).
The plugin trace of this run shows that no problem was exposed: on all 45 the first factorization has the right
inertia, so the value was set before IPOPT first read it
(`python tools/bench/ipopt_numneg_exposure.py <run>/logs/smf8`). The iteration-count luck
described below (qcqp1500-1nc) is therefore not that bug. What remains of it here is deterministic and small:
after a factorization with too few negative eigenvalues IPOPT perturbs instead of first asking for better
quality, on 7 factorizations in total (NARX_CFy 1, WM_CFy 4, dtoc2 2). Not re-measured with a patched IPOPT.

## Result: 1.27 -> 1.18 against POUNCE, nothing lost

Measure: shifted geometric mean (10 s shift) of wall time over the 39 scored problems, a failed problem counted
as 7200 s, IPOPT+smf8 divided by POUNCE.

| | nlp1 (11-08) | nlpm1 (main 11-10) | nlpm2 (main 11-11) |
|---|---|---|---|
| IPOPT+smf8 / POUNCE (39 problems) | 1.317 | 1.265 | **1.181** |
| Problems solved (of 45) | 41 | 42 | 41 |
| Mittelmann table, our smf8 column (KNITRO = 1, our POUNCE 4.46) | 6.32 | 6.12 | **5.71** |

What moved it (nlpm2 against nlpm1):

| Problem | Iterations | Linear solver, s | Wall, s | MA57 linear solver, s | Cause |
|---|---|---|---|---|---|
| qcqp1500-1c | 99 -> 99 | 119.6 -> **4.8** | 186.1 -> 71.5 | 36.2 | #75, rule fired once |
| qcqp1000-2c | 99 -> 99 | 21.1 -> **2.3** | 37.9 -> 19.1 | 16.8 | #75, rule fired once |
| qcqp1500-1nc | 2806 -> 1411 | 286.5 -> 139.2 | 1499.8 -> 752.5 | 679.7 (1466 it) | roundoff, see below |

- **The two #75 problems reproduce the PR's numbers** inside the full run. They alone take the number from 1.265
  to 1.208.
- **qcqp1500-1nc is luck, not a fix.** The rule never fires on it (every factorization is `scale=-`). The two
  runs print identical iteration lines up to iteration 139 and differ in the third digit of one column at
  iteration 140; from there the paths separate and this run converges in half the iterations. The problem is
  nonconvex and MA57 takes 1466 iterations on it. It is worth 0.03 of the 1.181, so read the result as
  **1.18 measured, 1.21 without the lucky path**.
- **Where the rule fired:** qcqp1500-1c, qcqp1000-2c and cont5_2_4_l, once each. On cont5_2_4_l it is harmless
  (16 iterations against 17, linear solver 9.0 -> 8.7 s). It fired on no other problem.
- **Everything else is unchanged.** 40 of 45 problems take the same number of iterations as in `nlpm1`. On the 22
  of them with more than 0.5 s of linear-solver time (the two qcqp problems above left out), linear-solver time is
  1.011x and wall 1.005x, range 0.98-1.05. The trace output is part of that 1%.
- **No problem lost.** WM_CFy is listed as unsolved, but it is the time limit: 624 iterations in both runs,
  1794 s in `nlpm1` (just under the 1800 s limit) and 1802 s now. It is not one of the 39 scored problems.
- **Iteration changes:** NARX_CFy 477 -> 642 (it was 796 in `nlp1`; this problem moves from run to run and is not
  scored), arki0003 338 -> 323, cont5_2_1_l 40 -> 41. None is above 1.5x.
- **#74 cannot be confirmed from this run.** The slowdown it fixes was measurable on qcqp1500-1c and qcqp1000-2c,
  and both are now dominated by #75. The small qcqp problems sit at 0.99-1.05x of `nlp1`, inside noise.

## What is left against POUNCE

IPOPT+smf8 is faster than POUNCE on 31 of the 39 problems. The eight losses:

| Problem | IPOPT+smf8, s | POUNCE, s | Why |
|---|---|---|---|
| robot_a | fails (3000 iterations) | 64.4 | IPOPT's iteration, with every linear solver |
| robot_c | fails (3000 iterations) | 68.5 | same |
| robot_b | fails (3000 iterations) | 223.2 | same |
| qcqp500-3c | 546.5 | 14.2 | AMPL function evaluation (linear solver 1.3 s) |
| qcqp750-2nc | 441.0 | 15.6 | AMPL function evaluation (linear solver 1.0 s) |
| qcqp750-2c | 261.2 | 12.2 | AMPL function evaluation (linear solver 0.5 s) |
| qcqp500-3nc | 106.7 | 5.1 | AMPL function evaluation (linear solver 0.6 s) |
| clnlbeam | 64.9 | 33.8 | IPOPT's iteration: 2556 iterations against POUNCE's 501 (MA57 fails at 3000) |

- With robot_a/b/c solved in POUNCE's times the number is **0.75**; leaving the three out on both sides gives 0.72.
- None of the eight is a linear-solver loss. qcqp1500-1c was the last one on this set.
