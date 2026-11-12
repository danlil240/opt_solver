# Why IPOPT fails on robot_a/b/c and POUNCE does not (2026-11-11)

**Question:** on the Mittelmann AMPL-NLP set, IPOPT stops at 3000 iterations on robot_a, robot_b and robot_c with
every linear solver (smf, MA57, MUMPS). POUNCE, a Rust port of IPOPT, solves them in 64-223 s. These three
failures are what keeps IPOPT+smf behind POUNCE (1.18 with POUNCE = 1; see
`2026-11-11-ipopt-nlpm2-rerun.md`). What is different?

**Setup:** 12900K, IPOPT 3.14.20 + smf plugin from main `93ff3d5`, 8 threads. Probe records:
`benchmarks/results/remote/nlp_mittelmann/robot1/`, `robot2/`, full-set run `nlpx1/`.

**Note added 2026-11-12:** this run used stock IPOPT, whose MA97 interface (the one smf is loaded through) can
pick its strategy at the start of a solve on an uninitialised value (`docs/ipopt-ma97-interface-numneg.md`).
The plugin trace of the full-set run `nlpx1` shows that no problem was exposed: on all 45 the first factorization
has the right inertia (`python tools/bench/ipopt_numneg_exposure.py <run>/logs/smf8`). The
iteration-count luck mentioned below is therefore not that bug. Not re-measured with a patched IPOPT.

## Cause: the filter's constraint-violation cap

The three problems have 1001 variables and about 52,000 inequality constraints, no equalities, and the start
point is feasible.

- IPOPT's filter line search rejects any trial point whose constraint violation `theta` is above
  `theta_max = theta_max_fact * max(1, theta at the start point)`. The default `theta_max_fact` is 1e4 and the
  start is feasible, so the cap is 1e4.
- Good steps on these problems need a violation above that cap: robot_a and robot_b only solve with a cap of 1e7
  or more (Test 3). Once the cap binds, every large step is rejected. IPOPT then takes steps of about 1e-4 with
  11-17 backtracks each, the barrier parameter stays at 10^-1.7, and the run ends at 3000 iterations.
  Restoration is never entered (no restoration iteration in the log), because a tiny step is always accepted.
- POUNCE prints **the same iteration lines as IPOPT** (objective and violation agree to 6-7 digits) up to the
  point where IPOPT starts crawling: iteration 17 on robot_b. Its log then says why it continues:
  `theta_max raised 1.000e4 -> 1.000e6 (raise 1 of 4): 3 consecutive line searches had every trial rejected at
  the theta_max gate`, and a second raise to 1e8. IPOPT has no such rule.
- That rule fired on robot_a, robot_b and robot_c and on **none of the other 42 problems** of the set.

## Test 1: raise the cap for these three problems

`theta_max_fact=1e8`, nothing else changed:

| Problem | IPOPT+smf8, default | IPOPT+smf8, `theta_max_fact=1e8` | POUNCE |
|---|---|---|---|
| robot_a | fails, 3000 iterations, 389 s | **122 iterations, 6.9 s** | 189 iterations, 64.4 s |
| robot_b | fails, 3000 iterations, 365 s | **247 iterations, 12.6 s** | 277 iterations, 223.2 s |
| robot_c | fails, 3000 iterations, 406 s | **109 iterations, 5.7 s** | 216 iterations, 68.5 s |

Objectives agree with POUNCE's to five digits (1.04320, 2.33310, 1.40598). With these three times and every
other problem as in run `nlpm2`, the number against POUNCE is **0.61** (from 1.18).

## Test 2: the option on the whole set is not free

Run `nlpx1`: all 45 problems with `theta_max_fact=1e8` (same plugin and settings as `nlpm2`).

| | `nlpm2` (default) | `nlpx1` (`theta_max_fact=1e8`) |
|---|---|---|
| IPOPT+smf8 / POUNCE (39 problems) | 1.181 | **0.768** |
| Solved (of 45) | 41 | 43 |

- **Gained:** robot_a, robot_b, robot_c.
- **Lost: henon_120.** Default: 266 iterations, 47 s. With the option the path changes at iteration 230, IPOPT
  enters restoration and is still there when the 1800 s limit ends the run.
- **Slower: qcqp750-2c**, 71 -> 104 iterations, 261 -> 345 s.
- qcqp1500-1nc is back at 2701 iterations (1411 in `nlpm2`, 2806 in `nlpm1`). Its path splits at a roundoff-level
  difference again, so this is the same luck as before, in the other direction.
- Small changes either way: svanberg 34 -> 31, lane_emden_120 84 -> 86, cont5_2_1_l 41 -> 39, cont5_2_4_l
  16 -> 19.
- The other 34 problems take the same iterations; on the 21 of them with more than 0.5 s of linear-solver time,
  linear-solver time is 0.994x and wall 1.008x.

Without the henon_120 loss the number would be 0.63; the rest of the distance to 0.61 is qcqp750-2c and
qcqp1500-1nc.

## Test 3: no single value works

| `theta_max_fact` | robot_a | robot_b | robot_c | henon_120 |
|---|---|---|---|---|
| 1e4 (default) | fails | fails | fails | 47 s |
| 1e5 | not solved in 300 s | not solved in 300 s | 595 it, 55.8 s | not solved in 300 s |
| 1e6 | fails, 3000 it | fails, 3000 it | 143 it, 9.0 s | not solved in 300 s |
| 1e7 | 199 it, 14.8 s | 308 it, 18.7 s | 136 it, 7.4 s | not solved in 300 s |
| 1e8 | 122 it, 6.9 s | 247 it, 12.6 s | 109 it, 5.7 s | not solved in 1800 s |

robot_a and robot_b need 1e7 or more; henon_120 is lost at every value above the default.

## Conclusion

- The three failures are IPOPT's globalization, not the linear solver. Nothing in smf or its plugin can change
  them.
- A global option is the wrong tool: it has to be per-problem (not a fair benchmark setting) or it costs
  henon_120.
- What works is what POUNCE does: raise the cap **only after the line search has been rejected at it several
  times in a row**. On this set that rule touches the three robot problems and nothing else, so the expected
  result of the same rule in IPOPT is the 0.61 above. It needs a change in IPOPT's filter acceptor
  (`IpFilterLSAcceptor`), which means building IPOPT from source.

## Side finding: clnlbeam

clnlbeam (IPOPT+smf8 2556 iterations, 64.9 s; POUNCE 501 iterations, 33.8 s) is path luck at 8 threads. The same
IPOPT+smf at 1 thread solves it in 483 iterations and 12.6 s, IPOPT+MUMPS in 543 iterations, and IPOPT+MA57 fails
at 3000. All of them take steps of 1e-4 to 1e-2 for hundreds of iterations on this problem.
