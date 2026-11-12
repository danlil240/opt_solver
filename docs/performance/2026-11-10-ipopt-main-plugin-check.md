# IPOPT end-to-end check of main's plugin (run `nlpm1`, 2026-11-10)

**Question:** does main `63bdc35` (8T engine #65-#73, including the two new defaults: IPOPT runs expect 25
refactorizations, and parallel front assembly from 256 rows) help or hurt IPOPT+smf on the Mittelmann AMPL-NLP
set? Baseline is run `nlp1` (2026-11-08, same 12900K, plugin before the 8T engine).

**Setup:** 12900K, smf8 only (8 threads), 45 problems, 1800 s per-problem limit, plugin `smf_ma97.dll` built
from the #73 tip (tree equal to main). Records: `benchmarks/results/remote/nlp_mittelmann/nlpm1/`.
Cross-day comparison: treat differences under ~5% on one problem as noise.

**Note added 2026-11-12:** this run used stock IPOPT, whose MA97 interface (the one smf is loaded through) can
pick its strategy at the start of a solve on an uninitialised value (`docs/ipopt-ma97-interface-numneg.md`).
Iteration-count differences that this note calls luck may come from that bug; nothing here was re-measured with
a patched IPOPT.

## Result: better, no new failures

| | nlp1 (11-08) | nlpm1 (main) |
|---|---|---|
| Problems solved (of 45) | 41 | 42 |
| Linear-solver time, geomean over 22 problems with equal iterations and > 0.5 s | 1.00 | **0.884** |
| Wall time, same 22 problems | 1.00 | **0.936** |
| Mittelmann score, our smf8 column (KNITRO = 1, same-machine POUNCE = 4.46) | 6.32 | **6.12** |

- **No problem lost.** The one new solve is WM_CFy, which finished at 1794 s, just under the 1800 s limit
  (nlp1 hit the limit at 1803 s). It is no faster; count it as noise, not a gain.
- **Iterations:** 36 of 45 problems take the same number of iterations. None takes more than 1.5x as many; the
  biggest change is NARX_CFy, 796 -> 477 (fewer).
- **Big wins, cont5 family** (PDE control, same or near-same iterations): linear-solver time cont5_1_l 0.21x,
  cont5_2_2_l 0.27x, cont5_2_3_l 0.23x; wall 0.31-0.36x. cont5_2_1_l 0.88x. These are the problems where the
  "expect 25 refactorizations" default picks the cheaper-to-refactor ordering.
- **Typical problems:** linear-solver time 0.80-0.97x (elec, henon, lane_emden, dirichlet, ex*, robot_*,
  clnlbeam, corkscrw).
- **Small regression, qcqp family** (same iterations): linear-solver time qcqp1000-2c 1.08x, qcqp1500-1c 1.08x,
  qcqp750-2c 1.04x, qcqp500-3c 1.03x, qcqp1000-1nc 1.03x, qcqp750-2nc 1.03x. Each is near the noise line, but
  the whole family moves the same way, so it is probably real. Wall time on those problems is dominated by AMPL
  function evaluation, so it moves less (1.00-1.05x). qcqp1500-1c was already smf's worst per-iteration loss
  against MA57 (4.9x); this makes it slightly worse.
- **cont5_2_4_l** 1.08x with 14 -> 17 iterations: iteration luck, not a solver slowdown.
- dtoc1nd 1.33x is 0.10 -> 0.13 s, too small to read.

## What this changes

- The 8T engine is confirmed as a net win inside IPOPT, not only on replayed KKT sequences.
- Our gap to same-machine POUNCE in the score narrows from 6.32/4.46 = 1.42 to 6.12/4.46 = 1.37. The remaining gap
  is still robot_a/b/c (iteration-limit failures with every linear solver: an IPOPT-options problem) and the qcqp
  family (evaluation cost, plus qcqp1500-1c's per-iteration factor cost).
- **The qcqp regression is found and fixed: PR #74** (see below).

Scoring: `benchmarks/fixtures/mittelmann_nlp/score_nlp.py` on nlp1 records with smf8 replaced by nlpm1's smf8.

## Follow-up (2026-11-11): the qcqp slowdown came from #68 and is fixed in PR #74

**Bisect** (12900K, IPOPT smf8, qcqp1000-2c, plugin built at each merge commit, 2 runs each, factor time):
11-08 plugin, 0a7e8d6 (before the stack), #66 and #67: 19.0-19.3 s. **#68** (3e9bd0d, factor written in place in
per-thread slices), #69, #70, #71 and main: 20.5-20.8 s. Turning off either new default (#72 expected refactors
back to 10, #73 parallel assembly) or `factor_in_place` changes nothing on this problem.

**Cause:** delays. From IPOPT iteration ~78 on, qcqp1000-2c picks up up to 9.7k delayed pivots and the factor grows
from 1M predicted entries to 32M (per-factorization trace, `SMF_PLUGIN_STATS=2`). That is far past the factor
store's reservation (4x the predicted factor + largest front squared), so each such factorization moved the factor
to freshly allocated, zero-filled memory (~1.5 GB of new pages), and the next one freed it and did it again. Before
#68 the factor was a `std::vector`, which kept its capacity between factorizations.

**Fix (PR #74):** the store's next reservation for the same analysis follows the size the factor actually reached
(+25%), so later factorizations reuse committed pages. No change when nothing outgrows its reservation.

| linear-solver time, same day | before #68 | main | PR #74 |
|---|---|---|---|
| qcqp1000-2c | 19.4-19.5 s | 21.0-21.1 s | 19.7 s |
| qcqp1500-1c | 111.5-112.4 s | 119.8-119.9 s | 110.0-110.3 s |

PR #64 probe cases at 8T (one binary, the new rule switched on/off, 5 rounds): geomean 0.995, all within noise.
Tests 83/83 on both MKL builds. Data: `benchmarks/results/remote/nlp_mittelmann/bisect_qcqp*`,
`fix_factor_store_qcqp`, `benchmarks/results/remote/ab_store_floor_knob2`.
