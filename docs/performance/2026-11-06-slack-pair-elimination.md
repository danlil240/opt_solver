# Slack-pair elimination at analysis level: design note and prototype (2026-11-06)

Phase 1 is a design note plus a standalone measurement (`solver/benchmarks/proto_slack_elim.cpp`, not wired into smf). Phase 2, wiring it into smf, is not started.

## The structure

IPOPT's augmented system for inequality constraints d(x) - s = 0, one pair (s_i, y_i) per inequality:

| row/col | x | s_i | y_i |
|---|---|---|---|
| s_i | 0 | Σ_i + δ_w | -1 |
| y_i | b_i (row i of J_d, k entries) | -1 | -δ_c |

The pair touches the rest of the matrix only through b_i.

## Detection (analysis, once per pattern)

- A pair is a column s of degree 1 with a stored diagonal, whose neighbour y has degree at most k_max + 1 (k_max = 8). None of y's other neighbours may be slacks.
- The prototype also catches degree-1 variables with a zero diagonal (a variable that appears in one constraint only). That is why it eliminates the pair as one 2x2 block P = [a_ss a_ys; a_ys a_yy], which is exact whatever a_ss is, instead of two 1x1 pivots.
- **Reduced pattern:** A on the kept rows, plus every pair's k x k block on its states. Analyse that reduced pattern with the normal analyse(); store the pair list, value slots, block target slots and the reduced→full index maps.

## Factorization

**Schur update (flat loop over pairs, no fronts):**
- det = a_ss a_yy - a_ys^2, and (P^-1)_yy = a_ss / det.
- The states receive -(P^-1)_yy b b^T, which is k(k+1)/2 adds into precomputed reduced slots.

**Then:** form the reduced values (gather the kept entries plus the staged updates) and factor them with the normal engine.

**Inertia:**
- Each pair contributes (+1, -1) when det < 0. When det > 0 it contributes 2 of the sign of a_ss + a_yy.
- That is exact: the reduced matrix is the Schur complement, and Sylvester's law of inertia holds.
- The prototype reproduces the full KKT's inertia exactly on all five problems.

**Threshold test (per pair):**
- The 2x2 pivot's growth bound, (|a_ss| + |a_ys|) * max|b| / |det| <= 1/u, the same form as the TPP kernel's 2x2 test.
- A pair that fails would be delayed by the general kernel.
- In the prototype, 22 pair-steps out of 17,602 x 40 fail at u = 1e-8 on the truck. Its solution was accurate all the same.

**Fallback:**
- **Wholesale:** factor the full KKT on its own (kept) analysis for that step. This needs both analyses kept.
- **Per pair (better):** keep a failing pair's s and y in the reduced pattern from the start, as "optional" rows with an identity coupling that is off by default. Cheaper is the wholesale fallback with a counter that disables the reduction after repeated fallbacks.
- Phase 2 should measure how often steps fall back on the robotics set and the front tier before choosing.

## Solve

- **Reduced right-hand side:** r_r - b (P^-1 r_p)_y for each pair.
- **Reduced solve:** the normal solve on the reduced factor.
- **Back-substitution:** x_p = P^-1 (r_p - [0; b^T x_r]) for each pair.
- Multiple right-hand sides use the same loops per column. Partial solves (SolveJob) map to the same split: forward plus diagonal on the pairs is P^-1 applied blockwise, with the transformation into the reduced right-hand side.

## How it composes with the rest of smf

- **Scaling:** the pair pass runs on scaled values (S A S). The reduced factor inherits S restricted to the kept rows. Simplest: apply scaling in the pair pass, then factor the reduced matrix unscaled.
- **Scale-on-delay rescue and its work limit:** run on the reduced factorization only; the pairs never delay.
- **Static mode:** a failing pair under static pivoting gets the static pivot as the general kernel would (perturb |det|). Off in phase 2 v1: the reduction is disarmed.
- **IpmSession:** the inertia it checks is the combined (pairs + reduced). The session's analysis reuse holds both analyses.
- **C API and plugin:** unchanged entry points. FactorKeep carries the pair data (P^-1 entries per pair, about 3 doubles each) and the reduced FactorKeep. The plugin's num_two count adds the pairs with det < 0 handled as 2x2 blocks.
- **Deterministic mode:** the pair pass is serial and in fixed order, so it is bitwise across thread counts. The reduced factorization keeps its deterministic guarantees.

## Prototype numbers

12900K, 1 thread, captured IPOPT KKT sequences, u = 1e-8, best of 5 runs. Times are ms per factorization step. "Reduced" is the pair pass plus smf's factorization of the reduced matrix.

| case | n | pairs | reduced n | reduced | full smf | full MA27 | reduced vs MA27 | solution (last step, b = 1) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| truck_trailer_100 | 36607 | 17602 | 1403 | **0.284** (0.180 + 0.105) | 3.009 | 1.965 | **6.9x faster** | omega 1.4e-13 (full smf 1.9e-12), x within 9.5e-13 |
| quad_forest_200 | 15150 | 4779 | 5592 | **0.717** | 1.269 | 1.814 | 2.5x | omega 9.0e-13, x within 4.4e-14 |
| arm7_100 | 8239 | 2376 | 3487 | **0.578** | 0.895 | 2.308 | 4.0x | omega 3.6e-12, x within 3.7e-14 |
| drone_swap_200 | 19179 | 1203 | 16773 | 6.72 | 6.79 | 11.68 | 1.7x (no gain over smf) | unrefined omega 7e-3, full smf 1.8e-2 (both ill-conditioned) |
| car_track_200 | 11710 | 48 | 11614 | 4.34 | 4.39 | 4.94 | flat | unrefined omega 4e-2, full smf 3.4e-2 |

- Inertia: exact on every case. MA27 itself differs by one on the car.

## Verdict

- The full project clears "truck at or below MA27" by a wide margin (6.9x faster).
- It is a large win on inequality-heavy KKTs: quad 1.8x and arm 1.5x faster than smf full.
- It is neutral where inequalities are few (swap, car).
- Phase 2 risk is in the integration (fallback policy, FactorKeep/solve plumbing, the plugin), not in the numerics.
