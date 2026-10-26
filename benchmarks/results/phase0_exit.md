# Phase 0 exit report

Hard IPOPT corpus, single-thread baseline. Pressure columns come from the smf plugin trace.

| model | split | n | factorizations | corrections | corr frac | delays/fact | 0.01 n | qualified by | ma27 status (iters) | mumps status (iters) | smf status (iters) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pglib_opf_case118_ieee | tuning | 4830 | 27 | 0 | 0.000 | 648.44 | 48.30 | delays | Optimal Solution Found. (26) | Optimal Solution Found. (26) | Optimal Solution Found. (26) |
| pglib_opf_case300_ieee | heldout | 10955 | 37 | 0 | 0.000 | 1838.46 | 109.55 | delays | Optimal Solution Found. (36) | Optimal Solution Found. (36) | Optimal Solution Found. (36) |
| pglib_opf_case500_goc__api | tuning | 18889 | 53 | 1 | 0.019 | 2850.75 | 188.89 | delays | Optimal Solution Found. (51) | Optimal Solution Found. (51) | Optimal Solution Found. (51) |
| pglib_opf_case1354_pegase | heldout | 50759 | 46 | 0 | 0.000 | 8253.28 | 507.59 | delays | Optimal Solution Found. (44) | Optimal Solution Found. (44) | Optimal Solution Found. (44) |
| pglib_opf_case2000_goc__sad | heldout | 87432 | 75 | 18 | 0.240 | 15600.07 | 874.32 | both | Optimal Solution Found. (53) | Optimal Solution Found. (53) | Optimal Solution Found. (55) |
| pglib_opf_case2869_pegase | tuning | 115365 | 52 | 0 | 0.000 | 19985.81 | 1153.65 | delays | Optimal Solution Found. (50) | Optimal Solution Found. (50) | Optimal Solution Found. (50) |
| pglib_opf_case4661_sdet | tuning | 159871 | 60 | 0 | 0.000 | 30482.10 | 1598.71 | delays | Optimal Solution Found. (57) | Optimal Solution Found. (58) | Optimal Solution Found. (57) |
| pglib_opf_case9241_pegase | heldout | 400511 | 73 | 3 | 0.041 | 67536.66 | 4005.11 | delays | Optimal Solution Found. (68) | Optimal Solution Found. (68) | Optimal Solution Found. (68) |
| cops_glider_400 | tuning | 3613 | 3197 | 1890 | 0.591 | 38.55 | 36.13 | both | Optimal Solution Found. (1347) | Optimal Solution Found. (1372) | Optimal Solution Found. (1306) |
| cops_rocket_800 | heldout | 5609 | 32 | 4 | 0.125 | 38.38 | 56.09 | corrections | Optimal Solution Found. (27) | Optimal Solution Found. (27) | Optimal Solution Found. (27) |
| cops_catmix_800 | tuning | 15200 | 19 | 0 | 0.000 | 759.84 | 152.00 | delays | Optimal Solution Found. (18) | Optimal Solution Found. (18) | Optimal Solution Found. (18) |
| cops_chain_800 | tuning | 5609 | 4814 | 1814 | 0.377 | 36149.23 | 56.09 | both | Optimal Solution Found. (1475) | Optimal Solution Found. (1571) | Maximum Number of Iterations Exceeded. (3000) |
| cops_pinene_400 | tuning | 16005 | 13 | 0 | 0.000 | 4101.31 | 160.05 | delays | Optimal Solution Found. (12) | Optimal Solution Found. (12) | Optimal Solution Found. (12) |
| cops_steering_800 | tuning | 7213 | 23 | 7 | 0.304 | 1853.39 | 72.13 | both | Optimal Solution Found. (15) | Optimal Solution Found. (15) | Optimal Solution Found. (15) |
| cops_elec_200 | tuning | 800 | 744 | 439 | 0.590 | 199.59 | 8.00 | both | Optimal Solution Found. (157) | Optimal Solution Found. (154) | Optimal Solution Found. (304) |
| cops_polygon_200 | heldout | 40600 | 70 | 29 | 0.414 | 1.09 | 406.00 | corrections | Optimal Solution Found. (40) | Optimal Solution Found. (40) | Optimal Solution Found. (40) |

Qualification: inertia_corrections/factorizations >= 0.1 or delays_total/factorizations >= 0.01 * n; only tuning models count.

Qualifying tuning models (10): pglib_opf_case118_ieee, pglib_opf_case500_goc__api, pglib_opf_case2869_pegase, pglib_opf_case4661_sdet, cops_glider_400, cops_catmix_800, cops_chain_800, cops_pinene_400, cops_steering_800, cops_elec_200

Qualifying by inertia corrections only (4): cops_glider_400, cops_chain_800, cops_steering_800, cops_elec_200. delays_total counts delay events, not distinct columns, so the delay criterion is weak evidence; the verdict above uses the plan's either-criterion rule.

**Phase 0 verdict: PASS** (need >= 3 tuning models)

## acopf_grid_160 accuracy triage

`acopf_grid_160` (n = 110931, 5 recorded KKT values) replayed as `Inaccurate` in the IPOPT track. Reproduced with the 1-thread `build/` binary: `bench_replay --solver smf --mode warm_sequence --track ipopt` (u = 1e-8, Auto scaling, matching ordering). The fixture holds 5 factorizations (steps 0..4); only step 4 fails.

| step | max abs A | eta_inf | omega | gate |
|---|---|---|---|---|
| 0 | 1.0e2 | 5.4e-17 | 1.7e-16 | ok |
| 1 | 8.1e4 | 4.5e-18 | 2.0e-14 | ok |
| 2 | 2.2e5 | 2.2e-18 | 1.5e-14 | ok |
| 3 | 1.5e9 | 5.7e-21 | 4.7e-12 | ok |
| 4 | 2.9e11 | 8.1e-19 | **1.4e-6** | fail (omega > 1e-8) |

Bisection, one replay each (smf, 1 thread; "before" = harness refinement as it was):

| setting | outcome | worst omega (step) |
|---|---|---|
| baseline (u 1e-8, Auto scaling, matching, 3 refinement steps) | Inaccurate | 1.4e-6 (4) |
| pivot_u 1e-8 / 0.01 | Inaccurate / Success | 1.4e-6 / 5.7e-12 |
| scaling Auto / None | Inaccurate / Inaccurate | 1.4e-6 / 2.0e-7 |
| ordering matching / AMD | Inaccurate / Inaccurate | 1.4e-6 / 1.5e-4 (AMD delays 159k pivots, 3x slower) |
| refinement steps 3 / 10 | Inaccurate / Inaccurate | 1.4e-6 / 1.4e-6 (identical; 0 steps also identical for steps 1-4) |
| pivot_u sweep 1e-10, 1e-6, 1e-4, 1e-3, 1e-2, 0.1 | Success, Inaccurate (2.9e-8), Success, Success, Success, Success | non-monotone in u |
| MA27 (u 1e-8) for reference | Inaccurate | 8.7e-4 (4) |

Root cause. This is not growth that only a larger u cures, and not scaling/ordering. On step 4 the factorization is good enough that one more correction solve takes omega from 1.4e-6 to 1.7e-15 (checked with up to 8 extra sweeps). The harness refinement never ran it: its stopping rule is normwise only (stop when eta <= 1e-14). With max|A| = 2.9e11 the normwise denominator ||A|| ||x|| is enormous, so eta = 8e-19 already at sweep 0, while rows whose own entries are small (|A||x|_i ~ 1, residual 2.8e-6) carry a componentwise error that the gate (omega <= 1e-8) rejects. That is why refinement steps 3 vs 10 vs 0 are identical, and why changing u moves omega around without a principled threshold (u = 1e-6 fails, 1e-10 passes narrowly at 6.9e-9). Real IPOPT with the smf plugin solves this model in 23 iterations like MA27 and MUMPS, so the failure is a replay-harness artifact (rhs = ones and a normwise-only refinement rule), not an smf defect.

Decision: code defect in the harness refinement, fixed test-first. The harness refinement moved to `solver/benchmarks/common/replay_refine.hpp` (solver-agnostic, solve callable) so a gtest can drive it; the stopping rule now requires both eta <= 1e-14 and omega <= 1e-10 and stops on stagnation of max(eta/1e-14, omega/1e-10). The track pivot threshold (u = 1e-8) is unchanged, and the accuracy gates are unchanged. Regression test: `Refinement.Acopf160` (failed with omega = 1.3819e-6 before the fix, passes after; skipped when the fixture is absent).

After the fix: acopf_grid_160 replay is `Success` (step 4: eta 4.1e-25, omega 1.7e-15); refinement time 20 ms -> 35 ms of 940 ms wall. MA27 on this model still fails (omega 8.7e-4 at step 4; its refinement stagnates), now unchanged by the fix.

Other 6 IPOPT KKT models, median of 7 interleaved replays, wall_ms (includes refinement), before -> after:

| model | wall before | wall after | change | refine before -> after (ms) | max omega before -> after |
|---|---|---|---|---|---|
| acopf_grid_80 | 133.1 | 134.5 | +1.1% | 3.8 -> 6.2 | 5.1e-9 -> 1.1e-11 |
| obstacle_surface_200 | 76.8 | 78.3 | +1.9% | 1.5 -> 3.1 | 1.0e-15 -> 1.0e-15 |
| pde_control_120 | 110.9 | 112.8 | +1.7% | 4.4 -> 5.4 | 9.8e-13 -> 9.8e-13 |
| pde_control_250 | 810.1 | 818.1 | +1.0% | 46.2 -> 52.5 | 2.2e-16 -> 2.2e-16 |
| portfolio_20k | 67.3 | 67.7 | +0.6% | 3.0 -> 4.2 | 1.3e-12 -> 1.3e-12 |
| robot_arm_8k | 172.8 | 178.2 | +3.1% | 33.8 -> 40.7 | 4.4e-11 -> 4.4e-11 |

All within the 5% budget (the added cost is the componentwise check, harness-side; factor and solve time are untouched). Side effect: acopf_grid_80, previously passing with omega 5.1e-9 (within a factor 2 of the gate), now sits at 1.1e-11.

New replay knobs (kept): `bench_replay --pivot-u X --scaling none|equil|matching|auto --ordering auto|amd|metis|matching --refine-steps N`, applied after the track defaults.

Note: `phase0_exit.md` is regenerated by `tools/bench/phase0_report.py`; this section is hand-written and would be lost on regeneration.
