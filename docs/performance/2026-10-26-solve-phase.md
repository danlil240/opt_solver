# Solve phase: closing the backsolve gap to MA27, tree-parallel solve

**Date:** 2026-10-26
**Purpose:** the large-scale write-up left two solve-phase gaps open. Inside IPOPT, smf's backsolve took ~1.8x MA27's time (robot_arm 0.36 s vs 0.20 s). And the solve was serial while PARDISO's scales 2–3x with threads.

## Result

**IPOPT backsolve** (`LinearSystemBackSolve`, IPOPT's default MA97 options for smf, 1 thread). Both solvers were rerun back to back in `benchmarks/results/ipopt_e2e_v8_t1.json`, on a machine with ~15% background load. The "smf before" column is from the 2026-10-26 large-scale write-up.

| model | MA27 | smf before | smf now | smf IPOPT wall vs MA27 |
|---|---:|---:|---:|---:|
| pde_control_120 | 0.039 s | 0.048 s | **0.030 s** | 0.53 s vs 0.54 s |
| pde_control_250 | 0.258 s | 0.280 s | **0.188 s** | 2.94 s vs 4.89 s |
| robot_arm_8k | 0.135 s | 0.363 s | **0.133 s** | 2.41 s vs 6.90 s |
| obstacle_surface_200 | 1.376 s | 0.938 s | **1.148 s** ¹ | 13.5 s vs 15.3 s |
| acopf_grid_80 | 0.412 s | 0.071 s | **0.076 s** | 2.87 s vs 5.08 s |
| acopf_grid_160 | 3.767 s | 0.439 s | **0.495 s** | 14.1 s vs 35.1 s |
| portfolio_20k | 0.020 s | 0.014 s | **0.019 s** | 0.54 s vs 9.88 s |

¹ obstacle takes 133 IPOPT iterations here vs MA27's 130 (127 in the "before" run): pivoting changes the iterate path. Per solve, smf is faster.

smf's backsolve is now at or below MA27's on all 7 models, and its IPOPT wall time is lower on all 7. Factorization inside IPOPT also got faster (pde_control_250 2.36 s → 1.97 s, robot_arm 0.66 s → 0.58 s). optim20 against MA27 is unchanged (1.42–1.48x geomean over three runs; noise under load).

**Parallel solve** (bench_profile, one solve after a cold factorization; the IPOPT rows sum 5 solves). Same run, same load:

| matrix | 1 thread | 8 threads | speedup |
|---|---:|---:|---:|
| oilpan | 8.00 ms | 2.76 ms | 2.9x |
| thermal1 | 2.83 ms | 1.28 ms | 2.2x |
| bmwcra_1 | 42.4 ms | 19.1 ms | 2.2x |
| cont-201 | 3.61 ms | 1.75 ms | 2.1x |
| af_shell7 | 60.6 ms | 31.1 ms | 2.0x |
| acopf_grid_160 (5 solves) | 39.5 ms | 21.5 ms | 1.8x |
| pde_control_250 (5 solves) | 36.4 ms | 20.1 ms | 1.8x |
| copter2 | 7.35 ms | 4.64 ms | 1.6x |
| helm3d01 | 4.08 ms | 2.82 ms | 1.4x |

Before this work the solve was serial at every thread count. 16 threads are not faster than 8: the solve is memory-bound, and the top of the tree has little parallelism.

## What was wrong

**1. The matching dropped matrix entries (the robot_arm gap).** IPOPT ran 129 solves with smf against 65 with MA27 on identical iterates. smf's first residual was ~1e-4 per iteration (MA27: ~1e-8), so IPOPT refined about 3 times per step. smf's per-solve speed was already level with MA27's. The trace showed:
- IPOPT's first factorization delayed 8,456 pivots (MA27: 0). That tripped IPOPT's `ma97_switch1 = od_hd_reuse` high-delay rule, so IPOPT switched to MC64 scaling and reused it for the remaining 38 factorizations. The scaled factorizations had 11.5k delays and the 1e-4 residuals.
- The delays came from the KKT ordering falling back to AMD. On robot_arm, `compute_symmetric_matching` reported the matrix structurally singular: 79,997 of 119,998 columns matched. A pure-Python check confirmed the fixture has full structural rank.
- The bug was in `build_cost_graph` (`scaling_matching.cpp`). It merged duplicates with `slot_of[i] >= begin`. Dropping a column's explicit zeros moves `out` back, so slots left by the previous column could lie past the next column's start, and real entries were summed into dead slots as false duplicates. IPOPT's KKT has 200k explicit zeros.
- The fix is a last-seen-column marker. It comes with a 3x3 regression test (`Matching.ExplicitZerosKeepLaterEntries`).

With it, robot_arm's first factorization has 10 delays, IPOPT never switches scaling, and the solve count drops to 66. It also made the KKT analysis faster (acopf_grid_80 94 → 20 ms, acopf_grid_160 553 → 89 ms), since those matchings had been running on corrupted graphs.

**2. Amalgamation stored up to half the factor as zeros.** pde_control_120's front storage was 2.46M values against 1.27M with no amalgamation (nemin = 1, which matches MA27's 1.22M at u = 1e-8). Every solve streams the stored values. The relaxed rule merged any child into its parent while both had fewer than nemin columns, with no bound on the zeros added. On KKT trees of 1–2 column nodes with ~390-row fronts, that doubled the factor.
- The new rule keeps nemin as the gate and takes a merge only if it adds ≤ 16 stored zeros or leaves the merged node ≤ 30% zeros. A node of k columns and q rows below stores k (k + q) values, upper triangle of the pivot block included.
- Tuned against optim20 (one-shot solves, where merging tiny nodes saves per-node overhead) and the IPOPT KKT sequences. A CHOLMOD-style "always merge k ≤ 4" was the worst option on KKT trees.

| IPOPT KKT fixture (u = 1e-8) | stored L before | stored L after |
|---|---:|---:|
| pde_control_250 | 13.29M | 8.06M |
| acopf_grid_160 | 12.05M | 11.26M |
| pde_control_120 | 2.46M | 1.47M |
| acopf_grid_80 | 2.30M | 2.11M |
| obstacle_surface_200 | 2.08M | 1.91M |

Over the 7 fixtures (5 factorizations and solves each), total factor time went 2271 → 1945 ms and solve 128 → 104 ms, measured in the same sweep.

On the large SuiteSparse set, factorization is unchanged and solve is 5–17% faster.

**3. Solve kernels.**
- The small-front backward update (`b[j] -= L(:, j) · be`) was 38% of the solve on pde_control_250. Under `/fp:precise` its dot products stay scalar. It now uses AVX2: four columns per pass, sharing each load of `be` (`sub_dots`). In-process solve went 10–24% faster.
- The p = 1, 2 cases use two accumulators.
- Small forward fronts (p ≤ 16) under direct updates gather all f rows once, eliminate column by column over the contiguous vector, and scatter once. This replaces the triangular kernel, the L21 product and the update pass.
- D is now stored inline: the lean engine's unit-L fronts had 1.0 in their diagonal slots. They now keep D there, with a 2x2 block's off-diagonal in the unused strictly upper slot (c, c + 1); `FactorKeep::inline_d`. The backward pass applies D⁻¹ front by front, which removes `solve_diag`'s pass over 40-byte entries (4.8 MB per solve on robot_arm).
- The Full job does one gather (P^T S b) into a work vector and one scatter (S P y), instead of a scaling pass, two permute-and-copy passes and a scaling pass again.
- Replacing MKL's dgemv/dtrsv for large fronts with plain loops was tried and was 10–25% slower. MKL stays for p·q > 4096 and p > 16.

## Parallel solve

The tree-parallel factorization leaves the solve its layout (`FactorKeep::solve_subtrees`, `solve_top_*`):
- independent subtrees (heaviest first by stored entries), then the top of the tree;
- the top part's pivot rows and their slots;
- the top part's step tree.

The workers are a persistent `SolvePool` (std::thread, as in the factorization). They spin about 50 µs with `pause` before sleeping, so back-to-back solves do not pay thread start-up.

- **Forward:** subtrees run concurrently. Updates to top-owned rows go to per-thread buffers, which are reduced in contiguous parallel slices. The top then runs as a DAG: a front starts once its children are done, and its updates to ancestor rows are atomic adds, since only concurrently running siblings can share them.
- **Backward:** the top DAG runs from the roots (a front starts once its parent is done and writes only its own pivot rows), then the subtrees run concurrently.
- **DAG scheduling:** the thread that makes a front ready runs it next (the parent in forward, the first child in backward). Chains then stay on one core and off the shared queue. Without this, pde_control_250's 560 tiny top fronts ran 2.4x slower at 16 threads than with a serial top.
- **Gate:** used when `Control::num_threads > 1` and the factor has ≥ `solve_parallel_min_entries` (100k) values.
- **Tests:** `TreeParallel.ParallelSolveSpd` and `ParallelSolveIndefiniteWithDelays` compare against the serial solve to 1e-12, with 3 right-hand sides, the split Forward/DiagBack jobs, and pool reuse and resize.

## Tools

- `SMF_PLUGIN_STATS=1` (smf_ma97 plugin) prints factor and solve counts and times at unload. `=2` also traces every factorization: scaling, ordering, nemin, L, delays, inertia.
- `bench_profile --u <pivot_u>` sets the pivot threshold. `-v` prints eta and omega, L and times for failed runs, and the matching rank of indefinite cases. `--sample` honours `--scaling` and `--u`.

## Still open

- 16 threads give no gain over 8 in the solve. A bandwidth-aware thread count, or splitting the large top fronts across threads, would help.
- The parallel solve's floating-point sums depend on the schedule (rounding-level differences between runs, as with the tree-parallel factor's step order). `Control::deterministic` is not honoured by the solve.
- In-process, robot_arm's solve is level with MA27's. Its fronts are tiny (f ≤ 27), so the indirect gathers and scatters dominate.
- The official harness (`run.py`) and the 1/8/16-thread sweep were not rerun. All numbers here come from bench_profile and ipopt_kkt.py under ~15% background load.
