# Triage: smf vs MA27 per-iteration factorization on PGLib AC-OPF (IPOPT)

Date: 2026-10-27 (Task 17). Branch `feat/solver-performance`, base cb71627, fix 642873a.
All runs: 1 thread (`OMP/MKL/OPENBLAS_NUM_THREADS=1`), below-normal priority. Tuning split only
(PGLib case118, case500_goc, case2869_pegase, case4661_sdet; COPS tuning models for the IPOPT
check). Held-out cases (case1354/2000/9241) were only replayed for the R7 accuracy gate.

## TL;DR

- **The leading hypothesis is only half right.** Small fronts do carry a per-front fixed cost
  (about 500 cycles for an f <= 4 front, about 1,000 for f <= 8). But smf's numeric
  refactorization already beat MA27 on every PGLib tuning capture: 0.64-0.83x MA27's factor
  time on the replay harness. The 1.7-2.2x gap was elsewhere in the IPOPT path.
- **Where the gap was.** IPOPT's MA97 interface asks for MC64 scaling (`scaling = 1`) after
  high delays and on every IncreaseQuality call. The plugin computed smf's matching scaling
  for every request. On case4661 that was 4 requests at ~300 ms each, against a ~24 ms
  refactorization: 37% of IPOPT's `LinearSystemFactorization`. The deferred analysis (first
  factorization) added another ~0.5 s, ~85% of it in the same matching, which the auto ordering
  uses to choose between the matching ordering and AMD.
- **It was also an accuracy problem.** A matching scale reused with the AMD ordering the
  analysis keeps (the case9241 mechanism) failed 36/52 (case2869) and 46/60 (case4661) stored
  steps through the plugin, 29 of them with the wrong inertia. IPOPT still converged in the same
  iterations, but on inaccurate solves.
- **Fix (642873a).** (1) Plugin: once the analysis has computed the matching ordering and
  rejected it, an MC64 request is answered with the unit scale. This is the `ScalingMethod::Auto`
  rule. Patterns the matching was never tried on (n < 10,000) keep the matching scaling; COPS
  elec/chain depend on it. (2) Kernel: at u < 1e-6 a weak 1x1 pivot (|d| < 1e-6 of its column)
  gives way to the 2x2 with its partner when the 2x2's growth bound is smaller. This fixes
  case500 step 48, a pre-existing failure. (3) Matching search state packed, about 10% faster,
  bitwise identical.
- **Result.** IPOPT `LinearSystemFactorization` (s), smf before -> after, against MA27 in the
  same session: case2869 1.264 -> 0.753 (0.593), case4661 3.215 -> 1.822 (1.742), case500
  0.159 -> 0.119 (0.094). Iterations are unchanged, except cops_chain_800: 424 -> 167 (MA27
  1475). The full table is in the Task 17 report. What remains is the one-time analysis (the
  matching), not the per-iteration refactorization, which is 0.65-0.85x MA27's.

## 1. Starting gap (replay harness, ipopt track, all stored steps, median of 3)

`bench_replay --mode warm_sequence --track ipopt --mtype indefinite --seq-steps all`, total
factor ms over the sequence / median warm refactor ms:

| case | steps | smf | MA27 | MA57 | smf/MA27 |
|---|---|---|---|---|---|
| case118 | 27 | 9.3 / 0.33 | 11.4 / 0.41 | 86.0 / 3.28 | 0.82 |
| case500_goc | 53 | 79.9 / 1.47 (1 failing step) | 96.8 / 1.81 | 798 / 15.1 | 0.83 |
| case2869 | 52 | 488 / 9.3 | 638 / 12.1 | 4478 / 84.3 (28 failing) | 0.76 |
| case4661 | 60 | 1283 / 21.1 | 1990 / 33.2 | 11888 / 198 (57 failing) | 0.64 |

The harness's MA57 adapter runs MC64 scaling in every factorization (MA57's default
ICNTL(15) = 1), which is why it is 8-9x MA27 here. Inside IPOPT, MA57 is close to MA27
(below), because IPOPT's `ma57_automatic_scaling` defaults to no.

IPOPT end to end (`tools/bench/ipopt_kkt.py e2e`, `LinearSystemFactorization` s, iterations):

| model | smf | MA27 | MA57 |
|---|---|---|---|
| case118 | 0.017 (26) | 0.008 (26) | 0.015 (26) |
| case500_goc | 0.159 (51) | 0.088 (51) | 0.116 (51) |
| case2869 | 1.264 (50) | 0.629 (50) | 0.681 (50) |
| case4661 | 3.215 (57) | 1.803 (57) | 1.589 (60) |

The plugin trace (`SMF_PLUGIN_STATS=2`) splits smf's case4661 time as follows: 55 reuse
refactorizations take 1.395 s (25 ms each), 4 `scaling=1` requests take 1.194 s (~300 ms each),
and the rest of the 3.215 s, about 0.6 s, is the deferred analysis plus plugin overhead.

## 2. Profile: numeric refactorization (Step 1)

`build-prof/benchmarks/Release/bench_profile.exe --manifest benchmarks/manifests/ipopt_hard.json
--case <c> --sample factor --seconds 8 --lines`. bench_profile's sample mode was extended
(642873a): a case with a recorded value sequence cycles through all of its stored steps, and an
ipopt-track case factors at u = 1e-8. The adapter is the plain smf adapter (Auto scaling),
which matches IPOPT's reuse refactorizations after the fix. Each line is a percentage of the
samples. `mkl_blas_def_*` names are MKL code under a wrong symbol (no MKL PDBs).

case4661_sdet (60 steps):

| % | line | what |
|---|---|---|
| 14.50 | mkl_blas_def_ztbsv | MKL BLAS of the 84 fronts with f > 128 (dense work) |
| 11.57 | memmove | contribution-block stack copies of the large fronts, row-id copies |
| 11.12 | multifrontal_serial.cpp:437 | child contribution block scatter-add into the parent |
| 2.30 | multifrontal_serial.cpp:585 | D entry push (1x1) |
| 1.93 | multifrontal_serial.cpp:413 | original entries scattered into the front |
| 1.68 | ldlt_kernel.cpp:423 | small-front 1x1 elimination (apply_pivot<16>) |
| 1.58 | ldlt_kernel.cpp:519 | ldlt_factor_small_front dispatch |
| 1.58 | ldlt_kernel.cpp:243 | generic 1x1 rank-1 update (fronts > 32) |
| 1.58 | multifrontal_serial.cpp:578 | D entry push (2x2) |
| 1.30 | _NLG_Return2 | vcruntime/ntdll without symbols |

case2869_pegase (52 steps): memmove 5.57, :585 3.53, :413 3.39, :437 3.39, small-front
dispatch 2.86 + 2.27, :578 2.30, apply_pivot<16> 2.19, map_values_to_permuted 2.08, :458
(kernel call) 2.03. case500_goc: memmove 6.18, :585 3.61, :413 3.51, :437 3.38, then the same
lines at 2.7-1.4. Function level (case2869): process_front self 38.8%, select_pivot 11.2%,
ldlt_factor_small_front 7.9%, memmove 5.6%, two_by_two_ok 3.3%.

The refactorization profile is flat: no line reaches 10% on case500 or case2869. On case4661
the top entries are dense BLAS and contribution-block data movement in the few large fronts,
which MA27 does as well.

## 3. Per-front overhead (fronts with f <= 8)

Temporary rdtsc instrumentation around `process_front` in the serial loop (not committed),
cycling all stored steps, bucketed by symbolic front size. Per refactorization:

| case | bucket | fronts | cycles / front | share of multifrontal_factor_serial |
|---|---|---|---|---|
| case500 | f <= 4 | 1,402 | 459 | 8.1% |
| | 4 < f <= 8 | 2,101 | 854 | 22.5% |
| | 8 < f <= 16 | 1,691 | 1,984 | 42.0% |
| | 16 < f <= 32 | 151 | 9,515 | 18.0% |
| case2869 | f <= 4 | 8,722 | 510 | 8.5% |
| | 4 < f <= 8 | 13,488 | 932 | 23.9% |
| | 8 < f <= 16 | 9,874 | 2,331 | 43.7% |
| case4661 | f <= 4 | 11,229 | 567 | 4.9% |
| | 4 < f <= 8 | 14,963 | 1,084 | 12.4% |
| | 8 < f <= 16 | 16,124 | 2,543 | 31.4% |
| | f > 128 | 84 | ~0.4-1.2 M | 37.6% |

Fronts with f <= 8 are 60-67% of all fronts and take 17-32% of the numeric factorization. A
4x4 front does ~20 flops, so its ~500 cycles are almost all fixed cost. That cost is spread
over ~30 steps: row setup and pos[] marking, grow() checks, the delay-budget, fallback and
APTP selection, zeroing the buffer, the A and child scatters, the kernel call, the L copy-out,
the D and SolveStep pushes, the row-id copies and the pos[] reset. Three candidate fixes made
no measurable difference (Mcycles per refactorization unchanged within noise), and all were
reverted:

- keeping `FactorKeep::pivot_types`' per-front vectors across refactorizations (no per-front
  malloc/free);
- inline AVX copies instead of `std::copy`/memmove for the row ids and tags;
- nemin 4/16/32 instead of IPOPT's 8 (case500/2869 unchanged; case4661 -6% at 16, which
  changes the tree for every caller).

The smf refactorization is already 0.64-0.83x MA27's, so the per-front cost does not explain
the IPOPT gap.

## 4. Profile: the IPOPT path's real hotspot, the matching

`--sample factor --scaling 2` makes every factorization compute the matching scaling, as the
plugin did for every IPOPT MC64 request. On case4661 one factorization takes ~500 ms, and
93% of it is `compute_symmetric_matching`:

| % | line |
|---|---|
| 20.06 | scaling_matching.cpp:283 (Dijkstra relax: row state checks) |
| 16.32 | scaling_matching.cpp:284 (reduced cost) |
| 13.39 | scaling_matching.cpp:281 (edge loop) |
| 6.96 | scaling_matching.cpp:302 (heap push) |
| 4.76 | std::_Push_heap_by_index |
| 4.48 | std::_Pop_heap_hole_unchecked |
| 2.87 | scaling_matching.cpp:286 |
| 1.51 | scaling_matching.cpp:287 |
| 1.34 | scaling_matching.cpp:301 |
| 1.09 | scaling_matching.cpp:285 |

`--sample analyse --ordering 1` (the plugin's AutoParallel): the matching takes 74-85% of the
analysis on case500/2869/4661 (relax lambda 51-63%, search loop 9-11%, heap 11-12%, AMD 4.5-7.8%).

Why it is slow (instrumented counts, case4661): n = 159,871, 554k cost-graph entries. The
greedy plus length-2 initialization matches 138.8k columns, which leaves 21k Dijkstra searches,
29M edge relaxations and 8.3M finalized rows. The work is concentrated: in one step, the 10
largest searches finalize 34% of all rows, and the top 100 finalize 62%. Each of the largest
finalizes 110-120k rows (path length ~5-7 in log-magnitude units). These are long augmenting
paths across the grid network, and they are inherent to successive shortest paths: the column
order (forward, reverse, degree-sorted, random) changed the time by < 5%. A warm start from the
previous step's duals cut consecutive-step time ~3x, but only ~30% for IPOPT's actual rescale
spacing (steps 0, 1, 3, 43, 53). It is also history-dependent, so it was not pursued.

## 5. Accuracy finding

Plugin replay with IPOPT's scaling sequence (unscaled first factorization, MC64 at the second,
reuse after that; test `IpoptPluginRuntime.PglibReplayAccurate`, 3 refinement steps, gate
eta <= 1e-10 and omega <= 1e-8):

| | case500 | case2869 | case4661 |
|---|---|---|---|
| before (reused matching scale, AMD order) | 0/53 | 36/52 failing, 6 wrong inertia | 46/60 failing, 23 wrong inertia |
| plain per-factor matching (bench_replay --scaling matching) | 0/53 | 35/52 | 49/60 |
| no scaling, before the kernel fix (bench_replay Auto) | 1/53 (step 48) | 0/52 | 0/60 |
| after 642873a | 0/53 | 0/52 | 0/60 |

case500 step 48 without scaling: omega 1.0 unrefined, 2.4e-4 after 3 refinement steps, 1e-11
after 10. Raising u to 1e-7 fixes it, and so does a stable 2x2 in place of a 1x1 accepted at
|d| ~ 5e-8 of its column max. MA27 pivots that column in a 2x2 (unrefined omega 1.1e-4). The
kernel rule (weak 1x1 below 1e-6 of the column, 2x2 taken when its growth bound is smaller)
gives 3.6e-5 unrefined and 1.2e-13 after 3 steps.

## 6. Next levers (not done)

1. The analysis' matching (~0.25 s on case2869, ~0.5 s on case4661, 20-30% of IPOPT's
   factorization time after the fix). It only decides "matching ordering or AMD" on PGLib,
   which always picks AMD. Options: a cheaper exact algorithm (JV-style augmenting row
   reduction before the shortest paths, or multi-source phases), or a cheap pre-test that skips
   it. Both can change the auto ordering's decision on other KKT matrices, so they need the
   full corpus (optim20, fronttier, R7 replays) to validate.
2. Per-front fixed cost of f <= 8 fronts (17-32% of the refactorization): a batched path for
   chains of tiny fronts (assemble and factor several leaves in one buffer pass, one D/step
   push per batch), rather than trimming single operations.
3. case118-size patterns (n < 10,000) still compute the matching for each MC64 request
   (3.5 ms against a 0.45 ms refactorization). They keep it on purpose (COPS elec 180 vs 320
   iterations without it).
