# acopf at 8 threads: where the time goes (2026-11-09)

Input to the re-plan after Stage 2's stop rule failed. Code: the Stage 2 tip (`perf/8t-p6-lookahead` 620264e, defaults:
Part 7 on, Parts 5 and 6 off) plus a probe (`probe/s2-ftrace` b74df01, not for merge). Machine: i9-12900K, P-cores,
below-normal priority, one job at a time.

## Same-day projection (8T, PR #64 run lengths)

`e8t_s2_sameday`, smf and MKL PARDISO measured the same day, 3 trials.

| case | N | smf refactor ms | smf run ms | PARDISO run ms | ratio |
|---|---:|---:|---:|---:|---:|
| pde_control_250 | 12 | 25.28 | 375.8 | 435.5 | 0.863 |
| pde_control_120 | 13 | 4.88 | 78.1 | 86.2 | 0.906 |
| portfolio_20k | 17 | 4.70 | 96.2 | 143.6 | 0.670 |
| acopf_grid_160 | 24 | 34.52 | 894.2 | 794.9 | **1.125** |
| acopf_grid_80 | 25 | 6.73 | 185.3 | 163.5 | **1.133** |
| pglib case1354 | 46 | 2.33 | 126.0 | 324.5 | 0.388 |
| pglib case2000 | 75 | 4.27 | 352.0 | 644.7 | 0.546 |
| obstacle_surface_200 | 136 | 4.70 | 650.4 | 691.5 | 0.940 |

To win, acopf_grid_160 needs a refactorization of about 29.6 ms (-14%), acopf_grid_80 about 5.8 ms (-14%). smf's
analysis is already 70 ms (acopf160) and 16 ms (acopf80) faster than PARDISO's; the loss is all in the 23-24
refactorizations.

## smf and PARDISO on the same matrices

| | acopf_grid_160 | acopf_grid_80 |
|---|---:|---:|
| smf refactor 1T / 8T (ms) | 141 / 34.5 (4.1x) | 19.8 / 6.7 (2.9x) |
| PARDISO refactor 1T / 8T (ms) | 159 / 27.4 (5.8x) | 23.0 / 5.3 (4.4x) |
| smf factor entries (f x p per front) | 12.33 M | 2.27 M |
| PARDISO nnz(L) (iparm 18), 8T | 10.80 M | 2.09 M |
| smf flops / PARDISO Mflops (iparm 19) | 4.35 / 3.85 G | 0.39 / 0.41 G |
| smf delayed pivots per factorization | 3994 | 969 |

smf is faster at 1 thread and loses on scaling. smf's entry count includes the zero upper triangle of each pivot block
(about sum p^2/2), so the factors are within a few percent of each other in size. PARDISO perturbed no pivot on
either case.

## Where smf's 8T refactorization goes

Phase marks, no probe (`e8t_s2_stop_phase`), ms per refactorization:

| case | map+scale+setup | subtree phase | top phase | post | total |
|---|---:|---:|---:|---:|---:|
| acopf_grid_160 | 2.0 | 11.4 | 21.3 | 1.2 | 35.5 |
| acopf_grid_80 | 0.8 | 1.8 | 4.7 | 0.3 | 7.7 |

The subtree phase scales well (acopf160: 80 ms of 1T work in 11.4 ms, efficiency 0.87-0.92). The top phase does not:
acopf160's top fronts take 59.8 ms at 1T and 21.3 ms at 8T (2.8x); acopf80's take 7.0 and 4.1-4.7 ms (1.6x).

## The top phase is one chain

Per-front probe (`e8t_s2_ftrace2`, SMF_BENCH_PHASE_TRACE=3, median of 10 refactorizations; the probe adds ~1.5 ms).

acopf_grid_160 has 21 top fronts: two long chains of 600-1200-row fronts (124-334 pivots each, root 851) plus two
short ones, merging near the root. The top span (21.45 ms) equals the critical path through the front walls
(21.34 ms); some front is running at every instant (idle 0.1 ms). The critical path is one chain:

| front (rows:pivots) | threads | wall ms | assembly | panel | deferred | Schur | other | same front 1T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 640:124 | 2 | 1.36 | 0.28 | 0.15 | 0.19 | 0.48 | 0.26 | 1.15 |
| 770:154 | 2 | 1.68 | 0.38 | 0.23 | 0.18 | 0.70 | 0.18 | 1.84 |
| 956:224 | 2-6 | 2.64 | 0.69 | 0.44 | 0.41 | 0.88 | 0.24 | 3.50 |
| 1097:217 | 6 | 2.65 | 0.75 | 0.46 | 0.30 | 0.91 | 0.23 | 4.56 |
| 1214:334 | 6 | 3.59 | 0.86 | 0.74 | 0.61 | 1.20 | 0.18 | 7.58 |
| 1176:284 | 6-8 | 3.33 | 0.77 | 0.64 | 0.50 | 1.11 | 0.32 | 6.33 |
| 1160:851 | 8 | 5.15 | 1.06 | 1.31 | 1.83 | 0.63 | 0.32 | 10.84 |
| root tail (4) | 8 | 0.35 | | | | | | 0.30 |
| **path** | | **20.75** | **4.83** | **4.06** | **4.12** | **5.95** | **1.74** | **36.1** |

- Half the path is serial per front: assembly, the TPP panel (pivot search and in-panel updates over the full column
  height) and "other" (the contribution block copied out, D read out, rows). These run on one thread whatever the
  share, and run no faster than at 1T (assembly is slower: 4.8 vs 3.9 ms).
- The BLAS half (deferred + Schur, 10.1 ms) gets 2.7x on 2-8 threads; mid-size GEMMs on 6 threads reach ~3.5x.
- The chain's first three fronts run on 2 threads because four chains start together.
- So the path runs 1.7x faster than at 1T. acopf_grid_80 is worse: its critical path (8 fronts, 350-530 rows) takes
  4.07 ms at 8T against 4.32 ms at 1T (1.06x); half of it is serial, and its BLAS calls are too small for 8 threads.

Serial overhead around the tree is small: map+scale 1.7 ms and post 1.2 ms on acopf160.

## Levers tested

| lever | how | acopf160 | acopf80 | verdict |
|---|---|---:|---:|---|
| Merge chain fronts (bigger, fewer fronts) | cost-model merge price for blocks >= 256 rows: 2 (default), 16, 64, 256 (`e8t_amalg_ab`, 3 rounds) | 1.00 / 0.98 / 1.02 / 1.41 | 1.00 / 1.01 / 1.18 / 1.49 | rejected: +10-76% flops cost more than the assembly saved |
| Start top fronts before the subtree phase ends | from the trace: when each top front's subtree children finish | <= 1 ms | <= 0.3 ms | small: the critical chain's inputs finish last |
| Blocked TPP panel (bp128e, 2026-11-04 study) | microbench on saved acopf fronts, board bound | <= 7.9% | <= 2.7% | not enough alone |
| Parallel assembly (Part 5) on the critical path | per-front trace with it on from 256 / 512 rows (`e8t_s2_ftrace_pasm256/512` vs `_base3`) | path -0.3 ms | path +0.1 ms | the saving moves into BLAS (below) |
| METIS nested dissection on the matching-compressed graph | probe `SMF_PROBE_COMPRESSED=metis` (`e8t_ord_cmetis` vs `e8t_ord_default2`) | refactor -25% | refactor -18% | **the lever** (below) |

Other orderings (`e8t_ord_*`): plain AMD doubles acopf's flops and delays 40x; plain METIS cuts acopf160's flops 22%
but delays 12x more and analyses in 366 ms; the default is already the matching ordering (compressed AMD).

### Parallel assembly's saving reappears as slower BLAS

On acopf160's critical path, column-block assembly cuts assembly from 4.8 to 1.8 ms, but the same fronts' deferred and
Schur updates get 2.4 ms slower (the 1214-row front's Schur update 1.16 -> 1.73 ms on the same 6 threads). The panel
does not slow down. Idle smf workers spin only 50 us before sleeping, so they are not the cause. Suspect: MKL's
OpenMP teams (one per calling thread, spinning 1 ms after each call) and smf's helper threads competing for the
8 cores. Not root-caused; it also caps how far the BLAS parts scale (2.7x on the path).

### The ordering

smf's analysis already has METIS on the matching-compressed graph (`CompressedOrder::Metis`, MA97 ordering 8 style);
its auto rule tries it only when predicted flops x `expected_refactors` reach 4.5e5 per row, and the bench's ipopt
track runs with `expected_refactors = 1` (the IPOPT plugin passes 10). Forced on, 8T, 3 trials, same session:

| case | refactor ms | flops GF | entries M | delays | analysis ms | first factor ms |
|---|---|---|---|---|---|---|
| acopf_grid_160 | 34.70 -> **26.05** | 4.35 -> 2.90 | 12.33 -> 10.77 | 3994 -> 4011 | 58.9 -> 109.9 | 43.4 -> 33.0 |
| acopf_grid_80 | 6.89 -> **5.68** | 0.39 -> 0.37 | 2.27 -> 2.32 | 969 -> 986 | 15.4 -> 27.4 | 8.2 -> 6.5 |
| pde_control_250 | 25.17 -> 20.83 | 1.81 -> 1.48 | 8.63 -> 7.67 | 0 | 57.3 -> 159.9 | 37.0 -> 31.2 |
| pde_control_120 | 4.95 -> 4.82 | same | same | 0 | 13.3 -> 34.5 | 6.4 -> 5.9 |
| obstacle_surface_200 | 4.69 -> 4.78 | same | same | 0 | 8.8 -> 8.7 | 5.5 -> 6.2 |

At 1T acopf160 goes 140 -> 117 ms too. pde_control_250's 12-step run loses with it (+100 ms of analysis against
4.3 ms per refactor), so the choice has to weigh the analysis cost against the sequence length.

The existing auto rule with the sequence length as `expected_refactors` (probe `SMF_PROBE_EXPECTED_REFACTORS`,
`e8t_er_*`, all 8 cases, 8T, 3 trials, projected against `e8t_s2_sameday`'s PARDISO):

| case | er = 1 (bench today) | er = 10 (IPOPT plugin) | er = 24 | METIS forced |
|---|---:|---:|---:|---:|
| pde_control_250 | 0.856 | 0.840 | 0.854 | 0.993 |
| pde_control_120 | 0.938 | 0.891 | 0.945 | 1.145 |
| portfolio_20k | 0.745 | 0.710 | 0.711 | 0.712 |
| acopf_grid_160 | 1.132 | 1.117 | **0.926** | 0.925 |
| acopf_grid_80 | 1.128 | 1.093 | 1.185 | 1.124 |
| pglib case1354 | 0.403 | 0.344 | 0.393 | 0.350 |
| pglib case2000 | 0.538 | 0.537 | 0.531 | 0.552 |
| obstacle_surface_200 | 0.902 | 0.935 | 1.085 | 0.950 |

- At 24 the rule switches acopf160 only (flops 4.35 -> 2.90 GF, analysis 60 -> 106 ms); every other case keeps its
  ordering. At 10 nothing switches.
- acopf80's METIS ordering saves only 4% of flops; its refactor gain (5.7-6.2 vs 6.7-6.9 ms) varies between
  sessions and does not cover the +12 ms of analysis at N = 25.
- Every column other than "METIS forced" has the same structure on obstacle, pde_control and PGLib, so their spread
  (obstacle 0.90-1.09) is run-to-run noise.

## Measurement noise

- The merge-price variant 16 leaves acopf160's structure unchanged, yet moved obstacle and pde120 by 8-12%.
  Across the `e8t_er_*` runs, obstacle's ratio ranged 0.90-1.09 with identical structure.
- Within one run, smf's trials spread about +-7% on acopf_grid_80 (6.24-7.14 ms) and up to 20% on obstacle (4.55-5.68).
  PARDISO is steadier on acopf (27.2-29.0 ms on acopf160).
- One logical CPU per P-core (affinity 0x5555, `e8t_aff_*`) does not speed smf up (acopf160 33.8-33.9 ms against
  33.1-33.8). It makes PARDISO stall at random: 260-525 ms factorizations on acopf80, 750 ms on obstacle. So
  hyperthread placement is not smf's problem, and the gate keeps affinity FFFF.
- The gate's "all 8 below 1.00" therefore needs more trials, with smf and PARDISO interleaved, before a 0.95-1.05
  case is called. Today that is obstacle (0.89-0.98 in the same-session runs).

## Conclusions

1. **acopf_grid_160 is an ordering problem, and the fix exists.**
   - METIS nested dissection on the matching-compressed graph cuts its flops by a third and its 8T refactorization
     by 25%. The analysis grows by about 50 ms, still well under PARDISO's 128 ms.
   - Projected ratio: 1.12 -> 0.93.
   - The existing auto rule picks this ordering once `expected_refactors` reflects the sequence (24), and switches no
     other case. The IPOPT plugin passes 10 and the bench passes 1, so neither picks it today.
2. **acopf_grid_80 is the hard one.**
   - Its 8T critical path is a chain of 350-530-row fronts that runs at 1T speed (4.07 vs 4.32 ms).
   - About half of that path is serial per-front work: assembly, the TPP panel and the CB copy-out. The BLAS calls
     are too small to scale on 8 threads.
   - The METIS ordering helps it little (-4% flops), and bigger fronts by merging hurt.
3. **Stage 2's parts did not move the span.** The chains' serial half doesn't shrink with more threads (Part 7),
   and parallel assembly (Part 5) shows a real but unexplained interaction: assembly time saved comes back as slower
   MKL calls on the same fronts. That interaction, not the assembly kernel, is what stands between Part 5 and a
   -3 ms critical path on acopf160.
4. **PARDISO's edge is parallel efficiency on the same work.** Its factor is the same size as smf's, and smf is
   faster at 1T. At 8T it gets 4.4-5.8x where smf's top chains get 1.1-1.7x.

## Options

- **A. Ordering policy for long sequences (small, high confidence).**
  - Let the auto rule see a realistic refactorization count: the plugin's estimate (10 today) raised for IPM use,
    and the bench's ipopt track passing the same value.
  - Re-check the board for short-sequence cells that would switch and pay the METIS analysis.
  - Expected: acopf160 to about 0.93; no other probe case moves.
- **B. Root-cause the BLAS slowdown next to parallel work.**
  - A controlled microbenchmark: MKL calls on a 500-1200-row front with and without smf helpers active just before.
  - If MKL's OpenMP teams and smf's workers are fighting, fix the thread ownership (for example, one pool driving
    sequential BLAS on the top chain). Then Part 5, a parallel CB copy-out and the blocked panel can cut the
    chain's serial half.
  - This is the only path found to acopf80, at roughly -10 to -15% if it works; uncertain.
- **C. Accept acopf80 (about 1.1) as the remaining loss** after A, and move to "best optimizer".

## Follow-up: steps A and B (Daniel chose "A, then B", 2026-11-09)

### A: tell the analysis how long IPOPT runs are (PR #72)

- The IPOPT plugin's refactorization estimate goes 10 -> 25 (`kIpoptExpectedRefactors`).
  - Mittelmann AMPL-NLP iterations: lower quartile 21, median 55.
  - PR #64 runs: median 24.5 factorizations.
- The bench's ipopt track passes it for replayed sequences only. A cold cell measures one factorization and keeps 1.
- acopf_grid_160 switches from 12.
- Same-day (`e8t_a_sameday`): 8T acopf_grid_160 1.125 -> 0.945; all 8 cases win at 1T; acopf_grid_80 1.139 unchanged.
- Board: with 25 on cold ipopt cells, 8 of them switch ordering.
  - Each factorization gets 1.3-3x faster, but the analysis grows.
  - That loses on one factorization (1.16-1.68x wall) and is recovered within 2-8 factorizations of a real run.
  - So the bench keeps 1 for those cells, while the plugin uses 25 for everything.

### B: why parallel assembly slowed the next MKL call, and the fix

Microbenchmark (`bench_blas_interference`, probe branch `probe/blas-interference`): the 1214:334 front at 6 threads.

| assembly before the Schur update | assembly ms | Schur update ms |
|---|---:|---:|
| none (team hot) | - | 0.70 |
| serial | 0.55 | 0.72 |
| smf pool threads (Part 5) | 0.17 | 0.76-1.27 |
| smf pool, then 2 ms pause or 64 MB flush | 0.17 | 0.79-0.83 (as serial after the same pause) |
| OpenMP region on the calling thread's team | **0.12** | **0.71** |

- MKL blocktime 0 removes the penalty but loses the hot team (every call 0.80-0.86 ms).
- So the cause is smf's pool threads running next to MKL's OpenMP team, which spins 1 ms after each call. It is not
  cache placement.
- The deps' `libiomp5md.dll` forwards to `libomp.dll`, so smf's own OpenMP regions already run on MKL's runtime.

Fix (branch `perf/8t-omp-assembly`, on #72):
- Part 5's column-block assembly and the contribution-block copy-out run as OpenMP loops on the calling thread's
  team, bit for bit the serial result.
- Parallel assembly is on by default from 256 rows.
- 8T sweep, 5 rounds, refactorization geomean over the 8 cases: from 256 rows 0.948, from 512 0.971, from 1024 0.978.
  Gains: acopf160 -5%, acopf80 -6%, pde_control -7 to -10%, portfolio -9%.
- Board, 1 round: 8T geomean 0.990, 16T 0.993; flagged cells rerun over 3 rounds (below).
- Flagged board cells:
  - A 3-round rerun of 24 cells left 7 at 1.05 or worse.
  - A 5-round rerun of those 7 with thresholds 256/512/1024 puts the default (256) at 0.94-1.035 everywhere, at 8T
    and 16T.
  - The 1024 variant leaves those cells untouched and still moved them by +-20%: one-round board flags on cells
    under ~50 ms are noise.
- PR #73. Same-day with A + B (`e8t_b3_sameday`, 8T):

| case | ratio |
|---|---:|
| acopf_grid_160 | **0.906** |
| pde_control_250 | 0.813 |
| pde_control_120 | 0.889 |
| portfolio | 0.892 |
| case1354 | 0.389 |
| case2000 | 0.592 |
| acopf_grid_80 | **1.114** |
| obstacle | 1.015 (0.89-0.98 on other days; its refactorization varies 4.5-5.3 ms between runs) |

### Where that leaves the bar ("win all 8 at 8T")

- 6 cases win clearly.
- **obstacle** sits at parity and flips with run-to-run noise.
- **acopf_grid_80** is about 10% short. It needs its refactorization at about 5.9 ms; it is at 6.6-6.8 ms.
- Its remaining levers are each small:
  - **Value preparation on the OpenMP team** (probed): map 0.49 -> 0.27 ms, about 3%.
  - **Critical-chain start shares:** its first chain fronts run on 1 thread, while 12 fronts start together.
  - **Overlapping the top phase with the subtree tail:** at most 0.3 ms.
  - **The blocked panel:** at most about 3%.

### Gate run and the last acopf_grid_80 levers (2026-11-09, late)

Gate projection on #73 (`e8t_gate3`: smf and PARDISO the same day, 7 trials):

| | 1T | 8T |
|---|---|---|
| wins | all 8 (0.40-0.89) | 7 of 8 |

| case (8T) | ratio |
|---|---:|
| case1354 | 0.400 |
| case2000 | 0.611 |
| portfolio | 0.765 |
| pde250 | 0.879 |
| pde120 | 0.885 |
| acopf160 | 0.903 |
| obstacle | 0.960 |
| **acopf80** | **1.050** |

acopf80 needs its refactorization at about 5.95 ms; it is at 6.32 ms.

Levers tried for it after B (8T, 5 rounds, 8 cases):

| lever | result | verdict |
|---|---|---|
| value preparation on the OpenMP team | geomean 0.998; acopf80 1.074, obstacle 1.154 (the team then spins next to the subtree phase's pool) | dropped |
| MKL blocktime 0 / 3 ms (default 1) | geomean 1.078 / 1.014 | default stays |
| critical-path front starts with >= 2/3/4 threads | geomean 1.024 / 1.026 / 1.044 | dropped |
| compressed METIS forced for acopf80 | 1.050 -> 1.030 (refactor -9%, analysis +12 ms); hurts the pde cases | only as a narrow rule; not enough alone |

The one identified lever left is starting top fronts as soon as their subtrees finish, instead of after the whole
subtree phase: at most about 0.3 ms on acopf80 (4%). It is an engine change, not a tuning.
