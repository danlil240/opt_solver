# Analyse time on hard indefinite and KKT matrices: a cheaper matching trial

**Date:** 2026-10-30
**Branch:** `feat/cheaper-matching-trial` (3 commits on main `4d8421e`)
**Purpose:** after the SSIDS threading fix, the remaining 1-thread losses on hard_indef and KKT cases are mostly analysis time against MA97 and MA57 (29 of 46 losing case-tracks, 60% of the summed log-loss; see `2026-10-29-path-to-number-1.md`). The task was to cut smf's analyse time on those cases with accuracy and inertia unchanged.

## Result

Every change is **output-identical**:
- the same ordering, matching scale, rejected flag and trial verdict on all 50 n >= 10000 indefinite corpus matrices (hash survey, `bench_matching_trial`);
- the same factor entries, flops, delays and accuracy on every front-tier case-track in the A/B sweeps.

Analysis gets cheaper; nothing else changes.

**Front-tier score, 1 thread** (smf vs best accurate competitor, shifted geomean). Both columns use today's interleaved smf runs against the same corrected competitor records (`fronttier_20261029-ssidsfix`). Each cell is tuning | held-out.

| class / track | main | this branch |
|---|---|---|
| hard_indef recommended | 1.043 \| 1.078 | **0.978 \| 1.019** |
| hard_indef ipopt | 1.024 \| 1.134 | **0.957 \| 1.063** |
| large_kkt recommended | 0.695 \| 0.715 | 0.687 \| 0.693 |
| large_kkt ipopt | 0.864 \| 1.030 | 0.857 \| 1.026 |
| small_kkt recommended | 0.913 \| 0.993 | 0.912 \| 0.999 |
| small_kkt ipopt | 0.526 \| 0.775 | 0.527 \| 0.769 |

hard_indef at 1 thread is now below 1 on the tuning split. The tuning records are the final sweep; cvxqp3 comes from a separate 3-round A/B of the last commit, the only front-tier case it changes. The held-out records are from the binary before that commit, which changes no held-out case: on the corpus it only affects cvxqp3. The "main" column is lower than the 1.11 | 1.13 quoted on 2026-10-29: smf ran on a quieter machine today, and the competitors' records are the same.

**smf against smf** (interleaved A/B, 3 rounds, fresh process per run, 1 thread, B/A shifted geomean):

| split | class | case-tracks | wall | analyse |
|---|---|---:|---:|---:|
| tuning | hard_indef | 42 | **0.937** | **0.897** |
| tuning | large_kkt | 21 | 0.991 | 0.990 |
| tuning | small_kkt | 26 | 1.000 | 1.002 |
| tuning | all | 89 | 0.968 | 0.948 |
| held-out | hard_indef | 20 | **0.941** | **0.896** |
| held-out | large_kkt | 8 | 0.983 | 0.984 |
| held-out | small_kkt | 9 | 0.999 | 0.990 |
| held-out | all | 37 | 0.964 | 0.937 |

The cases that move (median ms, 1 thread, main → branch):

| case | track | split | analyse | wall |
|---|---|---|---|---|
| TSOPF_FS_b300_c1 | ipopt / rec | tuning | 888 → 517 (0.58) | 1176 → 808 (0.69) |
| TSOPF_FS_b300_c2 | ipopt | tuning | 1528 → 723 (0.47) | 2098 → 1292 (0.62) |
| TSOPF_FS_b300_c2 | rec | tuning | 1509 → 719 (0.48) | 2413 → 1626 (0.67) |
| TSOPF_FS_b300_c3 | ipopt | held-out | 2430 → 1154 (0.48) | 3303 → 2016 (0.61) |
| TSOPF_FS_b300_c3 | rec | held-out | 2400 → 1144 (0.48) | 3687 → 2430 (0.66) |
| ncvxqp5 | ipopt / rec | tuning | 828 → 549 (0.66) | 1483 → 1211 (0.82) |
| ncvxqp7 | ipopt / rec | held-out | 2067 → 1567 (0.76) | 5454 → 4945 (0.91) |
| ncvxqp3 | ipopt / rec | tuning | 2411 → 2017 (0.84) | 4952 → 4590 (0.93) |
| cvxqp3 | ipopt / rec | tuning | 263 → 194 (0.74) | 462 → 394 (0.85) |

Against the best competitor (tuning, 1 thread, wall ratio with the 10 ms shift):

| case | track | best | main | branch |
|---|---|---|---:|---:|
| TSOPF_FS_b300_c2 | ipopt | SSIDS | 1.35 | **0.83** |
| TSOPF_FS_b300_c2 | rec | SSIDS | 1.49 | **1.01** |
| TSOPF_FS_b300_c1 | ipopt | SSIDS | 1.42 | **0.98** |
| TSOPF_FS_b300_c1 | rec | SSIDS | 1.38 | **0.95** |
| ncvxqp5 | ipopt | MA97 | 1.51 | 1.23 |
| ncvxqp5 | rec | PARDISO | 1.41 | 1.14 |
| cvxqp3 | ipopt | MA97 | 1.41 | 1.22 |
| cvxqp3 | rec | MA57 | 1.07 | **0.92** |
| ncvxqp3 | ipopt | MA97 | 2.28 | 2.11 |
| ncvxqp3 | rec | MA57 | 1.61 | 1.48 |
| darcy003, mario002 | both | MA57 | 1.20-1.22 | 1.19-1.23 (unchanged) |

On the 50-matrix corpus (1 thread, default budget) total analysis falls from 15.4 s to 11.7 s (0.76x). The dear-factor adopters now cost exactly what the exact trial (budget -1) costs.

**8 threads** (12 trial-heavy cases, both tracks, interleaved A/B, 3 rounds). Here the concurrent trial adds to the 1-thread changes. Factors are identical (entries, flops, delays, accuracy).

| case | analyse ms | wall ms |
|---|---|---|
| TSOPF_FS_b300_c3 (ipopt) | 2418 → 796 (0.33) | 3032 → 1401 (0.46) |
| TSOPF_FS_b300_c2 (ipopt) | 1532 → 542 (0.35) | 1886 → 902 (0.48) |
| TSOPF_FS_b300_c1 (ipopt) | 883 → 460 (0.52) | 982 → 560 (0.57) |
| ncvxqp5 (ipopt) | 838 → 514 (0.61) | 1385 → 1057 (0.76) |
| cvxqp3 (ipopt) | 262 → 171 (0.65) | 481 → 387 (0.81) |
| ncvxqp7 (ipopt) | 2058 → 1478 (0.72) | 5017 → 4424 (0.88) |
| case39 (ipopt) | 120 → 93 (0.77) | 138 → 110 (0.80) |
| ncvxqp3 (ipopt) | 2399 → 1959 (0.82) | 4724 → 4262 (0.90) |
| boyd1 (ipopt) | 98 → 87 (0.88) | 117 → 105 (0.90) |
| pglib case2000 (ipopt) | 44 → 39 (0.90) | 110 → 107 (0.97) |
| darcy003 / cops_pinene_400 | within noise (0.94-1.02) | within noise |

Shifted geomean on these cases: hard_indef wall 0.735 (tuning, 12 case-tracks) and 0.750 (held-out, 8); analyse 0.63 and 0.64. The rejecters (darcy003, PGLib) gain little: their trial is cut short by the initial verdict anyway, and AMD and the budgeted searches are the critical path.

**IPOPT replays** (all 16 `ipopt_hard.json` models, every stored KKT step, ipopt track, u = 1e-8; main vs branch). Gate **PASS**:
- 0 failing steps on either side;
- the same last-step inertia;
- the same delays and factor entries on all 16 models.

Their analyse times are unchanged too: none of these models has a dear factor, so their trials run as on main. cops_chain_800's replay record carries no per-step validations on either side (pre-existing).

The only inertia flag in the front-tier tables, stokes128 on the ipopt track, is off by one against the majority (smf agrees with MA97). It is the same on main.

## Where analysis time went

`bench_matching_trial --pieces` (new) times each piece of the trial standalone, 1 thread, median of 3, ms:
- ctor: matcher construction (cost graph, duals, greedy, length-2 augmentations);
- budget / rest: the searches up to the default budget / the rest of them;
- initial: the initial matching's compressed ordering, AMD only / Auto (Auto adds METIS when the compressed factor is dear);
- exact: the exact matching's compressed ordering;
- base AMD: adjacency plus AMD, i.e. analyse's "ordering" phase.

The pieces are main's algorithm; the analyse column is analyse() of the first commit, which already skips METIS where compressed AMD passes.

| case | trial | ctor | budget | rest | initial AMD / Auto | exact | base AMD | analyse |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ncvxqp3 | resumed | 5 | 13 | **1614** | 57 / **408** | 328 | 48 | 2448 |
| ncvxqp7 | resumed | 5 | 14 | **1088** | 81 / **511** | 347 | 68 | 2076 |
| ncvxqp5 | resumed | 4 | 12 | 212 | 37 / **286** | 272 | 28 | 590 |
| cvxqp3 | resumed | 1 | 3 | 105 | 13 / **81** | 54 | 11 | 265 |
| TSOPF_FS_b300_c3 | resumed | 81 | 48 | 382 | **395 / 1248** | 190 | **402** | 1623 |
| TSOPF_FS_b300_c2 | resumed | 54 | 33 | 252 | **224 / 797** | 131 | 214 | 982 |
| TSOPF_FS_b300_c1 | resumed | 26 | 18 | 126 | 89 / **361** | 243 | 62 | 609 |
| c-big | exact | 28 | 54 | 0 | n/a | **989** (METIS kept) | 84 | 1256 |
| case39 | resumed | 6 | 4 | 56 | 15 / 14 | 16 | 12 | 118 |
| darcy003 | rej-init | 34 | 64 | (32) | 75 / 74 | (78) | 50 | 279 |
| pglib case2000 | rej-init | 4 | 11 | (92) | 13 / 12 | (11) | 7 | 43 |

Two things dominated:
1. **The initial matching's compressed ordering on adopters.** It ran before the base ordering's flops were known, so Auto tried METIS whenever the compressed factor qualified. Its only use is a resume-or-reject verdict, and its ordering is then thrown away. On the resumed adopters with dear factors it cost 0.3-1.2 s. That made the budgeted trial *slower* than the exact one: on main, TSOPF_FS_b300_c3 analysed in 2.4 s with the default budget and 1.17 s with budget -1.
2. **The exact searches on ncvxqp3/7.** Search diagnostics (temporary instrumentation, not committed) on ncvxqp3:
   - 16,421 searches finalize 21.9 M rows, 1,332 per search;
   - 97% of those rows come off the zero-reduced-cost "level" lists, i.e. tight plateaus that every search re-explores;
   - no search ends at distance 0;
   - 14,661 of the 16,421 path lengths are distinct.

## Changes

1. **Deferred initial ordering** (`605564e`). The initial matching is ordered at the decision point, once the base flops are known.
2. **Concurrent trial at `num_threads > 1`** (`605564e`). The matcher's searches and the exact matching's ordering run on a worker thread, while the calling thread computes the base AMD ordering, its counts and the initial verdict. A rejection stops the worker between search slices.
   - Same results as the serial path: new test `MatchingTrial.ConcurrentTrialIsTheSerialOne` covers every budget path at 1 vs 4 threads.
   - Uses std::threads, like the tree-parallel factor.
3. **Cost graph in one pass** (`605564e`). For cleaned input there is one `log` per stored entry, and no copy-and-compact pass. Bitwise the same graph.
4. **A dear factor resumes the trial outright** (`89f4f19`, `5f76656`). Once the budget pauses the searches, they resume without an initial verdict when either factor is dear: the base ordering's predicted flops, or the initial matching's compressed AMD ordering's, times `expected_refactors`, reach `kMetisFlopsPerRow` = 4.5e5 per row.
   - For the compressed ordering, that is exactly when Auto would have tried METIS. The initial verdict is now built from compressed AMD alone, and `matching_order_from` keeps main's signature.
   - **Why it is safe.** A ResumedExact trial is bitwise the exact trial (budget -1), as the existing tests assert. Resuming instead of rejecting can therefore only move a verdict *towards* the exact trial's, never away from it.
   - **Corpus margin.** Every corpus rejection has base flops of at most 7.1e4 per row (turon_m; PGLib <= 820, darcy003 2.2e3) and compressed AMD flops of at most 2.7e5 (turon_m). The dear adopters sit at 2.8e5-1.5e7.
   - Test: `MatchingTrial.DearBaseFactorResumesWithoutInitialVerdict`, serial and threaded.

ctest 67/67. The matching-trial binary passes 11/11, including the TSOPF_FS_b162_c1 fixture test (the large fixtures linked into the worktree).

## Levers evaluated

| lever (from the brief) | outcome |
|---|---|
| 1. Two-tier matching | **Done in the decision structure:** the cheap tier (the greedy initial matching) now never pays METIS, and is skipped when the factor is dear. The exact searches that remain run only on adopters, which need the exact matching for their ordering and scaling. A different (auction or epsilon-optimal) matching would change both. An exact batched (primal-dual, many augmentations per phase) variant needs equal path lengths, and ncvxqp3's are 89% distinct. |
| 2. Skip or short-circuit the trial with predictors | **Done for adopters** (the dear-factor rule). **Not found for rejecters** (darcy003/mario002, PGLib); see below. |
| 3. Run the candidates concurrently | **Done** (`ConcurrentTrial`). At 8 threads analysis on the trial-heavy cases falls to 0.33-0.88x of main, against 0.47-1.0x at 1 thread (TSOPF_FS_b300_c3 0.33x, case39 0.77x). |
| 4. Why is MA57's analysis 6x cheaper on darcy003/mario002? | MA57's analysis is structural only: `ma57ad_` gets the pattern, and its MC64 scaling runs inside the factorization (`ma57bd_`, 219 ms there against smf's 64 ms). smf's 279 ms is 178 ms of matching trial, which always rejects (initial ratio 3.68), plus about 100 ms of base analysis: AMD and adjacency 50, counts 22, the rest 28. Without the trial, smf would analyse in about 2x MA57's time and factor 3.4x faster. |

**Tried and dropped:**
- **The base ordering with its pairs merged in place, as a cheap first candidate for the initial verdict.** It would cost one counts pass instead of compressed AMD. Its flops ratio passed 2.4 on only 8 of 50 matrices: it fails TSOPF_FS_b300_c2/c3 (3.7, 4.5) and ncvxqp3/7 (8.9, 12.3), and adds 2-37 ms to every rejection. Net loss.
- **A two-pass relax loop**: a branch-free distance scan per column chunk, then an in-order apply. It is output-identical, but TSOPF and case39 searches got 1.6-2.0x *slower* and ncvxqp3 only 5% faster. Reverted.
- **A degree-based compression proxy to skip sure rejections**: the squared compressed degrees of the initial pairs against the plain ones. It does not separate the classes. darcy003 scores 2.0 while adopters reach 2.8-4.0 (bratu3d 3.0, cont-201 2.8, boyd2 4.0), and PGLib's 4.8-5.1 overlaps TSOPF_FS_b300's 4.1.

## What is left

- **ncvxqp3/7 exact searches: 1.1-1.6 s, with no exact-preserving fix in sight.** Every change in the search order or algorithm changes the duals, and so the matching scaling. Constant-factor tuning of the relax loop failed (above). Combined with the cost-model amalgamation branch (ncvxqp3 factor 2.7 s → about 1 s), ncvxqp3 would still lose to MA97 at 1 thread (about 3.0 s against 2.17 s).
  - The bigger lever is lever L3 of `hard-gaps-diagnosis`: a plain METIS candidate. MA97's METIS ordering, without matching or scaling, factors ncvxqp3 in 1.49 s and analyses it in 0.64 s.
  - It changes orderings, so it needs an accuracy guard (TSOPF_FS_b300_c2 ipopt and bratu3d).
- **darcy003/mario002 (4 tuning case-tracks at 1.19-1.23 against MA57).** The trial costs 178 ms and always rejects. Skipping it would make smf about 0.65x of MA57 there. That needs a predictor that knows the compressed ordering's fill without computing it; none of the cheap ones tested above works.
- **The TSOPF base AMD (60-400 ms).** It is needed only for the verdict's base flops, since the matching ordering wins about 50x there. An AMD that stops once its running flop count exceeds the matching ordering's would avoid most of it. That means vendoring SuiteSparse AMD (BSD-3).

## Reproduce

- Build: `build/` in the worktree, Release with `/Zi`. Run from the main checkout root, where the fixtures live.
- Corpus survey: `bench_matching_trial --list list50.txt --trials 3`, base and branch binaries. The list is the 50-matrix corpus of the 2026-10-28 fast-matching survey. Compare `perm_hash`, verdicts and `r_exact`.
- Pieces: `bench_matching_trial --list ... --pieces --trials 3`.
- A/B results are in `benchmarks/results/analyse_ab_2026-10-29/`:
  - `tuning_1t_final/`, `heldout_1t_final/`, `final3_1t/` and `final3_8t/`, each with a `base/` and a `new/` directory of records.jsonl;
  - `tuning_1t_final_merged/`: the tuning records with the final cvxqp3 runs swapped in, which the tuning scores above use;
  - `compare_*.md` comparisons;
  - `score_*/`: front-tier scores against the corrected competitor records;
  - `replay_gate.md`.
- Driver: a two-binary variant of the amalgamation branch's `tools/bench/ab_run.py`, with the same protocol: `run.py`, 1 trial per run, variant order alternating by case and round, below-normal priority.
