# Scoreboard against every other solver (2026-10-30)

One page: where smf stands against every other solver.

**Builds:** `integ/amalg-cheaper-trial` (cost-model amalgamation plus the cheaper matching trial) was merged to main the same day (PR #14). In this file, "main" means main *before* that merge, and "integ branch" means main *after* it. The decisions taken with the merge are in section 2.

**Scores** are smf time / other solver time, as a shifted geometric mean (10 ms shift, `score.py`'s method). **Below 1 means smf is faster.** "W/L" counts the case-tracks where smf is faster or slower. Only case-tracks that both solvers solve accurately are timed. The accuracy gate is eta ≤ 1e-10 and omega ≤ 1e-8.

**Data:**
- **1 thread:** a fresh interleaved run of main against `integ/amalg-cheaper-trial`, 2026-10-30 09:51-10:21. It covers all 144 front-tier case-tracks, 3 rounds each, smf only. Records: `benchmarks/results/integ_ab_20261030/`.
- **Competitors:** the corrected records (SSIDS threading fixed) from `fronttier_20261029-ssidsfix/score_fm_ssidsfix_t816`.
- **8/16 threads:** that same board. Its smf is the pre-merge main.
- **Scripts and scores:** `integ_ab_20261030/score_vs_competitors/`.

## 1. Where smf stands

### 1a. Against the best other solver on each case (the hard score)

Each case-track is timed against whichever solver was fastest on it. Cells are tuning | held-out.

| class / track | main 1T | **integ branch 1T** | main 8T | main 16T |
|---|---|---|---|---|
| small_kkt recommended | 0.92 \| 1.00 | 0.93 \| 0.99 | — | — |
| small_kkt ipopt | 0.53 \| 0.78 | 0.53 \| 0.78 | — | — |
| large_kkt recommended | 0.72 \| 0.72 | 0.73 \| 0.73 | 0.66 \| 0.45 | 0.67 \| 0.47 |
| large_kkt ipopt | 0.88 \| 1.04 | **0.81 \| 0.96** | 0.80 \| 0.71 | 0.87 \| 0.76 |
| hard_indef recommended | 1.13 \| 1.16 | **0.92 \| 0.96** | 0.88 \| 1.07 | 0.89 \| 1.09 |
| hard_indef ipopt | 1.11 \| 1.19 | **0.89 \| 1.00** | 0.92 \| 1.18 | 0.91 \| 1.18 |
| spd | 0.93 \| 0.97 | **0.87 \| 0.94** | 0.55 \| 0.61 | 0.51 \| 0.62 |

- **At 1 thread, main loses only on hard_indef. The integration branch is at or below 1.0 in every class.** smf is the fastest solver on 80 of 137 case-tracks with main and on 99 of 137 with the branch. Losses above 1.15x fall from 43 to 21.
- **At 8/16 threads, the remaining loss is hard_indef held-out (1.07-1.18), mostly against SSIDS.** The integration branch has not been run on the 8/16-thread board yet. Its amalgamation part measured 0.55x factor time at 8/16 threads on 12 case-tracks (c-big at 16T: 14.0 s → 1.7 s, against SSIDS's 2.2 s), so most of that gap should close.

### 1b. Head to head: smf against each solver

1 thread, all classes, main → integ branch. W/L and accuracy are for the integration branch. Accuracy is identical on main: no case changed status.

| solver | smf / solver time | W / L | case-tracks solved accurately: solver vs smf |
|---|---|---|---|
| MA27 | 0.36 → **0.34** | 77 / 28 | 105 vs 123 (of 125) |
| MA57 | 0.67 → **0.60** | 104 / 14 | 119 vs 138 (of 140) |
| MA97 | 0.68 → **0.62** | 90 / 17 | 107 vs 135 (of 137) |
| SSIDS | 0.42 → **0.37** | 126 / 6 | 132 vs 135 (of 136) |
| MKL PARDISO | 0.65 → **0.58** | 70 / 6 | 76 vs 137 (of 139) |
| MUMPS | 0.30 → **0.26** | 108 / 0 | 108 vs 134 (of 136) |
| CHOLMOD (SPD only) | 0.78 → **0.77** | 11 / 0 | 11 vs 11 |
| cuDSS (GPU, reference only) | 0.68 → **0.62** | 44 / 12 | 56 vs 138 (of 140) |

Per class, integration branch, 1 thread (time ratio, W/L):

| solver | small_kkt | large_kkt | hard_indef | spd |
|---|---|---|---|---|
| MA27 | **1.09** (16/16) | 0.25 (19/5) | 0.17 (32/7) | 0.23 (10/0) |
| MA57 | 0.78 (29/2) | 0.50 (24/0) | 0.59 (39/12) | 0.50 (12/0) |
| MA97 | 0.81 (25/1) | 0.67 (17/7) | 0.53 (36/9) | 0.51 (12/0) |
| SSIDS | 0.20 (31/0) | 0.25 (26/1) | 0.59 (57/5) | 0.47 (12/0) |
| MKL PARDISO | 0.80 (20/0) | 0.60 (15/1) | 0.44 (25/3) | 0.62 (10/2) |
| MUMPS | 0.66 (20/0) | 0.24 (19/0) | 0.18 (57/0) | 0.39 (12/0) |

- **MA27 on small KKTs (1.09x) is the only pairing smf does not win.** MA27 is a reference column and is not in the best-competitor set.
- **16 threads, main** (all classes): MA97 0.51, SSIDS 0.56, PARDISO 0.46, MUMPS 0.23. Against SSIDS on hard_indef it is 0.77 (40 W / 22 L).

### 1c. What still loses (integration branch, 1 thread, against the best competitor)

- **Analysis-heavy cases.** smf's analysis alone takes longer than the rival's whole run:
  - darcy003 and mario002: 1.36-1.40x against MA57; analysis is 330 of 410 ms.
  - boyd1/boyd2: 1.30-1.80x.
  - case39: 1.50x.
  - ncvxqp3 ipopt: 1.35x against MA97; analysis is 2.0 s of 2.9 s.
- **Small, fast cases against MA97:**
  - stokes128 recommended: 2.46x (163 ms against 60 ms).
  - ncvxqp9: 2.24x.
  - cops_pinene_400 ipopt: 2.14x.
  - cops_elec_200 recommended: 1.60x.
- **At 16 threads, main:**
  - c-big: 4.1x against SSIDS. The branch should flip this.
  - nd24k: 2.6x.
  - ncvxqp7: 2.1-2.3x; ncvxqp3: 1.9x.
  - TSOPF_b300_c3: 1.8-2.1x.

**Caveats:**
- The 1-thread absolute level drifts by about ±0.1 from run to run. Main's hard_indef read 1.04 in an earlier quiet run today, and 1.13 here.
- The main → branch delta is clean, because the two ran interleaved in the same run.
- The competitors were not re-run today.

## 2. Merge decisions (2026-10-30)

**Merge `integ/amalg-cheaper-trial` as is**, with `scale_on_delay_ratio` left at 16.

**1. cops_polygon_200, replay step 68 (omega 1.37e-8 against the 1e-8 gate): accepted.**
- It is one late, near-singular iterate out of 70. Every other step and model of the 16-model replay gate passes.
- The normwise backward error is 7e-17.
- IPOPT end-to-end takes the same path: 40 iterations, the same objective to every printed digit, status Optimal, and less factorization time (0.335 → 0.274 s).
- Tightening the delay guard to pass this one step would give up merges everywhere else.

**2. The pde_control regression: the scale-on-delay ratio stays at 16.**

Sweep: `integ` binary, `SMF_BENCH_SCALE_ON_DELAY`, every indefinite recommended-track case-track (66), 2 interleaved rounds. Records: `benchmarks/results/sod_ratio_sweep_20261030/`.

| ratio | pde_control_250 | pde_control_120 | cops_elec_200 | pglib case300 | all 66, against ratio 16 |
|---|---|---|---|---|---|
| **16** | 2774 ms | 271 ms | 42 ms (1.62x MA97) | 24 ms | 1.000 |
| 8 | 1974 | 189 | 77 (**2.72x** MA97) | 32 | 0.993 |
| 6 | 1531 | 167 | 75 (**2.66x**) | 32 | 0.990 |
| 4 | 986 | 147 | 72 (2.58x) | 32 | 0.981 |

- A lower ratio fixes pde_control. That is only a regression against smf's own old tree: the best competitor there is SSIDS at 25.3 s (pde_control_250) and 1.18 s (pde_control_120), so smf leads by 4-25x at every ratio.
- At a ratio of 8 or lower, cops_elec_200 and case300 start rescaling. That turns cops_elec_200 from a 1.6x loss to MA97 into a 2.7x loss.
- The aggregate moves by 1-2%, which is the noise level of a 2-round sweep.
- A ratio of 4 also breaks `test_scale_on_delay`'s premise (`ratio > 4`).
- The proper fix is a trigger that tracks cost better, such as flops instead of eligible × rows². That is follow-up 1 of `2026-10-29-cost-model-amalgamation.md`.

**3. Two other experiments are not merged.** A per-front APTP retry after TPP delays (`feat/hybrid-aptp`) and a vendored AMD for the analysis (`feat/own-md`) were measured as well; neither has a net gain on top of main.

**Next:**
- Run the merged main on the 8/16-thread board. Expected: c-big flips against SSIDS, and the held-out hard_indef gap at 8/16 threads narrows.
- Then the flops-based scale-on-delay trigger.
- Then the analysis-heavy 1-thread losses: darcy003 and mario002 against MA57, boyd1/2, case39 (section 1c).
