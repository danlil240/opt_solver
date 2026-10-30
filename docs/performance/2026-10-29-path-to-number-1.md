# Path to #1: deep research + measurements (2026-10-29)

**Inputs:**
- A deep-research run: 104 agents, 22 primary sources, 110 claims extracted, 25 verified by 3-vote adversarial checks, 13 confirmed.
- Spot measurements on this machine at main `4d8421e` (binaries from 2026-10-28), compared against the front-tier records in `benchmarks/results/fronttier_20261029-day/score_fm`.

**Caveat:** every number below is a single trial on a desktop with background load. Re-measure before adopting anything.

## TL;DR

1. **The scoreboard was wrong. Fixed and re-baselined 2026-10-30 (section 1).**
   - The "1-thread" SSIDS records actually ran on **16 cores**.
   - The "8/16-thread" SSIDS records were **oversubscribed**.
   - So the 1-thread hard_indef and spd scores overstated smf's losses, and the 8/16-thread scores overstated smf's wins.
   - The memory and roadmap claim that "SSIDS is the 1-thread rival" came from this bug.
   - **Corrected:** hard_indef loses at 1 thread (1.13-1.22) and on held-out at 8/16 threads (1.07-1.18), where properly threaded SSIDS is the main rival. spd and large_kkt win at 8/16 threads and are at parity or better at 1 thread (worst: large_kkt ipopt held-out, 1.03).
2. **The biggest real lever the literature did not name is amalgamation.**
   - On the hard KKT matrices, smf's factorization is memory-bound: narrow supernodes carry huge contribution blocks.
   - On c-big, about 70% of factor time is assembly, zeroing and memmove; dense BLAS is about 25%.
   - Merging more aggressively made the factorization 2.5-4.3x faster on c-big, ncvxqp3 and cvxqp3, and was neutral on SPD.
   - Neither existing rule works everywhere, so this needs a cost-model rule.
3. **Parallel scaling on large fronts is the next gap.** At 16 threads smf speeds up only 1.2-1.7x on the biggest problems (nd24k, c-big, ncvxqp3). SSIDS's factorization speeds up 2.9-3.9x on them.
4. **Analysis cost on hard indefinite matrices** is the third gap: ncvxqp3 3.1 s against SSIDS 1.8 s, TSOPF_b300_c2 1.6 s against 1.0 s. The verified literature backs two-tier matching.
5. **The verified literature's #1 lever (APTP with a blocked in-front fallback) is already in smf.** Turning it on does not move c-big, TSOPF or ncvxqp3 (measured below). Pivoting is not where the remaining gap is.

## 1. Benchmark bug: SSIDS ignores `OMP_NUM_THREADS`

`tools/bench/run_wsl.py` runs SSIDS with `OMP_NUM_THREADS=t` and `OPENBLAS_NUM_THREADS=t`. SSIDS sizes its OpenMP regions from the hwloc topology (16 physical cores), so the `OMP_NUM_THREADS` setting does not cap it.

| check (cfd2 / bmwcra_1, SPD) | as benchmarked "1T" | `OMP_THREAD_LIMIT=1` | smf 1T |
|---|---|---|---|
| peak OS threads | **16** | 1 | 1 |
| cfd2 factor | 150 ms (208 GF/s) | 822 ms (38 GF/s) | **597 ms** (48 GF/s) |
| bmwcra_1 factor | 296 ms (178 GF/s) | 1274 ms (41 GF/s) | **1124 ms** (47 GF/s) |

The fingerprints in the records are consistent with this:
- "1-thread" SSIDS reaches 100-1300 GF/s, while one core peaks at about 70-80 GF/s.
- SSIDS gets *slower* with more threads: bmwcra_1 takes 188 ms at "1T", 506 ms at 8T and 1058 ms at 16T. At t > 1, OpenBLAS threads multiply inside SSIDS's 16 OpenMP threads.

**Fix** (`run_wsl.py`, SSIDS env):
- Set `OMP_THREAD_LIMIT=t`.
- Set `OPENBLAS_NUM_THREADS=1`: SSIDS does its own task parallelism, and BLAS threads only oversubscribe.
- Optionally use `taskset -c 0-(t-1)`. hwloc honours the cpuset.

Add a sanity gate to `score.py` so this cannot recur silently. It should flag any record where factor flops / factor time exceeds about 1.2x the per-core peak times threads, and record the peak thread count.

### Corrected standing (spot checks, true 1-thread SSIDS)

| case (recommended) | SSIDS recorded | SSIDS true 1T | smf 1T | ratio now (recorded) |
|---|---|---|---|---|
| cfd2 (spd) | 1.07 s | ~1.8 s | 1.52 s | **~0.84 (1.41)** |
| bmwcra_1 (spd) | 1.13 s | ~2.2 s | 1.99 s | **~0.9 (1.76)** |
| TSOPF_FS_b300_c2 | 1.14 s | 1.86 s | 2.58 s | ~1.38 (2.26) |
| ncvxqp3 | 2.29 s | 4.0 s | 5.77 s | ~1.44 (2.51) |
| c-big | 3.49 s | 9.1 s | 17.2 s | ~1.88 (4.91) |

- SPD at 1 thread is probably already a win.
- hard_indef at 1 thread is still a loss, but about half the reported size.
- At 16 threads, the old "1T" SSIDS records *are* a well-configured 16-core SSIDS. Against them, smf at 16 threads is 0.50 shifted geomean over 60 cases, but it loses 1.8-4.2x on c-big, nd24k, ncvxqp3/7, TSOPF_b300* and cvxqp3 (section 3).

**Action (done 2026-10-30, below):** re-run the SSIDS stage at 1/8/16 threads with the fixed env and re-score. `docs/roadmap.md` is still to update. MA97 and PARDISO look honest at 1 thread: their bmwcra_1 times match single-core throughput.

### Re-scored after the 1-thread SSIDS re-run (2026-10-29/30)

Records are in `benchmarks/results/fronttier_20261029-ssidsfix/`:
- `t1_ssids/`, 2026-10-29 21:48-22:29: every 1T case with the fixed env. It overlapped an smf A/B sweep from another session.
- `t1_ssids_retime/`, `t1_ssids_retime2/` and `t1_ssids_retime3/`, 2026-10-30, quiet machine: the 30 matrices where SSIDS is the best competitor or within 40% of it. SSIDS ran 4-29% faster than in the overlapped run (median about 12%). No near-miss case changed its best competitor.
- `t8_16_ssids/`, 2026-10-30 07:49-08:18, quiet machine: 8 and 16 threads on the 60 matrices that have 8/16-thread records.

`score_fm_ssidsfix_t816/` holds the 2026-10-29 fast-matching records with every old SSIDS row replaced by these runs; its `score_*.md` are the current scoreboard. Only SSIDS was re-run; the smf and other competitor rows are the 2026-10-29 day run.

| 1 thread (tuning \| held-out) | before (bug) | corrected |
|---|---|---|
| spd | 1.18 \| 1.33 | **0.95 \| 0.99** |
| hard_indef recommended | 1.36 \| 1.25 | **1.16 \| 1.18** |
| hard_indef ipopt | 1.30 \| 1.34 | **1.13 \| 1.22** |
| large_kkt recommended | 0.83 \| 0.83 | 0.73 \| 0.73 |
| large_kkt ipopt | 0.93 \| 1.03 | 0.88 \| 1.03 |
| small_kkt | unchanged | unchanged (±0.01) |

The overlapped run alone gave hard_indef 0.02-0.06 lower (better for smf). Do not quote it.

**The remaining 1-thread losses are mostly analysis, against MA97 and MA57.** 46 of 137 case-tracks lose by more than 1.15x.
- **Analysis-dominated (29 case-tracks, 60% of the summed log-loss).** Examples (smf against the best competitor):
  - ncvxqp3: 3.1 s against MA97's 0.64 s;
  - darcy003 and mario002: 360 ms against MA57's 57 ms;
  - boyd1/2: 100-130 ms against 22-29 ms;
  - case39: 124 ms against 23 ms;
  - cops_pinene: 43 ms against 4 ms;
  - TSOPF_b300*: 0.9-2.4 s against 0.45-1.4 s.
- **Factor-dominated (17 case-tracks, 40%):** c-big, stokes128, ncvxqp9, bratu3d, G3_circuit and nd24k.
- **SSIDS is the best competitor in 10 of the losses**, all hard: TSOPF_b300_c1/c2/c3 on both tracks (1.53-1.76x), ncvxqp7 on both tracks (1.50-1.60x), cvxqp3 recommended (1.44x) and nd24k (1.50x).

**This re-ranks the levers for 1 thread.** Analysis cost (section 4) comes first and amalgamation (section 2) second. Amalgamation stays first for 8/16 threads.

**Caveat:** pglib_opf_case2869/9241 time out at 150 s on SSIDS at 1 thread (both tracks).

### 8/16 threads after the SSIDS re-run

Properly threaded SSIDS's factorization speeds up 2.5x at 8 threads and 2.2x at 16 (geomean over 108 case-tracks; 3.8x on nd24k, 2.9x on c-big, 8.1x on G3_circuit at 16). Its analysis stays serial. The highest throughput is 30 GF/s per thread, well inside the `score.py` gate.

| 8 \| 16 threads (tuning / held-out) | before (bug) | corrected |
|---|---|---|
| hard_indef recommended | 0.78 \| 0.75 / 0.99 \| 0.95 | **0.88 \| 0.89 / 1.07 \| 1.09** |
| hard_indef ipopt | 0.83 \| 0.80 / 1.13 \| 1.07 | **0.92 \| 0.91 / 1.18 \| 1.18** |
| spd | 0.53 \| 0.45 / 0.54 \| 0.53 | 0.55 \| 0.51 / 0.61 \| 0.62 |
| large_kkt recommended | 0.63 \| 0.62 / 0.42 \| 0.43 | 0.66 \| 0.67 / 0.45 \| 0.47 |
| large_kkt ipopt | 0.76 \| 0.83 / 0.71 \| 0.76 | 0.80 \| 0.87 / 0.71 \| 0.76 |

SSIDS is now the best competitor on 17-18 of the 21 tuning hard_indef case-tracks at 8/16 threads.

**At 16 threads, smf loses by more than 1.15x on 30 of 103 case-tracks.** SSIDS is the best competitor in 19 of them, MA97 in 9 and PARDISO in 2.
- **Factor-dominated (17 case-tracks, 63% of the summed log-loss).** 16-thread factor time on the recommended track, with the wall-time score ratio over both tracks:

  | case | smf factor | SSIDS factor | wall ratio |
  |---|---|---|---|
  | c-big | 13.2 s | 2.22 s | 4.1-4.3x |
  | nd24k | 28.3 s | 8.25 s | 2.6x |
  | ncvxqp7 | 2.64 s | 0.77 s | 2.1-2.3x |
  | ncvxqp3 | 2.08 s | 0.69 s | 1.9x (2.9x on ipopt, against MA97) |
  | TSOPF_b300_c3 | 1.06 s | 0.25 s | 1.8-2.1x |

  The same holds for cvxqp3 (1.8-1.9x).
- **Analysis-dominated (13 case-tracks, 37%):** TSOPF_b300_c1/c2 (smf 0.9-1.5 s against SSIDS 0.4-0.9 s), ncvxqp5, boyd1, cops_pinene and case39.

This confirms the ranking for 8/16 threads: amalgamation (section 2) first, cross-front parallelism (section 3) second, then analysis cost.

## 2. Lever 1: cost-model amalgamation (measured, largest)

c-big factor profile at 1 thread (`bench_profile --sample factor --lines`):

| where | share |
|---|---|
| extend-add loop, `multifrontal_serial.cpp` (`Fc[cm[ii]] += *cv++`) | 32% |
| memchr (most likely misattributed zeroing) + memmove (contribution-block copies) | 39% |
| MKL BLAS | 25% |

smf does *fewer* flops than SSIDS here (1.83e11 against 3.1e11) but runs at 11 GF/s against 39.

The cause is the current rule (`supernode_detection.cpp:463`), which merges only when *both* child and parent have fewer than `nemin` = 8 pivots. A node with 10-100 pivots and a CB with thousands of rows is never merged. Its CB is copied, zeroed into the parent and extend-added at roughly 1 flop per word moved.

| case | factor @ nemin 8 | nemin 32 | nemin 128 | CHOLMOD rule |
|---|---|---|---|---|
| c-big | 17.0 s | 7.0 s | **4.0 s** | 18.7 s |
| ncvxqp3 | 2.85 s | 1.62 s | 1.16 s | **1.08 s** (delays 441 to 83) |
| cvxqp3 | 270 ms | 128 ms | **100 ms** | 112 ms |
| helm3d01 | 201 ms | 189 ms | 180 ms | |
| cfd2, bmwcra_1, TSOPF_b300_c2 | ±5% | ±5% | ±5% | |
| stokes128, ncvxqp9 | | | +10-13% (worse) | |

**Design:**
- Merge a child into its parent when the data-movement cost of its CB exceeds the padding flops the merge adds.
  - The CB costs about 3·cb² words: copy out, zero, then extend-add.
  - The merge adds about (explicit zeros) × (rows below).
- Keep the zero-share guard for small fronts, which is where stokes128 and ncvxqp9 lose.
- Complementary: assemble the last child in place, as in MUMPS/MA41. Lay the parent front over the largest child's CB on the stack so it is neither copied nor zeroed. This also cuts peak memory.

**Expected effect:** c-big moves from smf's worst case (4.9x recorded, about 1.9x true) to ahead of SSIDS and MA97. ncvxqp/cvxqp lose about half their factor time. At 8/16 threads it also creates the wide fronts that tile parallelism needs.

## 3. Lever 2: parallel scaling of the top of the tree (architecture)

smf's 16-thread speedup over its own 1-thread time:

| case (recommended, wall) | smf 1T | smf 16T | speedup | SSIDS 16T (fixed run) |
|---|---|---|---|---|
| nd24k | 51.8 s | 31.2 s | 1.66x | 12.2 s |
| c-big | 17.2 s | 14.5 s | 1.19x | 3.55 s |
| ncvxqp3 | 5.77 s | 4.46 s | 1.29x | 2.34 s |
| TSOPF_FS_b300_c3 | 3.80 s | 3.46 s | 1.10x | 1.66 s |

Part of this is lever 1 (narrow fronts cannot use threads) and part is the serial analysis (lever 3). The rest is architecture. Today's engine is "serial tree + threaded BLAS". It has APTP 2D top tiles, but only at u ≥ 1e-6 and f ≥ 512, and no cross-front pipelining (the deferred "Task 13b").

The SSIDS/MA87/PaStiX design is one DAG of tile tasks that spans fronts:
- a parent's tiles start as soon as the child tiles they depend on are final;
- independent subtrees run as tasks under a shared memory budget.

**Plan:**
1. Re-measure after lever 1.
2. Extend tile tasks to SPD fronts and to the u = 1e-8 (ipopt) track.
3. Add cross-front pipelining.

Gate it on the corrected 16-thread scoreboard against MA97 and properly threaded SSIDS.

## 4. Lever 3: analysis cost on hard indefinite matrices

ncvxqp3 analysis takes 3.1 s against SSIDS 1.8 s, TSOPF_b300_c2 1.6 s against 1.0 s, and c-big 1.3 s against 1.26 s. These are all serial at 16 threads too. Verified evidence:
- **Two-tier matching** (Hogg & Scott, NLA 22(4) 2015, [ePubs 11539865](https://epubs.stfc.ac.uk/work/11539865)). An approximate auction matching is as good as MC64 *for scaling*. Matching-based *ordering* needs a near-optimal matching: auction cardinality below 99% gave ncvxqp1 about 28,000 delays against 40. Run the auction to 100% cardinality, or exact matching, only when the compressed-ordering candidate is actually evaluated. The `feat/fast-matching` branch, which cut matching time to 0.77x, is a partial step.
- **Run the candidates concurrently** at num_threads > 1: AMD, METIS and the matching trial are independent. This is an inference, not a published result.
- **ParAMD** ([arXiv 2504.17097](https://arxiv.org/abs/2504.17097)) gives 3.2-7.3x ordering speedup at 64 threads with about 1.1x worse fill. That is a minor lever at 8-16 threads and for IPOPT-sized matrices.
- **ML ordering selection** ([arXiv 2511.10180](https://arxiv.org/abs/2511.10180)) is weak evidence: general matrices, not KKT, and an always-AMD baseline. At most, use a cheap feature predictor to skip clearly losing trials.

## 5. Pivoting: what the literature says, and what we measured

Verified, 3-0 unless noted:
- **APTP** (Duff, Hogg & Lopez, SISC 42(2) 2020; RAL-TR-2016-017) reaches Bunch-Kaufman-like speed with TPP stability. **smf already has it** (`ldlt_aptp.cpp`, with a blocked TPP pass on failed columns).
- **The failed-pivot policy decides the hard cases** (2-1). Skipping SSIDS's in-front TPP pass saved 21-33% on ncvxqp/cvxqp but made TSOPF 24-44x slower, because the fronts grow. SSIDS's fallback is unblocked. An adaptive in-front-vs-delay policy is unpublished but plausible.
- **Measured on smf:** APTP (≥ 64 or ≥ 256 eligible columns, or all fronts) changes c-big, TSOPF_b300_c2 and ncvxqp3 by ±5%, with the same delay counts. smf's 30k delays on TSOPF against SSIDS's 2.8k come from ordering and scaling. SSIDS runs its matching ordering with MC64 scaling there. That is a lever-3 and ordering-policy question, not a kernel one.
- **Keep MC64SYM-quality scaling** (Scott, RAL-TR-2009-007). On HSL_MA77 it cut factor time about 26x on cvxqp3, c-62 and ncvxqp1, and turned ncvxqp3/5 from unsolved to seconds.
- **Matching-compressed 2x2 ordering** (Duff & Pralet SIMAX 2005; Schenk & Gärtner ETNA 2006) gives up to 10⁴x fewer delays and zero failures, but about 2x the factor time. Keep it behind a per-matrix decision; smf already does.

## 6. Do not do

- **Pivot-free constrained orderings** (Bridson-style, u = 0): factors 2.3-11x denser, 6.5-47x slower on cvxqp3, ncvxqp3 and c-62/68 (Scott 2009, vote 2-1).
- **PARDISO-style static perturbation as the default:** 3/38 failures without matching, and about 2x cost with it. Keep it as an opt-in static-schedule mode. It is also the natural base for a future GPU path.
- **Re-proposing APTP/Hybrid as the default** without a new kernel result: it is measured neutral to negative here.

## 7. Unverified, long bets

Nothing about GPU, mixed precision or BLR survived verification; the claims were dropped before checking, for budget. Our own harness says:
- cuDSS is accurate on only 4/21 hard_indef and 6-9/13 small_kkt cases.
- The RTX 5080's FP64 rate is weak.

Accuracy is therefore the differentiator a GPU path would have to win on. An FP32 factor, FP64 GMRES-IR and the static-schedule mode is the plausible shape. Revisit after levers 1-3. It is the open question with the most upside and no evidence yet.

## Suggested order

0. ~~**Fix the SSIDS harness env and re-baseline** at 1/8/16 threads.~~ Done 2026-10-30 (section 1).
1. **Cost-model amalgamation**, plus in-place last-child assembly. Single-thread A/B on the tuning split, then the held-out split.
2. **Two-tier matching**, and running the analysis candidates concurrently.
3. **Cross-front tile DAG** for the top of the tree, gated at 8/16 threads against corrected MA97 and SSIDS.
4. **GPU / mixed-precision spike** on the existing KKT replays.

## Sources (verified)

- Duff, Hogg & Lopez, SISC 42(2):C23-C42, 2020; Hogg RAL-TR-2016-017. <https://epubs.stfc.ac.uk/work/46753068>
- Duff & Pralet, RAL-TR-2004-020 / SIMAX 27(2) 2005. <https://numerical.rl.ac.uk/media/reports/duprRAL2004020.pdf>
- Schenk & Gärtner, ETNA 23:158-179, 2006. <https://etna.ricam.oeaw.ac.at/vol.23.2006/pp158-179.dir/pp158-179.pdf>
- Hogg & Scott, NLA 22(4):648-663, 2015. <https://epubs.stfc.ac.uk/work/11539865>
- Scott, RAL-TR-2009-007. <https://epubs.stfc.ac.uk/manifestation/3472/sRAL2009007.pdf>
- Chang, Buluç & Demmel, ParAMD, arXiv 2504.17097. <https://arxiv.org/abs/2504.17097>
- Tang et al., arXiv 2511.10180 (unreviewed). <https://arxiv.org/abs/2511.10180>

Refuted in verification; do not cite: Hogg-Scott 2013 TOMS on matching ordering at u = 1e-8 and on static-pivoting inertia; "SBK is 2x faster than MA57 on IPOPT" in general; the Duff-Pralet MC64SYM failure counts.
