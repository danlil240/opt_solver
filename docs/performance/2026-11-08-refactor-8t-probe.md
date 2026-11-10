# 8-thread refactorization over a whole IPOPT run: probe

**Date:** 2026-11-08 · **Build:** main `0a7e8d6`, threaded MKL (`build-mtt`, cuDSS off) · **Machine:** i9-12900K,
P-cores, 3 trials + 1 warmup · **Data:** `benchmarks/results/remote/probe_refactor8t/` (local only) ·
**Context:** `reports/Paths to decisive solver leadership.md`, bet 2.

## Question

The front-tier board counts 5 refactorizations per case. A real IPOPT run analyses once and then factorizes many
times. Does smf stay ahead of MKL PARDISO, MA97 and MA57 over a real run's factorization count, at 1 and 8 threads?
If not, do the existing switches (re-analysis from the realized order, a forced matching ordering) close the gap?

## Method

- Cases: the 8 ipopt-track KKTs where the 2026-11-05 board's refactorization at 8 threads was slower than PARDISO's.
  Manifest `benchmarks/manifests/probe_refactor8t.json` (the 8 cases' ipopt entries from `fronttier.json`).
- New `run.py --seq-steps all` replays every stored factorization. The two PGLib cases store their whole run
  (46 and 75 factorizations). The six `ipopt_kkt` cases store only 5 evenly spaced steps of 12-136.
- Projected linear-algebra time of the run = analyse + replayed factorizations + (N - replayed) x median
  refactorization, with N the case's IPOPT factorization count. For PGLib this is a measurement, not a projection.
- Variants: smf on the board's ipopt track; smf as the IPOPT plugin runs it (`--track-override ipopt-plugin`);
  the plugin with `SMF_IPM_REANALYSE=1`; smf with `--ordering matching`. Rivals on the ipopt track.

## Result

smf / fastest accurate rival, projected run time (below 1 = smf faster):

| | 1 thread | 8 threads |
|---|---:|---:|
| smf (board settings) | **0.757** | **1.212** |
| smf as the IPOPT plugin | 0.776 | 1.279 |
| plugin + re-analysis | 0.820 | 1.325 |
| forced matching ordering | 0.837 | 1.227 |

At 1 thread smf wins every case (0.50-0.89). At 8 threads it loses 6 of 8 to PARDISO:

| case | N | smf 8T ms | PARDISO 8T ms | ratio | smf refactor 1T -> 8T (speedup) | PARDISO refactor 1T -> 8T (speedup) | smf delays per factorization |
|---|---:|---:|---:|---:|---|---|---:|
| pglib_opf_case2000_goc__sad | 75 | 1005 | 603 | 1.67 | 12.6 -> 13.0 (0.97x) | 23.0 -> 4.8 (4.80x) | 20,844 |
| acopf_grid_80 | 25 | 243 | 163 | 1.49 | 19.2 -> 9.0 (2.13x) | 23.0 -> 5.3 (4.36x) | 969 |
| acopf_grid_160 | 24 | 1153 | 787 | 1.46 | 139.4 -> 44.7 (3.12x) | 159.6 -> 27.3 (5.84x) | 3,994 |
| obstacle_surface_200 | 136 | 814 | 693 | 1.17 | 13.6 -> 5.9 (2.30x) | 20.7 -> 4.8 (4.31x) | 0 |
| pde_control_120 | 13 | 93 | 84 | 1.10 | 11.6 -> 6.0 (1.92x) | 15.6 -> 4.2 (3.72x) | 0 |
| pde_control_250 | 12 | 462 | 434 | 1.07 | 89.1 -> 32.8 (2.72x) | 93.2 -> 21.4 (4.35x) | 0 |
| pglib_opf_case1354_pegase | 46 | 286 | 298 | 0.96 | 5.8 -> 5.9 (0.99x) | 10.9 -> 4.1 (2.66x) | 13,932 |
| portfolio_20k | 17 | 133 | 138 | 0.96 | 6.8 -> 7.0 (0.97x) | 5.5 -> 4.7 (1.17x) | 0 |

- All smf runs are accurate. MA97 and MA57 are slower at 8 threads than at 1 on most of these cases; PARDISO is the
  only rival that scales.
- **The gap is thread scaling, not single-thread speed.** At 1 thread smf's refactorization is faster than PARDISO's
  on 7 of 8 cases. At 8 threads smf reaches 1.9-3.1x on delay-free cases and 0.97-0.99x on PGLib; PARDISO 3.7-5.8x.
- **PGLib does not scale at all.** Its refactorizations are tiny (6.8 MFlop in 5.8 ms on case1354, about 1.2 GFlop/s:
  memory traffic and per-front overhead). *Corrected 2026-11-09:* the cause is not its 14-21k delays. PGLib runs on one
  thread at 8T because `kTreeParallelMinWork = 2e7` is a hard floor and the planner prices flops only. Forced to 8
  threads the existing engine gets 2.2x with identical inertia (front trace 2026-11-09; see
  `docs/superpowers/specs/2026-11-09-refactor-8t-engine-design.md`).
- **Re-analysis does nothing here.** `SMF_IPM_REANALYSE=1` leaves the delays unchanged (13,932 / 20,844) and adds
  its own cost (PGLib 8T ratios 0.96 -> 1.22 and 1.67 -> 2.02).
- **The matching ordering removes most PGLib delays** (20,844 -> 2,001; 13,932 -> 1,972) and gives case2000 its
  first thread speedup (13.0 -> 7.3 ms at 8T, 1.78x). It costs 95-150 ms of analysis instead of 13-20 ms, so it
  helps only case2000 at 8T (1.67 -> 1.18) and hurts both at 1T. Even delay-free, case2000 stays 1.5x behind
  PARDISO's refactorization at 8T.

## Verdict

Go for bet 2, but as engineering, not a switch. No existing option closes the 8-thread gap. What a parallel
refactorization engine must do, from these numbers:
1. Parallelise the many small fronts of a sparse KKT tree. Delay-free cases reach only 1.9-3.1x on 8 threads where
   PARDISO reaches 3.7-5.8x; the earlier factor profile (2026-11-04) put serial setup and gather at 22-25%.
2. Price the planner in time, not flops. *Corrected 2026-11-09:* PGLib's delays do not serialise it; the planner's
   work floor does. With the floor removed it gets 2.2x on 8 threads.
3. Use the run length. With 12-136 factorizations ahead, schedule, scatter maps and buffers can be built once per run.

Re-analysis from the realized order (`SMF_IPM_REANALYSE`) is dead for this purpose.

## Caveats

- Six of the eight cases replay 5 stored steps; their N-factorization time is a projection from the median
  refactorization. The two PGLib runs are complete.
- Real IPOPT sessions add solves, refinement and inertia retries; this measures factorization only.
- One machine (12900K, P-cores). Rival records are fresh from the same run.
