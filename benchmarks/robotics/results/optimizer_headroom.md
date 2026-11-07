# Optimizer headroom probe (2026-11-07)

How much could a custom interior-point optimizer built around smf save against IPOPT + smf on the
robotics suite? This is a measurement only (no solver change). It gives an **upper bound** per cell.

## Setup

- i9-12900K, 1 thread, P-cores, below-normal priority. smf plugin built from main's solver tree
  (0a7e8d6), `SMF_SLACK_ELIM=auto`. CasADi 3.8.2 functions interpreted. The arm was also run once
  with `ROBO_STAGE_JIT=1`.
- The five problems at N = 100 and 200, seeds 0-2 with smf, plus MA27 at seed 0 for reference.
  118 solves; all returned `Solve_Succeeded` and were feasible (violation < 1e-8).
- Script: [`probe_warmstart.py`](../probe_warmstart.py) (`suite` runs the cells, `summary` prints
  the tables below). Records: `optimizer_headroom_smf.jsonl`, `_ma27.jsonl`, `_armjit.jsonl`.
- Each cell runs three solves of the same NLP:
  - **base:** the problem as built, solved cold.
  - **replan cold:** the initial state is perturbed and the problem is solved cold. Every pinned
    initial-state component moves by u × 1.5% of its range over the base trajectory, with
    u ~ U[-1, 1] seeded per seed.
  - **replan warm:** the same perturbed problem, started from base's primal and dual solution.
    Options: `warm_start_init_point=yes`, `mu_init=1e-4`, and every warm-start bound push/frac
    set to 1e-6.
- **car_track** is a periodic flying lap with no initial state, so it gets the base solve only.

## Definitions

- **eval:** CasADi function evaluation, summed over f, g, grad_f, jac_g and hess_l.
- **lin:** IPOPT's linear-solver time: symbolic factorization + numeric factorization + back
  solves.
- **other:** wall − eval − lin. This is IPOPT's own work plus the CasADi interface.
- **f/it:** plugin factorizations per IPOPT iteration (iteration 0 and restoration iterations
  included).
- **corr:** factorizations that directly follow another factorization, i.e. inertia-correction
  retries.
- **Ideal optimizer:** keeps eval as it is, has no `other`, and does one factorization per
  iteration. Its time is eval + symbolic + back solves + numeric factorization × (iterations ÷
  factorizations).
  - The factorization saving is priced by count, so it is slightly generous.
  - On replans, the ideal optimizer also gets the warm-start iteration count: the warm run's count,
    or the cold run's if that is lower.
- **One-shot bound:** base wall ÷ ideal.
- **Replan bound:** IPOPT replan wall (cold, then warm) ÷ ideal replan.

## Results (medians over seeds; smf, interpreted CasADi)

| problem | N | eval | lin | other | f/it | regularized it | corr | iters | replan iters cold / warm | replan wall cold / warm (s) | **one-shot bound** | **replan bound** vs IPOPT cold / vs IPOPT warm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| quad_forest | 100 | 64% | 23% | 13% | 2.01 | 80 | 117 | 100 | 136 / 9 | 0.781 / 0.050 | **1.28x** | 20.4x / **1.16x** |
| quad_forest | 200 | 66% | 23% | 10% | 1.97 | 239 | 362 | 421 | 343 / 11 | 3.420 / 0.111 | **1.24x** | 17.0x / **1.18x** |
| drone_swap | 100 | 42% | 51% | 7% | 1.60 | 14 | 18 | 35 | 36 / 29 | 0.805 / 0.640 | **1.31x** | 1.6x / **1.22x** |
| drone_swap | 200 | 58% | 34% | 9% | 1.23 | 6 | 8 | 35 | 39 / 30 | 1.296 / 0.980 | **1.16x** | 1.4x / **1.10x** |
| car_track | 100 | 24% | 62% | 13% | 1.14 | 4 | 5 | 45 | – | – | **1.25x** | – |
| car_track | 200 | 25% | 62% | 13% | 1.31 | 12 | 15 | 52 | – | – | **1.34x** | – |
| truck_trailer | 100 | 26% | 36% | 37% | 1.74 | 242 | 364 | 463 | 464 / 14 | 2.641 / 0.090 | **1.86x** | 47.6x / **1.63x** |
| truck_trailer | 200 | 25% | 37% | 38% | 1.72 | 329 | 496 | 688 | 663 / 18 | 9.100 / 0.247 | **1.96x** | 63.2x / **1.60x** |
| arm7 | 100 | 96% | 2% | 1% | 1.48 | 28 | 38 | 80 | 77 / 63 | 6.118 / 4.791 | **1.02x** | 1.3x / **1.02x** |
| arm7 | 200 | 96% | 2% | 1% | 1.35 | 22 | 29 | 84 | 228 / 68 | 36.428 / 10.842 | **1.02x** | 1.8x / **1.02x** |
| arm7, stage JIT (seed 0) | 100 | 80% | 14% | 6% | 1.66 | 79 | 116 | 177 | 189 / 159 | 3.121 / 2.643 | **1.12x** | 1.3x / **1.13x** |
| arm7, stage JIT (seed 0) | 200 | 80% | 14% | 6% | 1.80 | 130 | 194 | 244 | 239 / 226 | 8.709 / 8.202 | **1.13x** | 1.2x / **1.14x** |

Where each saving comes from (base solve, smf):

| problem | N | ms per iteration: wall / eval / lin / other | bound from removing overhead only | bound from 1 factorization per iteration only | eval split: g / jac_g / hess_l |
|---|---:|---|---:|---:|---|
| quad_forest | 100 | 5.42 / 3.45 / 1.27 / 0.68 | 1.14x | 1.10x | 7% / 23% / 69% |
| quad_forest | 200 | 10.08 / 6.68 / 2.37 / 1.04 | 1.12x | 1.10x | 6% / 23% / 70% |
| drone_swap | 100 | 23.50 / 9.88 / 12.03 / 1.63 | 1.07x | 1.21x | 6% / 22% / 72% |
| drone_swap | 200 | 34.95 / 20.06 / 11.73 / 3.11 | 1.10x | 1.05x | 7% / 21% / 72% |
| car_track | 100 | 5.67 / 1.38 / 3.52 / 0.72 | 1.15x | 1.07x | 14% / 28% / 52% |
| car_track | 200 | 12.31 / 2.82 / 7.63 / 1.45 | 1.15x | 1.14x | 13% / 27% / 53% |
| truck_trailer | 100 | 6.65 / 1.75 / 2.42 / 2.49 | 1.60x | 1.14x | 16% / 31% / 52% |
| truck_trailer | 200 | 14.05 / 3.54 / 5.20 / 5.28 | 1.60x | 1.11x | 18% / 30% / 51% |
| arm7 | 200 | 159.83 / 153.79 / 3.74 / 2.31 | 1.01x | 1.00x | 2% / 19% / 80% |
| arm7, stage JIT | 200 | 35.84 / 28.61 / 4.96 / 2.27 | 1.07x | 1.05x | 2% / 44% / 53% |

MA27 for reference (seed 0; IPOPT gives no factorization count):

| problem | N | eval | lin | other | iters | replan iters cold / warm |
|---|---:|---:|---:|---:|---:|---|
| quad_forest | 100 / 200 | 54% / 57% | 37% / 35% | 10% / 8% | 115 / 105 | 157 / 9, 160 / 11 |
| drone_swap | 100 / 200 | 51% / 50% | 43% / 45% | 6% / 5% | 27 / 35 | 28 / 24, 36 / 35 |
| car_track | 100 / 200 | 22% / 23% | 65% / 66% | 12% / 12% | 29 / 29 | – |
| truck_trailer | 100 / 200 | 21% / 21% | 52% / 55% | 27% / 24% | 521 / 639 | 478 / 14, 648 / 18 |
| arm7 | 100 / 200 | 94% / 94% | 5% / 5% | 1% / 1% | 175 / 254 | 191 / 151, 226 / 211 |

### Sensitivity to compiled functions

Deployments compile their functions, and the shares above use interpreted CasADi, which inflates
eval. On the arm, compiling cut eval per iteration 5.4x (153.8 → 28.6 ms at N = 200). If eval were
5x cheaper on every problem, with all other measured times unchanged, the bounds would be:

| problem | N | one-shot bound | replan bound vs IPOPT warm |
|---|---:|---:|---:|
| quad_forest | 100 / 200 | 1.81x / 1.70x | 1.37x / 1.45x |
| drone_swap | 100 / 200 | 1.56x / 1.35x | 1.39x / 1.22x |
| car_track | 100 / 200 | 1.33x / 1.46x | – |
| truck_trailer | 100 / 200 | 2.24x / 2.65x | 1.95x / 1.86x |
| arm7 | 100 / 200 | 1.09x / 1.09x | 1.09x / 1.10x |

## Observations

- **IPOPT already gets the warm-start win.**
  - On quad and truck, a warm replan takes 9-18 iterations against 136-663 cold, which is 17-63x
    faster on wall time.
  - Once IPOPT is warm-started too, the ideal optimizer is only 1.02-1.63x ahead (1.1-1.95x with
    compiled functions).
  - drone_swap and the arm gain little from warm start: 36 → 29 and 77-228 → 63-68 iterations.
    The arm's seed 0 needs 215-240 iterations either way.
- **Inertia correction is the biggest factorization waste.**
  - smf needs 1.2-2.0 factorizations per iteration. quad and truck refactor once on most
    iterations (corr ≈ regularized iterations × 1.5).
  - Removing every retry alone is worth only 1.05-1.21x, because a factorization is only part of
    `lin`.
- **IPOPT's own overhead matters only on truck_trailer.** There `other` is 37-38% of the time,
  about as large as the linear solver (2.5 ms per iteration at N = 100). Everywhere else it is
  1-13%.
- **Different local optima.** On quad_forest, the warm and cold replans sometimes converge to
  different local optima. That is expected in a non-convex forest; the largest objective gap was
  2.1 on seed 2, N = 100.
- **Iteration counts depend on the seed, not on smf.**
  - At seed 0, smf takes the same number of iterations as MA27: car_track 28 / 28 at both N, and
    quad N = 200 104 / 104.
  - Seeds 1-2 are harder instances (quad N = 200 seed 2: 690 iterations; car N = 200 seed 2: 86).
    That is why the smf medians sit above MA27's seed-0 column.
  - The worst instances run hundreds of iterations, so a better globalization (fewer iterations)
    could be a bigger lever than anything in the bound above. That is an algorithmic question this
    probe does not measure.

## Verdict

A custom optimizer is **not worth building now**. With the interpreted functions measured here, the
upper bound reaches 1.5x on only one problem of five: truck_trailer, at 1.86-1.96x one-shot and
1.6x against a warm-started IPOPT. The others are at 1.02-1.34x. With compiled functions, the bound
reaches 1.5x+ on three of five (quad, truck, drone at N = 100), but these are ceilings that assume
zero overhead and perfect inertia guesses. The large replan gains come from warm starting, which
IPOPT already does.

Cheaper levers come first:
- warm-start IPOPT in the MPC loop;
- cut inertia-correction refactorizations and truck_trailer's IPOPT overhead inside IPOPT + smf.

Reconsider a custom optimizer only if its algorithm, not just its overhead, would cut the hundreds
of iterations on the hard seeds.
