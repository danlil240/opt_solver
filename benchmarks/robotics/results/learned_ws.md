# Learned warm-start probe (2026-11-07)

Can a guess predicted from the task cut IPOPT's iterations on a new instance, and does a small
neural net beat reusing the nearest stored solution? Measurement only, no solver change.

## Setup

- Script: [`probe_learned_ws.py`](../probe_learned_ws.py). Fixed environment (obstacle seed 0), N = 100,
  240 training tasks and 60 held-out tasks per problem, drawn uniformly:
  - truck_trailer: start pose (x in [-28, 22], y in [-1, 8], heading in [-60, 60] deg), at rest;
    it must back into the same bay. Starts closer than 0.5 m to an obstacle are redrawn.
  - quad_forest: start and goal (y in [-3.5, 3.5], z in [0.8, 3]) across the same forest.
- Cloud box (4 cores, 4 workers in parallel), CasADi 3.8.2 interpreted, IPOPT with **MUMPS**.
  Iteration counts don't depend on the linear solver much; wall times are MUMPS times and are
  only compared within this run.
- Cold: the problem's own style of hand-made guess, default IPOPT options.
- Library (kNN): the successful training solutions (primal and duals); nearest by normalized task.
- MLP: task -> primal, 2 x 256 tanh, numpy, Adam, 4000 full-batch epochs (30-60 s to train,
  0.13-0.16 ms to evaluate).
- `primal` methods pass only x0 with default options. `warm` methods pass x0 plus the nearest
  library solution's duals with `warm_start_init_point`, `mu_init` 1e-4 and bound pushes 1e-6
  (the same options as the optimizer-headroom replan probe).

## Results (60 held-out tasks each)

| problem | start | solved | median iters | mean iters | median wall s | total wall speedup vs cold |
|---|---|---:|---:|---:|---:|---:|
| quad | cold | 60/60 | 140 | 155 | 1.46 | 1.00x |
| quad | kNN primal | 60/60 | 110 | 139 | 1.25 | 1.03x |
| quad | **kNN warm** | 60/60 | **14** | 20 | **0.16** | **7.44x** |
| quad | MLP primal | 60/60 | 145 | 196 | 1.70 | 0.58x |
| quad | MLP warm | 60/60 | 60 | 86 | 0.70 | 1.60x |
| truck | cold | 59/60 | 504 | 606 | 37.4 | 1.00x |
| truck | kNN primal | 60/60 | 576 | 578 | 45.7 | 0.98x |
| truck | **kNN warm** | 60/60 | **51** | 173 | **2.73** | **4.42x** |
| truck | MLP primal | 58/60 | 532 | 628 | 36.3 | 1.01x |
| truck | MLP warm | 58/60 | 229 | 323 | 13.2 | 2.29x |

Total wall speedup = sum of cold wall / sum of method wall over tasks both solved.

Local optima (objective vs the cold solve, tasks both solved):

| problem | start | better by >1% | worse by >1% | worst |
|---|---|---:|---:|---:|
| quad | kNN warm | 22 | 21 | +122% |
| quad | MLP warm | 17 | 33 | +61% |
| truck | kNN warm | 3 | 3 | +226% |
| truck | MLP warm | 2 | 3 | +222% |

## Findings

- A primal guess alone does nothing: with IPOPT's default barrier start, the iteration count is
  the same as cold, for both kNN and the MLP. The gain comes from starting the barrier small with
  the duals.
- The plain library lookup with duals gives 8-10x fewer iterations at the median and 4.4-7.4x
  total wall time. It passes the probe's 3x gate.
- The MLP loses to the lookup on both problems, so at this task dimension (3-4 numbers) a neural
  net isn't needed: a library of ~240 solves is enough.
- The truck's mean (173 iterations) hides a tail: a few tasks still take hundreds of iterations
  from their neighbour's start. On the quad, the neighbour often goes around a tree on the other
  side, so the answer is a different local optimum, better as often as worse.

## Follow-up (2026-11-08): nearest-3 in parallel, and moving obstacles

`probe_learned_ws.py neighbors <dir> --k 3` warm-starts each held-out task from its 2nd and 3rd
nearest library solutions too. **Race** = the K solves start together and the first success wins
(on K cores). Total speedup is over the 60 held-out tasks, against the cold solve.

Two new task families, both quadrotor, N = 100, with the obstacles as NLP parameters so one build
serves every task:
- `quadmov`: the seed-0 forest plus 4 movable obstacles (r 0.5) anywhere in the field, task = start/goal + 8 numbers.
- `quadrand`: a fresh 24-tree layout per task. The lookup and the MLP see start/goal plus a blurred
  31 x 10 occupancy grid.

| problem | cold median / p90 s | kNN-1 total | race of 3 nearest: total | median s | p90 s | max s | MLP + 2 nearest race |
|---|---|---:|---:|---:|---:|---:|---:|
| quad (fixed forest) | 1.46 / 2.39 | 7.44x | **12.8x** | 0.12 | 0.19 | 0.36 | 11.2x |
| truck (fixed yard) | 37.4 / 62.2 | 4.42x | **14.8x** | 1.26 | 7.21 | 17.9 | 12.0x |
| quad + 4 movable obstacles | 1.48 / 3.09 | 1.39x | **3.41x** | 0.42 | 1.11 | 2.97 | 2.72x |
| quad, random forest per task | 1.37 / 2.52 | 1.41x | 2.68x | 0.52 | 1.05 | 1.71 | **2.84x** |

Every held-out task was solved by at least one of the three starts in every family.

Findings:
- Racing the 3 nearest solutions roughly doubles the single-lookup gain and cuts the tail: the
  truck's p90 drops from 37 s (one neighbour) to 7 s, against 62 s cold.
- Waiting for all three and keeping the lowest objective (best-of-3) finds a better local optimum
  than cold far more often than a worse one (quad 31 better / 8 worse, truck 3 / 0), but costs
  the slowest of the three, so it's 0.8-3.2x total only.
- Moving obstacles hurt the lookup a lot: one neighbour gives 1.4x, three give 2.7-3.4x. The
  neighbour's duals belong to obstacles that moved.
- With a random forest, the MLP warm start alone (1.80x) beats one neighbour (1.41x), and an
  MLP + 2-neighbour race is the best of all (2.84x). This is the first case where the network earns
  its place: high-dimensional tasks.
