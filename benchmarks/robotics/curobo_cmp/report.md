# IPOPT + smf against NVIDIA cuRobo on the arm (2026-11-07)

## Bottom line

- **Same obstacle, same robot geometry, same limits, same PC:** cuRobo plans in 0.49 s median (GPU), IPOPT + smf in 0.66 s median (1 CPU thread, interpreted CasADi). IPOPT's motions are 7% faster to execute (1.29 s vs 1.39 s median) and never take a detour; cuRobo's took detours of 1.7-2.8 s on 3 of 10 scenes.
- **IPOPT holds the collision constraints, cuRobo approximates them.** IPOPT's worst penetration between knots is 0.7 mm. cuRobo's is 10 mm into the true sphere: its planner cannot load a sphere, so the sphere goes in as a triangle mesh, and its collision term is a penalty.
- **cuRobo's fast path is boxes.** With the sphere replaced by its bounding cube (a bigger obstacle, but cuRobo's native primitive), cuRobo plans in 28 ms median. Mesh obstacles cost it about 17x.
- **This PC runs cuRobo about 7x slower than NVIDIA's published numbers.** cuRobo's own 2,590-problem Franka benchmark gives 32 ms mean here (no payload) and 37 ms (3 kg). NVIDIA reports 4.5 and 4.9 ms on an RTX 5090 under native Linux. Here it is an RTX 5080 under WSL2. Scaling by that factor (an estimate, not measured) puts cuRobo's cube case near 4 ms and its mesh case near 70 ms on NVIDIA's setup.
- **The full arm7 problem** (torque inputs, RK4, N = 100, its own point collision model) takes 1.7-5.6 s. It solves a different problem: a dynamics-exact, torque-limited minimum-time motion (1.18 s), not a kinematic spline.

## What was compared

**Scenes:** the arm7 scenes for seeds 0-9 (`arm7_model.scene`):
- the Panda's start and goal configurations;
- a sphere of radius 0.18-0.22 m at (0.45, 0, 0.45);
- a table.

Both planners get the same start and goal joint states (cuRobo's `plan_cspace`).

**cuRobo:**
- Version: NVlabs/curobo main 5e4f23d (2026-11-06), torch 2.14 + CUDA 13.0, warp 1.19.
- Hardware: RTX 5080, WSL2 Ubuntu on the Ryzen 9950X3D PC.
- Settings: default `MotionPlannerCfg` (franka.yml, 4 trajectory seeds, L-BFGS on a 16-knot cubic B-spline, CUDA graphs, time optimization, graph planner from the second attempt).
- Collision: cuRobo's 61 robot spheres against the scene.
- Timing: wall clock with CUDA synchronization, after warmup, median of 7 repeats per scene.

**IPOPT:**
- Platform: Windows on the same PC, 1 thread, CasADi 3.8.2.
- Solver: smf through the MA97 plugin (build of 2026-11-05), with MA27 and MUMPS for reference.
- Formulation (`ipopt_arm7.py --form kin`), sized like cuRobo's:
  - the joint path is a piecewise cubic (jerk input) with 32 intervals;
  - limits are cuRobo's franka.yml limits: velocity, acceleration 15 rad/s², jerk 500 rad/s³;
  - torque is limited through the arm7 point-mass model;
  - the objective is minimum time.
- Collision:
  - `--geom curobo`: cuRobo's own 61 collision spheres, fitted exactly to the DH frames (`dump_spheres.py`, error under 1e-6 m), against the exact sphere and the table, at every knot.
  - `--geom arm7`: the arm7 problem's link points with a 6 cm margin.
- Cold start from a straight line.

**Checks:** both planners' trajectories are re-checked densely:
- cuRobo's: its 5 ms interpolation;
- IPOPT's: 10 samples per interval.

They are checked against the exact sphere with cuRobo's robot spheres (`sphere clearance`), and against the arm7 constraints.

## Results

Ten scenes each. Times are per plan; durations are the planned motion's length.

| configuration | success | median ms | range ms | median duration s | duration range s | worst sphere clearance mm |
|---|---|---|---|---|---|---|
| cuRobo, sphere as mesh | 10/10 | 487 | 124-658 | 1.39 | 1.31-2.76 | -10.0 |
| cuRobo, sphere as bounding cube | 10/10 | 28 | 24-184 | 1.42 | 1.36-1.61 | 20.3 |
| IPOPT+smf, cuRobo spheres | 10/10 | 657 | 495-937 | 1.29 | 1.22-1.31 | -0.7 |
| IPOPT+MA27, cuRobo spheres | 10/10 | 640 | 516-909 | 1.29 | 1.22-1.31 | -0.7 |
| IPOPT+MUMPS, cuRobo spheres | 10/10 | 925 | 712-1325 | 1.29 | 1.22-1.31 | -0.7 |
| IPOPT+smf, arm7 points | 10/10 | 188 | 146-201 | 1.29 | 1.22-1.31 | -24.4 |
| IPOPT+smf, arm7 points, compiled | 10/10 | 98 | 67-128 | 1.29 | 1.22-1.31 | -24.4 |
| IPOPT+MA27, arm7 points | 10/10 | 245 | 189-268 | 1.29 | 1.22-1.31 | -24.4 |
| IPOPT+MUMPS, arm7 points | 10/10 | 324 | 234-365 | 1.29 | 1.22-1.31 | -24.4 |
| full arm7 (torque, RK4, N=100), stage-compiled | 5/5 | 2307 | 1701-5590 | 1.18 | 1.12-1.20 | -69.3 |

**Reading the table:**
- **Cube clearance:** the cube row's positive clearance is against the true sphere; the cube encloses it.
- **Arm7-point rows:** they penetrate cuRobo's spheres by up to 2.4 cm. The arm7 point model with a 6 cm margin is thinner than the Panda's real links, so its timings are not a same-geometry comparison.
- **Compiled row:**
  - Seeds 0-4 ran alone: 67-87 ms.
  - Seeds 5-9 overlapped GPU runs.
  - Compiling the whole NLP takes about 4.5 minutes per scene.
- **MA27 and MUMPS rows (arm7 points):** they also overlapped a GPU run.

**Where IPOPT's time goes** (cuRobo spheres, smf, interpreted, medians):
- 97 iterations;
- KKT order 9,078;
- linear solves 0.23 s (35%);
- Hessian 0.25 s;
- constraint Jacobian 0.09 s.

Compiling the functions removes most of the Hessian and Jacobian time. The linear solve would then be over half: 2.3 ms per iteration on a 9k-row KKT, the same as MA27.

**cuRobo's mesh cost:**
- Its first attempt (linear seed, 50 + 25 L-BFGS steps) takes about 100-120 ms with the mesh and 10-20 ms with the cube.
- On about half the mesh scenes the first attempt fails and the graph planner seeds a second.

**cuRobo's own benchmark on this PC** (`benchmark/motion_plan_benchmark.py`, MotionBenchMaker + MPiNets, 2,590 problems; NVIDIA's published numbers in brackets):

| | success | planning mean | planning median | motion time mean |
|---|---|---|---|---|
| no payload, no dynamics | 99.73% | 31.9 ms [4.46 ms] | 30.8 ms | 1.36 s [1.18 s] |
| 3 kg, dynamics on | 99.73% | 37.1 ms [4.87 ms] | 34.8 ms | 1.47 s [1.27 s] |

## Formulation differences that remain

- **Constraints:**
  - cuRobo: collision, joint limits and torque are penalty costs; success is a post-check with a 2.5 mm activation distance.
  - IPOPT: everything is a hard constraint (1e-8 at the knots).
- **Self-collision:** cuRobo checks it; the IPOPT formulation does not (no motion here comes near it).
- **Torque model:**
  - cuRobo: the Panda's real inertias.
  - IPOPT: the arm7 point-mass model.
  - Under the arm7 model, cuRobo's trajectories reach up to 1.26x the torque limits. Neither model is checked against the other's.
- **Seeds:** cuRobo optimizes 4 seeds in parallel and runs IK and graph search when needed. IPOPT runs one local solve from a straight line, which happens to be a good start in these scenes. Harder scenes would need a seeding strategy.

## Files

- `curobo_arm7.py`: cuRobo runs and the GPU re-check of saved IPOPT paths.
- `ipopt_arm7.py`: IPOPT runs.
- `arm7_model.py`: scenes and checks.
- `dump_spheres.py`, `curobo_spheres.json`: cuRobo's spheres in the DH frames.
- `summarize.py`: the table above.
- `results/`: raw records, and cuRobo's benchmark KPIs (`mb_*.yml`).
