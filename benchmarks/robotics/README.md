# Robotics trajectory-optimization suite

CasADi + IPOPT problems for benchmarking IPOPT's linear solvers (MUMPS, MA27, MA57, MA97 and smf through its MA97-compatible plugin) on realistic robotics KKT systems. Every problem uses an exact Hessian, is parameterised by the horizon `N` and a `seed`, and checks its own solution: obstacles cleared, goal reached, track bounds held.

| module | problem | model | transcription |
|---|---|---|---|
| `p_quad_forest` | quadrotor, minimum time + energy through a seeded forest of 24 cylinders | 12-state rigid body, 4 rotor thrusts (`quadrotor.py`) | multiple shooting, RK4, free final time |
| `p_drone_swap` | 3 quadrotors swap places around a pillar, pairwise collision avoidance | 3 x the 12-state model | multiple shooting, RK4, fixed horizon |
| `p_car_track` | minimum lap time on a winding track with parked obstacles | dynamic bicycle, Pacejka lateral tires (Liniger et al. 2015 1:43 car), Frenet frame, spatial domain | Legendre collocation, degree 3 |
| `p_truck_trailer` | tractor-trailer reversing into a loading bay | kinematic tractor + on-axle trailer, circle-covered bodies, hitch-angle limit | multiple shooting, RK4, free final time |
| `p_arm7` | 7-DoF arm (Panda kinematics) point-to-point around a sphere | point-mass links, gravity from the potential, configuration-dependent diagonal inertia, torque limits | multiple shooting, RK4, free final time |

## Running

```bash
python benchmarks/robotics/run_one.py p_quad_forest 100 0 smf
python benchmarks/robotics/run_suite.py --out benchmarks/results/robotics/suite_t1.jsonl --sizes 50,100,200,400
python benchmarks/robotics/diag_suite.py --out diag.jsonl --cells p_truck_trailer:100 --solvers ma27,smf
```

- **HSL library:** `SMF_HSL_LIBRARY` (full CoinHSL `libhsl.dll`). The default is `build-deps/coinhsl-*/bin/libhsl.dll`.
- **smf plugin:** `SMF_PLUGIN_DLL`. The default is `build-board/Release/smf_ma97.dll`, then `build/Release`. The harness runs it with its opt-in slack-pair elimination on (`SMF_SLACK_ELIM=auto`: where slack pairs are at least a fifth of the rows); set `SMF_SLACK_ELIM=off` in the environment to measure the plugin without it.
- **PATH:** must hold the plugin's DLL directory (MKL and friends).
- **Thread counts:** every runner sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `SMF_NUM_THREADS` before anything loads (`common.thread_env` / `common.ensure_thread_env`): `run_suite.py` and `diag_suite.py` (`--threads`) put them in each child's environment, and `run_one.py`, `diag_one.py` and `stagewise.py` (1 thread) start themselves again in such an environment when the inherited values differ. The mingw HSL and MUMPS read the C runtime's environment, which never sees `os.environ` changes made inside Python.
- **JIT:** `ROBO_JIT=1` compiles CasADi's NLP functions with `zig cc`. It is off by default, because compiling the full derivatives takes minutes at N >= 100.

`run_suite.py` records:
- wall time, iterations, objective and constraint violation;
- CasADi's function-evaluation split;
- IPOPT's timing statistics: symbolic factorization, numerical factorization and back solves;
- the KKT order and nonzeros;
- the problem's own check;
- for smf, the plugin's factor and solve call counts.

`diag_one.py` / `diag_suite.py` add:
- IPOPT's regularisation counts (iterations with `lg(rg)`, delta_c > 0, trial factorizations, restoration);
- for smf, the plugin's per-factorization trace: static pivots, inexact-inertia reports, aborts and zero pivots.

The KKT sequences IPOPT requested can be captured with the plugin (`SMF_KKT_DUMP_DIR`) and replayed with `bench_replay`. `benchmarks/manifests/robotics_kkt.json` is the manifest; the fixtures in `benchmarks/fixtures/robotics/` are not committed.

## Results

`results/` holds the measurements made with the suite (2026-11-06, 1 thread, seed 0 unless noted):

| file | what |
|---|---|
| `report_step4.md` | first measurements and where smf lost (smf main d6fd34a against MUMPS, MA27, MA57, MA97) |
| `summary_t1.md` | that run's per-cell table (N = 50, 100, 200, 400) |
| `suite_t1.jsonl`, `suite_seeds.jsonl` | its raw records: `run_suite.py` output, and the seed sweep |
| `diag_n100.jsonl` | `diag_suite.py` records at N = 100 (regularisation counts, plugin trace); rerun on 2026-11-06 with the thread variables set to 1 in the child (the first run did not set them; its iteration, regularisation and trial-factorization counts were identical). The smf rows are the merged plugin with `SMF_SLACK_ELIM=auto` (the elimination runs on truck_trailer only) |
| `slack_elim_live_first.jsonl` | smf main #56 against the first slack-pair elimination build, all five problems at N = 100 and 200 |
| `stagewise_fatrop.jsonl` | `stagewise.py` records: FATROP against IPOPT (MUMPS, MA27, smf) on stage-ordered copies of quad_forest, drone_swap, car_track and truck_trailer at N = 100 and 200 |
| `slack_elim_live_guard.jsonl` | the same against the merged build (#57, `SMF_SLACK_ELIM=auto`, with its solve accuracy check and inertia guard), truck, quad and arm |

`kkt_nnz` (the lower-triangle nonzeros of IPOPT's KKT matrix) was corrected on 2026-11-06 in every record and in `summary_t1.md`: it had counted the Hessian's diagonal entries twice (once in IPOPT's Hessian count, once in the diagonal). It now matches the captured matrices exactly (the five `robotics_kkt.json` cases).

Records in the `slack_elim_*` files carry `variant` (`main` / `slk`), `factors` (plugin factorizations), `slack_armed` (factorizations through the elimination) and `slack_fallbacks` (pairs failing the guard).

With the merged elimination (`auto`), against smf main #56 on the same run:
- **truck_trailer:** N = 200 15.8 s -> 9.1 s (all 1192 factorizations eliminated); N = 100 21.6 s -> 5.2 s.
- **quad_forest:** N = 100 0.96 s -> 0.60 s; N = 200 1.24 s -> 1.19 s.
- **arm7:** about unchanged (function evaluations dominate).
- **drone_swap, car_track:** too few slack pairs; `auto` leaves them alone.

FATROP (`stagewise_fatrop.jsonl`, `stagewise.py`, 1 thread, seed 0), against IPOPT on the same stage-ordered NLPs (iterations / wall seconds):

| problem | N | FATROP | IPOPT + MA27 | IPOPT + smf |
|---|---:|---|---|---|
| quad_forest | 100 | 138 / 0.80 (failed) | 74 / 0.48 | 74 / 0.45 |
| quad_forest | 200 | 230 / 2.71 | 313 / 4.07 | 357 / 4.15 |
| drone_swap | 100 | 27 / 0.60 | 27 / 0.50 | 27 / 0.41 |
| drone_swap | 200 | 35 / 1.56 | 31 / 1.15 | 32 / 0.97 |
| car_track | 100 | 29 / 0.13 | 29 / 0.15 | 29 / 0.13 |
| car_track | 200 | 27 / 0.24 | 29 / 0.29 | 29 / 0.25 |
| truck_trailer | 100 | 323 / 1.84 | 453 / 2.71 | 429 / 2.04 |
| truck_trailer | 200 | 369 / 4.91 (failed) | 428 / 5.61 | 442 / 4.78 |

Configuration:
- The smf rows are the merged plugin (main be9cec2) with `SMF_SLACK_ELIM=auto`, the default of every runner. The elimination ran on every factorization of quad_forest and truck_trailer, and not at all on drone_swap and car_track.
- The FATROP, MUMPS and MA27 rows are from the first run.
- The drone_swap rows were rerun after a fix to `stagewise.py`. Its control-smoothing term coupled adjacent stages, which FATROP's stage-wise cost does not allow, and both earlier FATROP runs had failed (103 and 285 iterations). The previous control is now carried as a state: the same NLP, with the same optimum.
