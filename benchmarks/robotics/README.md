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
- **Thread counts:** `run_suite.py` puts them in each child's environment. The mingw HSL and MUMPS read the C runtime's environment, which never sees `os.environ` changes made inside Python.
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
