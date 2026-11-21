# smfplan

A trajectory planner with a memory of past solutions, on IPOPT + CasADi with
smf (or any IPOPT linear solver) as the engine.

- **Library.** Every solved task is stored with its primal and dual solution.
- **Race.** A new task starts from its 3 nearest stored solutions at once, on a
  pool of solver processes that stay alive between plans (the NLP and its
  CasADi functions are built once per worker). The first start to converge
  wins; IPOPT's iteration callback stops the others at their next iteration.
- **Warm start.** A stored start passes its duals with
  `warm_start_init_point`, `mu_init` 1e-4 and tight bound pushes. A primal
  guess alone doesn't help (see
  [learned_ws.md](../benchmarks/robotics/results/learned_ws.md)).
- **Replanning.** `plan(task, hint=spec.shift_hint(prev.as_hint(), frac))`
  races the previous plan, shifted by the executed fraction of its horizon
  (trajectory and duals resampled onto the N nodes), alongside the library.

Results (cloud box, MUMPS, N=100, 60 seeded tasks; [results/](results)):

| truck | cold median / p90 | planner median / p90 | total speedup |
|---|---|---|---:|
| 20 new start poses | 28.0 / 39.4 s | 2.0 / 9.9 s | 7.6x |
| 24 replans while moving, previous plan shifted | 35.1 / 121 s | 1.1 / 18.2 s | 10.5x |
| 24 replans while moving, not shifted | 28.8 / 87 s | 7.7 / 20.2 s | 4.4x |

The planner solved every task; cold missed 2 of the shifted run's replans.

```python
from smfplan import Planner
import specs  # planner/examples

opts, env = specs.solver_setup("smf")
with Planner(specs.truck_factory(100), workers=3, k=3, solver_options=opts, env=env) as pl:
    pl.seed(tasks)            # optional: fill the library with cold solves in parallel
    res = pl.plan(task)       # res.x, res.obj, res.wall_s, res.source ("lib1", "cold", ...)
```

A problem is a *spec* built inside each worker by a picklable factory; see
[smfplan/problem.py](smfplan/problem.py). [examples/specs.py](examples/specs.py)
adapts the robotics suite's truck and quadrotor families.

Tests: `python -m pytest planner/tests -q` (small quadrotor, MUMPS).
Demo: `python planner/examples/truck_replan.py --solver smf`.
