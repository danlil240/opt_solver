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
- **Replanning.** `plan(task, hint=prev.as_hint())` races the previous plan
  alongside the library.

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
