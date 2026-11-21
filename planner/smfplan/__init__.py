"""smfplan: a trajectory planner with a memory of past solutions.

A planning problem is built once (CasADi NLP, solved by IPOPT with any linear
solver, smf by default in the examples). Every solved task goes into a
library. A new task is started from its K nearest stored solutions (primal
and duals) in parallel on a pool of warm solver processes; the first to
converge wins and the others are stopped.

  from smfplan import Planner
  with Planner(spec_factory, workers=3) as pl:
      res = pl.plan(task)

See problem.py for what a spec provides.
"""

from .library import SolutionLibrary
from .planner import PlanResult, Planner
from .pool import SolverPool

__all__ = ["Planner", "PlanResult", "SolutionLibrary", "SolverPool"]
