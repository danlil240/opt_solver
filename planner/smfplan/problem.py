"""What a planning problem (a "spec") provides.

A spec is built inside each solver process by a picklable factory: a tuple
(module_name, attribute, args, kwargs, sys_paths). The attribute is called
with args and kwargs and returns an object with:

  nlp           {"x": SX/MX, "f": ..., "g": ..., optional "p": ...}
  lbg, ubg      constraint bounds (arrays)
  bounds(task)  -> (lbx, ubx): the task enters through bounds (pinned start
                and goal states) ...
  params(task)  -> p vector or None: ... and/or through NLP parameters
                (obstacle positions)
  cold_guess(task) -> x0 used when no stored solution is available
  features(task)   -> 1-D array; nearness in this space should mean the
                solutions look alike (scale the components yourself)

The task itself is any picklable value (a numpy array in the examples).
"""

from __future__ import annotations

import importlib
import sys
from typing import Any


def build_spec(factory: tuple) -> Any:
    module, attr, args, kwargs, paths = factory
    for p in paths:
        if p not in sys.path:
            sys.path.insert(0, p)
    return getattr(importlib.import_module(module), attr)(*args, **kwargs)
