"""Solve one problem instance with one linear solver; print the record as JSON.

  python run_one.py <problem module> <N> <seed> <solver> [threads]

Each solve runs in its own process (run_suite.py spawns them), so IPOPT's
library state and the plugin's statistics belong to that one solve.
"""

from __future__ import annotations

import importlib
import json
import sys

import common


def main() -> None:
    mod, N, seed, solver = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    threads = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    common.ensure_thread_env(threads)
    prob = importlib.import_module(mod).build(N, seed)
    rec = common.solve(prob, solver, threads)
    print("RESULT " + json.dumps(rec, default=float))


if __name__ == "__main__":
    main()
