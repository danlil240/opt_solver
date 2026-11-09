#!/usr/bin/env python3
"""Fit the thread plan's per-front costs (Control::thread_cost_per_front_entry, thread_cost_per_front).

The plan prices a front as f^2 p + c_asm f^2 + c_front work units (thread_plan.hpp, FrontCostModel). Its time
on one thread is modelled as t = a f^2 p + b f^2 + c, fitted by least squares over the fronts of 1-thread
per-front traces, so c_asm = b / a and c_front = (c - fprintf_s) / a, where fprintf_s is the trace's own cost
per front (one fprintf; measured as traced minus untraced refactorization time over the front count).

Traces: the probe build of branch probe/front-trace-8t with SMF_PROBE_FRONTS=<path> and SMF_PROBE_MINF=0 (every
front), columns `phase f ne p me blas tiled lapack flops ta tk0 tk1 s parent ...`, one `# factor t=` line per
factorization. A front's time is the next front's ta minus its ta (assembly, kernel and the hand-off of its
contribution block); a factorization's last front counts tk1 - ta.

  python tools/bench/fit_front_costs.py --fprintf-us 0.5 <trace files...> [--min-f 1]
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import sys

import numpy as np


def parse(path: str | pathlib.Path) -> list[tuple[int, int, int, float]]:
    out: list[tuple[int, int, int, float]] = []
    block: list[tuple[int, int, int, float, float]] = []

    def close() -> None:
        for i, (f, ne, p, ta, tk1) in enumerate(block):
            t = (block[i + 1][3] - ta) if i + 1 < len(block) else (tk1 - ta)
            out.append((f, ne, p, t))
        block.clear()

    for ln in pathlib.Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if ln.startswith("# factor"):
            close()
            continue
        if not ln or ln.startswith("#"):
            continue
        x = ln.split()
        if len(x) < 12:
            continue
        block.append((int(x[1]), int(x[2]), int(x[3]), float(x[9]), float(x[11])))
    close()
    return out


def fit(fronts: list[tuple[int, int, int, float]], fprintf_s: float) -> tuple[float, float]:
    f = np.array([x[0] for x in fronts], dtype=float)
    p = np.array([x[2] for x in fronts], dtype=float)
    t = np.array([x[3] for x in fronts], dtype=float)
    X = np.column_stack([f * f * p, f * f, np.ones_like(f)])
    (a, b, c), *_ = np.linalg.lstsq(X, t, rcond=None)
    return float(b / a), float((c - fprintf_s) / a)


def r2(fronts: list[tuple[int, int, int, float]], c_asm: float, c_front: float, fprintf_s: float) -> float:
    f = np.array([x[0] for x in fronts], dtype=float)
    p = np.array([x[2] for x in fronts], dtype=float)
    t = np.array([x[3] for x in fronts], dtype=float)
    w = f * f * p + c_asm * f * f + c_front
    a = float(np.dot(w, t - fprintf_s) / np.dot(w, w))
    pred = a * w + fprintf_s
    return 1.0 - float(np.sum((t - pred) ** 2) / np.sum((t - t.mean()) ** 2))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--fprintf-us", type=float, required=True, help="trace cost per front, microseconds")
    a = ap.parse_args(argv)
    fp = a.fprintf_us * 1e-6
    per_case: dict[str, list] = collections.defaultdict(list)
    for tr in a.traces:
        name = pathlib.Path(tr).name
        case = name[len("fronts_"):name.index("_T1_")] if name.startswith("fronts_") and "_T1_" in name else name
        per_case[case] += parse(tr)
    every = [x for fs in per_case.values() for x in fs]
    c_asm, c_front = fit(every, fp)
    print(f"all {len(every)} fronts: c_asm = {c_asm:.4g}, c_front = {c_front:.4g}, "
          f"R2 = {r2(every, c_asm, c_front, fp):.3f}")
    for case, fs in sorted(per_case.items()):
        ca, cf = fit(fs, fp) if len(fs) > 3 else (float("nan"), float("nan"))
        print(f"  {case:34s} {len(fs):7d} fronts  own fit c_asm {ca:9.4g} c_front {cf:9.4g}  "
              f"R2 at the global fit {r2(fs, c_asm, c_front, fp):6.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
