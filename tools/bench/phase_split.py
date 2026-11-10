#!/usr/bin/env python3
"""Segment times of smf factorizations from SMF_BENCH_PHASE_TRACE=1 output.

The library prints `# seg <name> t=<seconds>` at fixed points of a factorization (solver/src/phase_trace.hpp);
a factorization starts at `fi_entry`. With SMF_BENCH_PHASE_TRACE=2 it also prints
`# sum <name> s=<seconds>` (top-front kernel time summed over the factorization), kept as "sum:<name>" (ms). Each factorization gives the time between consecutive marks it reached
("<a>-><b>", ms) and its total ("fi_entry-><last>").

  python tools/bench/phase_split.py <dir of phase_<case>_T<threads>_<trial>.txt files> [--segments a->b,...]

prints the median of every segment per case and thread count, over every factorization of every trial.
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import re
import statistics
import sys

MARK = re.compile(r"^# seg (\S+) t=([0-9.eE+-]+)\s*$")
# Kernel sums of a factorization (SMF_BENCH_PHASE_TRACE=2), printed at mf_p3.
SUM = re.compile(r"^# sum (\S+) s=([0-9.eE+-]+)\s*$")
FILE = re.compile(r"^phase_(.+)_T(\d+)_([^_]+)\.txt$")


def split(lines: list[str]) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    marks: list[tuple[str, float]] = []
    sums: dict[str, float] = {}

    def close() -> None:
        if len(marks) < 2:
            return
        d = {f"{a}->{b}": (tb - ta) * 1e3 for (a, ta), (b, tb) in zip(marks, marks[1:])}
        d[f"{marks[0][0]}->{marks[-1][0]}"] = (marks[-1][1] - marks[0][1]) * 1e3
        d.update(sums)
        out.append(d)

    for ln in lines:
        sm = SUM.match(ln.strip())
        if sm:
            sums[f"sum:{sm.group(1)}"] = sums.get(f"sum:{sm.group(1)}", 0.0) + float(sm.group(2)) * 1e3
            continue
        m = MARK.match(ln.strip())
        if not m:
            continue
        name, t = m.group(1), float(m.group(2))
        if name == "fi_entry" and marks:
            close()
            marks = []
            sums = {}
        marks.append((name, t))
    close()
    return out


def medians(directory: pathlib.Path) -> dict[tuple[str, int], dict[str, float]]:
    acc: dict[tuple[str, int], dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for f in sorted(pathlib.Path(directory).glob("phase_*.txt")):
        m = FILE.match(f.name)
        if not m:
            continue
        if m.group(3) == "w":  # phase_run.py's warmup run
            continue
        key = (m.group(1), int(m.group(2)))
        for fac in split(f.read_text(encoding="utf-8", errors="replace").splitlines()):
            for seg, ms in fac.items():
                acc[key][seg].append(ms)
    return {k: {s: statistics.median(v) for s, v in segs.items()} for k, segs in acc.items()}


def table(directory: pathlib.Path, segments: list[str] | None = None) -> str:
    rows = medians(directory)
    segs = segments or sorted({s for r in rows.values() for s in r})
    lines = ["| case | T | " + " | ".join(segs) + " |", "|---|---|" + "---|" * len(segs)]
    for (case, t) in sorted(rows):
        r = rows[(case, t)]
        lines.append(f"| {case} | {t} | " + " | ".join(
            f"{r[s]:.2f}" if s in r else "-" for s in segs) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--segments", default="", help="comma-separated segment names to show (default: all)")
    a = ap.parse_args(argv)
    print(table(pathlib.Path(a.dir), [s for s in a.segments.split(",") if s] or None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
