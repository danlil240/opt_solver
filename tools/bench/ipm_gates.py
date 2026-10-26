#!/usr/bin/env python3
"""Gates for the smf IPM-mode work: checks that a change left IPOPT's run alone.

identical_iterations(log_a, log_b) compares only IPOPT's iteration table
(lines like "  12  1.0e+00 ...", "  13r 1.0e+00 ..." or "2999r-5.1e+03 ...");
everything else in an
ipopt.log, such as the plugin's "[smf_ma97] ... t=..." trace and timing
lines, is ignored.

Usage (from the repo root):
  <smfbench>/python.exe tools/bench/ipm_gates.py identical <log_a> <log_b> [...]
    compares pairs (log_a, log_b), (log_c, log_d), ...; exit 1 if any differ.
"""

from __future__ import annotations

import pathlib
import re
import sys

# IPOPT prints "%4d%c%14.7e ...": "r" marks restoration, and a 14-character
# negative objective leaves no blank after it ("2999r-5.0728013e+03 ..."),
# so "-" also ends the iteration number.
ITERATION_LINE = re.compile(r"^\s*\d+r?[\s-]")


def iteration_lines(log: str) -> list[str]:
    """IPOPT iteration-table lines of a log, in order, trailing blanks stripped."""
    return [line.rstrip() for line in log.splitlines() if ITERATION_LINE.match(line)]


def identical_iterations(log_a: str, log_b: str) -> bool:
    """True when both logs have the same IPOPT iteration table."""
    return iteration_lines(log_a) == iteration_lines(log_b)


def _first_difference(a: list[str], b: list[str]) -> str:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"line {i}:\n  a: {x}\n  b: {y}"
    return f"lengths differ: {len(a)} vs {len(b)}"


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[0] != "identical" or len(argv[1:]) % 2:
        print(__doc__)
        return 2
    paths = [pathlib.Path(p) for p in argv[1:]]
    ok = True
    for pa, pb in zip(paths[0::2], paths[1::2]):
        la = pa.read_text(encoding="utf-8", errors="replace")
        lb = pb.read_text(encoding="utf-8", errors="replace")
        same = identical_iterations(la, lb)
        n = len(iteration_lines(la))
        print(f"{'IDENTICAL' if same else 'DIFFERENT'}  {n:5d} iteration lines  {pa}  vs  {pb}")
        if not same:
            print(_first_difference(iteration_lines(la), iteration_lines(lb)))
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
