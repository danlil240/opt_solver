"""Parse the smf MA97 plugin's per-factorization trace (SMF_PLUGIN_STATS>=2).

Trace line, one per factorization, on stderr:
  [smf_ma97] factor <k>: corr=<0|1> delays=<d> static=<s> pos=<p> neg=<q> zero=<z> L=<entries> t=<seconds>
corr=1 means the previous plugin call was also a factorization (no solve in
between), i.e. IPOPT re-factorized to correct the inertia.
"""
from __future__ import annotations

import re

_LINE = re.compile(
    r"\[smf_ma97\] factor (?P<k>\d+): corr=(?P<corr>[01]) delays=(?P<delays>-?\d+) "
    r"static=(?P<static>-?\d+) pos=(?P<pos>-?\d+) neg=(?P<neg>-?\d+) zero=(?P<zero>-?\d+) "
    r"L=(?P<L>-?\d+) t=(?P<t>\S+)"
)


def parse_trace(text: str) -> dict:
    n = corr = delays = static = 0
    factor_s = 0.0
    for m in _LINE.finditer(text):
        n += 1
        corr += int(m["corr"])
        delays += int(m["delays"])
        static += int(m["static"])
        factor_s += float(m["t"])
    return {
        "factorizations": n,
        "inertia_corrections": corr,
        "delays_total": delays,
        "static_total": static,
        "factor_s": factor_s,
    }
