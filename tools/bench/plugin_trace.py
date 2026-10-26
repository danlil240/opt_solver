"""Parse the smf MA97 plugin's per-factorization trace (SMF_PLUGIN_STATS>=2).

Trace line, one per factorization, on stderr:
  [smf_ma97] factor <k>: corr=<0|1> delays=<d> static=<s> pos=<p> neg=<q> zero=<z> L=<entries> t=<seconds>
corr=1 means the previous plugin call was also a factorization (no solve in
between), i.e. IPOPT re-factorized to correct the inertia. A trailing
ra=<tried>/<adopted>/<reverted> gives the session's re-analysis counts so far;
adopted > reverted means the factorization ran on a re-analysed order.

At unload the plugin prints the IpmSession totals:
  [smf_ma97] ipm: sessions=<s> ... reanalyses_tried=<t> reanalyses_adopted=<a>
  reanalyses_reverted=<r> reanalyse_s=<seconds> ...
"""
from __future__ import annotations

import re

_LINE = re.compile(
    r"\[smf_ma97\] factor (?P<k>\d+): corr=(?P<corr>[01]) delays=(?P<delays>-?\d+) "
    r"static=(?P<static>-?\d+) pos=(?P<pos>-?\d+) neg=(?P<neg>-?\d+) zero=(?P<zero>-?\d+) "
    r"L=(?P<L>-?\d+) t=(?P<t>\S+)(?:.* ra=(?P<rt>\d+)/(?P<ra>\d+)/(?P<rr>\d+))?"
)
_IPM = re.compile(r"\[smf_ma97\] ipm: (.*)$", re.MULTILINE)


def parse_trace(text: str) -> dict:
    n = corr = delays = static = n_re = delays_re = 0
    factor_s = 0.0
    for m in _LINE.finditer(text):
        n += 1
        corr += int(m["corr"])
        delays += int(m["delays"])
        static += int(m["static"])
        factor_s += float(m["t"])
        if m["ra"] is not None and int(m["ra"]) > int(m["rr"]):
            n_re += 1
            delays_re += int(m["delays"])
    ipm = {}
    for m in _IPM.finditer(text):
        ipm = dict(tok.split("=", 1) for tok in m.group(1).split() if "=" in tok)

    def ipm_field(key, cast):
        return cast(ipm[key]) if key in ipm else None

    return {
        "factorizations": n,
        "inertia_corrections": corr,
        "delays_total": delays,
        "static_total": static,
        "factor_s": factor_s,
        "reanalyses_tried": ipm_field("reanalyses_tried", int),
        "reanalyses_adopted": ipm_field("reanalyses_adopted", int),
        "reanalyses_reverted": ipm_field("reanalyses_reverted", int),
        "reanalyse_s": ipm_field("reanalyse_s", float),
        "factorizations_reanalysed": n_re,
        "delays_reanalysed": delays_re,
    }
