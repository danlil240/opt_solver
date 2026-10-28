#!/usr/bin/env python3
"""Phase 0 exit report for the IPM-mode work.

Reads the hard-corpus baseline (ipopt_kkt.py e2e output, smf results carry the
plugin trace fields) and the hard manifest, then writes a table and the Phase 0
exit verdict: at least 3 *tuning* models must show inertia-correction or
delayed-pivot pressure, otherwise more hard models are needed before Phase 1.

Usage (from the repo root):
  <smfbench>/python.exe tools/bench/phase0_report.py
"""

from __future__ import annotations

import argparse
import json
import pathlib

import hard_models

REPO = pathlib.Path(__file__).resolve().parents[2]
CORR_FRAC = 0.10        # inertia_corrections / factorizations
DELAY_FRAC = 0.01       # delays_total / factorizations, relative to n
MIN_MODELS = 3


def _split(r: dict) -> str:
    return r.get("split") or hard_models.split_of(r["model"])


def _pressure(r: dict, n: int) -> tuple[float, float, bool]:
    fact = r["factorizations"]
    corr_frac = r["inertia_corrections"] / fact
    delay_per_fact = r["delays_total"] / fact
    return corr_frac, delay_per_fact, corr_frac >= CORR_FRAC or delay_per_fact >= DELAY_FRAC * n


def qualified_by(r: dict, n: int) -> str:
    """Which half of the criterion a model's smf run meets: corrections, delays, both or no."""
    if not r.get("factorizations"):
        return "no"
    fact = r["factorizations"]
    c = r["inertia_corrections"] / fact >= CORR_FRAC
    d = r["delays_total"] / fact >= DELAY_FRAC * n
    return "both" if c and d else "corrections" if c else "delays" if d else "no"


def corrections_only(results: list[dict], n_by_model: dict[str, int]) -> list[str]:
    """Tuning models that qualify through inertia corrections (alone or with delays).

    delays_total counts delay events, not distinct columns, so the delay half is
    weak evidence; this is the stricter view reported next to the verdict.
    """
    names: list[str] = []
    for r in results:
        if not str(r.get("solver", "")).startswith("smf") or _split(r) != "tuning" or r["model"] in names:
            continue
        if qualified_by(r, n_by_model[r["model"]]) in ("corrections", "both"):
            names.append(r["model"])
    return names


def exit_ok(results: list[dict], n_by_model: dict[str, int]) -> tuple[bool, list[str]]:
    """True when >= 3 tuning models qualify; also returns the qualifying tuning models.

    A model qualifies when, in its smf run, inertia_corrections/factorizations >= 0.10
    or delays_total/factorizations >= 0.01 * n. Held-out models never count.
    """
    names: list[str] = []
    for r in results:
        if not str(r.get("solver", "")).startswith("smf") or not r.get("factorizations"):
            continue
        if _split(r) != "tuning" or r["model"] in names:
            continue
        if _pressure(r, n_by_model[r["model"]])[2]:
            names.append(r["model"])
    return len(names) >= MIN_MODELS, names


def _by(results: list[dict]) -> dict:
    out: dict[str, dict[str, dict]] = {}
    for r in results:
        out.setdefault(r["model"], {})[r["solver"]] = r
    return out


def report(results: list[dict], n_by_model: dict[str, int]) -> str:
    ok, names = exit_ok(results, n_by_model)
    by = _by(results)
    solvers = sorted({r["solver"] for r in results})
    lines = [
        "# Phase 0 exit report",
        "",
        "Hard IPOPT corpus, single-thread baseline. Pressure columns come from the smf plugin trace.",
        "",
        "| model | split | n | factorizations | corrections | corr frac | delays/fact | 0.01 n | qualified by | "
        + " | ".join(f"{s} status (iters)" for s in solvers) + " |",
        "|" + "---|" * (9 + len(solvers)),
    ]
    for model, runs in by.items():
        smf = runs.get("smf")
        n = n_by_model.get(model, 0)
        if smf and smf.get("factorizations"):
            cf, df, _ = _pressure(smf, n)
            cols = [str(smf["factorizations"]), str(smf["inertia_corrections"]), f"{cf:.3f}",
                    f"{df:.2f}", f"{DELAY_FRAC * n:.2f}", qualified_by(smf, n)]
        else:
            cols = ["-"] * 5 + ["no"]
        status = [f"{runs[s]['status']} ({runs[s]['iterations']})" if s in runs else "-" for s in solvers]
        lines.append(f"| {model} | {hard_models.split_of(model)} | {n} | " + " | ".join(cols)
                     + " | " + " | ".join(status) + " |")
    corr = corrections_only(results, n_by_model)
    lines += [
        "",
        f"Qualification: inertia_corrections/factorizations >= {CORR_FRAC} or "
        f"delays_total/factorizations >= {DELAY_FRAC} * n; only tuning models count.",
        "",
        f"Qualifying tuning models ({len(names)}): " + (", ".join(names) if names else "none"),
        "",
        f"Qualifying by inertia corrections only ({len(corr)}): " + (", ".join(corr) if corr else "none")
        + ". delays_total counts delay events, not distinct columns, so the delay criterion is weak "
        "evidence; the verdict above uses the plan's either-criterion rule.",
        "",
        f"**Phase 0 verdict: {'PASS' if ok else 'FAIL'}** (need >= {MIN_MODELS} tuning models)"
        + ("" if ok else "; add harder models before Phase 1."),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(REPO / "benchmarks" / "results" / "ipopt_hard_baseline_t1.json"))
    ap.add_argument("--manifest", default=str(REPO / "benchmarks" / "manifests" / "ipopt_hard.json"))
    ap.add_argument("--output", default=str(REPO / "benchmarks" / "results" / "phase0_exit.md"))
    args = ap.parse_args()
    results = json.loads(pathlib.Path(args.baseline).read_text(encoding="utf-8"))
    manifest = json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))
    n_by_model = {c["matrix_id"]: c["n"] for c in manifest["cases"]}
    for r in results:  # models whose capture failed: fall back to the IPOPT problem size
        n_by_model.setdefault(r["model"], (r.get("n_vars") or 0) + (r.get("n_eq") or 0))
    text = report(results, n_by_model)
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    ok, _ = exit_ok(results, n_by_model)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
