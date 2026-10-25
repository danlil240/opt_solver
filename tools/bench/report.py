"""Benchmark reporting and gate evaluation for SMF."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import statistics
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

MAX_ETA_INF = 1e-10
MAX_OMEGA = 1e-8


def is_validation_accurate(
    val: Optional[Dict[str, Any]],
    max_eta_inf: float = MAX_ETA_INF,
    max_omega: float = MAX_OMEGA,
) -> bool:
    if not isinstance(val, dict):
        return False
    if not val.get("finite", False):
        return False
    eta = val.get("eta_inf")
    omega = val.get("omega")
    if eta is None or omega is None:
        return False
    try:
        eta_f = float(eta)
        omega_f = float(omega)
    except (TypeError, ValueError):
        return False
    if not (math.isfinite(eta_f) and math.isfinite(omega_f)):
        return False
    return eta_f <= max_eta_inf and omega_f <= max_omega


def qualify_record_outcome(rec: Dict[str, Any]) -> str:
    raw = str(rec.get("outcome", "Failed"))
    if raw != "Success":
        return raw
    if not is_validation_accurate(rec.get("validation")):
        return "Inaccurate"
    for sub_key in ("per_rhs_validation", "per_update_validation"):
        sub_list = rec.get(sub_key)
        if isinstance(sub_list, list):
            for v in sub_list:
                if not is_validation_accurate(v):
                    return "Inaccurate"
    return "Success"


def _geomean(values: Iterable[float]) -> Optional[float]:
    vals = [v for v in values if v is not None and v > 0.0 and math.isfinite(v)]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def _percentile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def evaluate_report(
    records: List[Dict[str, Any]],
    gate: str = "G0",
    required_baselines: Optional[List[str]] = None,
    bootstrap_samples: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    req_baselines = list(required_baselines or [])
    reasons: List[str] = []

    measured = [r for r in records if not r.get("warmup", False)]
    if not measured:
        return {
            "gate": gate,
            "qualified": False,
            "status": "no measured records",
            "geomean_speedup": None,
            "ci95_low": None,
            "ci95_high": None,
            "win_rate": 0.0,
            "coverage": 0.0,
            "instances": {},
            "reliability": {},
            "reasons": ["no_measured_records"],
        }

    all_matrices: List[str] = []
    matrix_meta: Dict[str, Tuple[str, str]] = {}
    all_solvers: List[str] = []
    # by_matrix_solver[(matrix_id, solver_id)] -> list of qualified records
    by_matrix_solver: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    for rec in measured:
        mid = str(rec.get("matrix_id", ""))
        sid = str(rec.get("solver_id", ""))
        if not mid or not sid:
            continue
        if mid not in matrix_meta:
            all_matrices.append(mid)
            seq_id = str(rec.get("sequence_id") or mid)
            fam_id = str(rec.get("family_id") or seq_id)
            matrix_meta[mid] = (fam_id, seq_id)
        if sid not in all_solvers:
            all_solvers.append(sid)
        q_outcome = qualify_record_outcome(rec)
        rec_copy = dict(rec)
        rec_copy["qualified_outcome"] = q_outcome
        by_matrix_solver.setdefault((mid, sid), []).append(rec_copy)

    # Check required baselines
    for base in req_baselines:
        base_records = [
            r
            for (m, s), rlist in by_matrix_solver.items()
            if s == base
            for r in rlist
        ]
        if not base_records or any(
            r["qualified_outcome"] == "Unavailable" for r in base_records
        ):
            reasons.append(f"missing_required_baseline:{base}")

    # Reliability table (Timeout and OutOfMemory remain in denominators!)
    total_cases_count = len(all_matrices)
    reliability: Dict[str, Dict[str, Any]] = {}
    for sid in all_solvers:
        solved = 0
        outcomes_count: Dict[str, int] = {}
        for mid in all_matrices:
            attempts = by_matrix_solver.get((mid, sid), [])
            if not attempts:
                outcomes_count["Missing"] = outcomes_count.get("Missing", 0) + 1
                continue
            # A case is solved by sid only if all measured attempts succeeded and were accurate
            outcomes = [a["qualified_outcome"] for a in attempts]
            if all(o == "Success" for o in outcomes):
                solved += 1
                outcomes_count["Success"] = outcomes_count.get("Success", 0) + 1
            else:
                first_bad = next((o for o in outcomes if o != "Success"), outcomes[0])
                outcomes_count[first_bad] = outcomes_count.get(first_bad, 0) + 1
                if sid == "smf":
                    reasons.append(f"smf_failure:{mid}:{first_bad}")
        reliability[sid] = {
            "total_cases": total_cases_count,
            "solved_cases": solved,
            "solve_rate": (solved / total_cases_count) if total_cases_count > 0 else 0.0,
            "outcomes": outcomes_count,
        }

    # Evaluate per-instance medians and speedups
    instances: Dict[str, Dict[str, Any]] = {}
    jointly_solved_count = 0
    time_ratios_smf_over_base: List[float] = []
    rss_ratios_smf_over_base: List[float] = []

    # Group jointly solved speedups by family_id -> sequence_id -> list of step speedups
    family_seq_speedups: Dict[str, Dict[str, List[float]]] = {}

    for mid in all_matrices:
        fam_id, seq_id = matrix_meta[mid]
        smf_attempts = by_matrix_solver.get((mid, "smf"), [])
        smf_valid = (
            len(smf_attempts) > 0
            and all(a["qualified_outcome"] == "Success" for a in smf_attempts)
        )
        smf_median_ms: Optional[float] = None
        smf_rss: Optional[float] = None
        if smf_valid:
            smf_times = [
                float(a["wall_ms"])
                for a in smf_attempts
                if a.get("wall_ms") is not None and float(a["wall_ms"]) > 0.0
            ]
            if smf_times:
                smf_median_ms = statistics.median(smf_times)
            rss_vals = [
                float(a["peak_rss_bytes"])
                for a in smf_attempts
                if a.get("peak_rss_bytes") is not None and float(a["peak_rss_bytes"]) > 0
            ]
            if rss_vals:
                smf_rss = max(rss_vals)

        fastest_base_ms: Optional[float] = None
        fastest_base_solver: Optional[str] = None
        fastest_base_rss: Optional[float] = None

        for sid in all_solvers:
            if sid == "smf":
                continue
            b_attempts = by_matrix_solver.get((mid, sid), [])
            # Unavailable, Timeout, OutOfMemory, Inaccurate, Failed NEVER count as valid baseline solves
            if not b_attempts or not all(
                a["qualified_outcome"] == "Success" for a in b_attempts
            ):
                continue
            b_times = [
                float(a["wall_ms"])
                for a in b_attempts
                if a.get("wall_ms") is not None and float(a["wall_ms"]) > 0.0
            ]
            if not b_times:
                continue
            b_med = statistics.median(b_times)
            if fastest_base_ms is None or b_med < fastest_base_ms:
                fastest_base_ms = b_med
                fastest_base_solver = sid
                b_rss = [
                    float(a["peak_rss_bytes"])
                    for a in b_attempts
                    if a.get("peak_rss_bytes") is not None and float(a["peak_rss_bytes"]) > 0
                ]
                fastest_base_rss = max(b_rss) if b_rss else None

        speedup: Optional[float] = None
        winner: Optional[str] = None
        if smf_median_ms is not None and fastest_base_ms is not None:
            jointly_solved_count += 1
            speedup = fastest_base_ms / smf_median_ms
            time_ratios_smf_over_base.append(smf_median_ms / fastest_base_ms)
            if smf_rss is not None and fastest_base_rss is not None and fastest_base_rss > 0:
                rss_ratios_smf_over_base.append(smf_rss / fastest_base_rss)
            if speedup >= 1.05:
                winner = "smf"
            elif speedup <= (1.0 / 1.05):
                winner = fastest_base_solver
            else:
                winner = "tie"
            family_seq_speedups.setdefault(fam_id, {}).setdefault(seq_id, []).append(
                speedup
            )
        elif smf_median_ms is not None:
            winner = "smf_only"
        elif fastest_base_ms is not None:
            winner = fastest_base_solver

        instances[mid] = {
            "family_id": fam_id,
            "sequence_id": seq_id,
            "smf_ms": smf_median_ms,
            "fastest_baseline_ms": fastest_base_ms,
            "fastest_baseline_solver": fastest_base_solver,
            "speedup": speedup,
            "winner": winner,
        }

    coverage = (
        jointly_solved_count / total_cases_count if total_cases_count > 0 else 0.0
    )

    if jointly_solved_count == 0 and any(s != "smf" for s in all_solvers):
        reasons.append("insufficient_coverage:0_jointly_solved")
        return {
            "gate": gate,
            "qualified": False,
            "status": "insufficient coverage",
            "geomean_speedup": None,
            "ci95_low": None,
            "ci95_high": None,
            "win_rate": 0.0,
            "coverage": coverage,
            "instances": instances,
            "reliability": reliability,
            "reasons": reasons,
        }

    # Compute sequence-level speedups and family-level speedups (equal family weights)
    seq_speedups_all: List[float] = []
    family_speedups: Dict[str, List[float]] = {}
    for fam_id, seq_map in family_seq_speedups.items():
        for seq_id, step_ratios in seq_map.items():
            seq_gm = _geomean(step_ratios)
            if seq_gm is not None:
                seq_speedups_all.append(seq_gm)
                family_speedups.setdefault(fam_id, []).append(seq_gm)

    per_family_gm: List[float] = []
    for fam_id in sorted(family_speedups.keys()):
        fgm = _geomean(family_speedups[fam_id])
        if fgm is not None:
            per_family_gm.append(fgm)

    geomean_speedup = _geomean(per_family_gm)
    win_rate = (
        sum(1 for s in seq_speedups_all if s >= 1.05) / len(seq_speedups_all)
        if seq_speedups_all
        else 0.0
    )

    # Bootstrap 95% confidence interval by resampling whole families (and whole sequences within families)
    ci95_low: Optional[float] = None
    ci95_high: Optional[float] = None
    if per_family_gm:
        if len(per_family_gm) == 1 and len(seq_speedups_all) == 1:
            ci95_low = geomean_speedup
            ci95_high = geomean_speedup
        else:
            rng = random.Random(seed)
            fam_keys = sorted(family_speedups.keys())
            boot_means: List[float] = []
            for _ in range(max(100, bootstrap_samples)):
                sampled_fams = [rng.choice(fam_keys) for _ in fam_keys]
                sampled_fam_gms: List[float] = []
                for fk in sampled_fams:
                    seqs = family_speedups[fk]
                    sampled_seqs = [rng.choice(seqs) for _ in seqs]
                    gm_fk = _geomean(sampled_seqs)
                    if gm_fk is not None:
                        sampled_fam_gms.append(gm_fk)
                bgm = _geomean(sampled_fam_gms)
                if bgm is not None:
                    boot_means.append(bgm)
            boot_means.sort()
            ci95_low = _percentile(boot_means, 0.025)
            ci95_high = _percentile(boot_means, 0.975)

    p95_time_ratio = (
        _percentile(sorted(time_ratios_smf_over_base), 0.95)
        if time_ratios_smf_over_base
        else None
    )
    max_rss_ratio = max(rss_ratios_smf_over_base) if rss_ratios_smf_over_base else None

    # Gate checks
    if gate in ("G2", "G3", "G4"):
        if coverage < 0.90:
            reasons.append(f"coverage_below_0.90:{coverage:.3f}")
        min_gm = 1.5 if gate == "G3" else 1.2
        min_win = 0.70 if gate == "G3" else 0.60
        if geomean_speedup is None or geomean_speedup < min_gm:
            reasons.append(f"geomean_below_{min_gm}:{geomean_speedup}")
        if ci95_low is None or ci95_low <= 1.0:
            reasons.append(f"ci95_low_not_gt_1.0:{ci95_low}")
        if win_rate < min_win:
            reasons.append(f"win_rate_below_{min_win}:{win_rate:.3f}")
        if p95_time_ratio is not None and p95_time_ratio > 1.25:
            reasons.append(f"p95_time_ratio_exceeds_1.25:{p95_time_ratio:.3f}")
        if max_rss_ratio is not None and max_rss_ratio > 1.25:
            reasons.append(f"rss_ratio_exceeds_1.25:{max_rss_ratio:.3f}")

    qualified = len(reasons) == 0
    return {
        "gate": gate,
        "qualified": qualified,
        "status": "ok" if qualified else "failed_qualification",
        "geomean_speedup": geomean_speedup,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "win_rate": win_rate,
        "coverage": coverage,
        "p95_time_ratio": p95_time_ratio,
        "max_rss_ratio": max_rss_ratio,
        "instances": instances,
        "reliability": reliability,
        "reasons": reasons,
    }


def load_records_from_path(input_path: pathlib.Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    files: List[pathlib.Path] = []
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(input_path.glob("*.jsonl"))
    for fpath in files:
        with fpath.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    return records


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate SMF benchmark JSONL records.")
    parser.add_argument("--input", required=True, help="Path to JSONL file or directory")
    parser.add_argument("--gate", default="G0", choices=["G0", "G1", "G2", "G3", "G4"])
    parser.add_argument(
        "--require-baselines",
        nargs="*",
        default=None,
        help="Required baseline solver IDs",
    )
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    args = parser.parse_args(argv)

    in_path = pathlib.Path(args.input)
    if not in_path.exists():
        print(f"error: input path does not exist: {in_path}", file=sys.stderr)
        return 2

    try:
        records = load_records_from_path(in_path)
    except Exception as exc:
        print(f"error: failed to read JSONL records: {exc}", file=sys.stderr)
        return 2

    summary = evaluate_report(
        records,
        gate=args.gate,
        required_baselines=args.require_baselines,
    )
    out_json = json.dumps(summary, indent=2)
    if args.output:
        out_path = pathlib.Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_json + "\n", encoding="utf-8")
    print(out_json)
    return 0 if summary["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
