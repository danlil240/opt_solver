#!/usr/bin/env python3
"""Fetch the extra SuiteSparse matrices of the front-tier set and (re)build
benchmarks/manifests/fronttier.json.

Classes
  hard_indef : Duff-Hogg-Lopez (RAL-TR-2018-012) hard indefinite set, plus every
               GHS_indef matrix already in large.json
  small_kkt  : optim20 KKT cases + PGLib-OPF captures from ipopt_hard.json
  large_kkt  : the 7 ipopt_kkt.json models
  spd        : large.json SPD matrices
Every indefinite case appears on two tracks: "recommended" (u = 0.01, default
ordering) and "ipopt" (u = 1e-8, matching ordering via the per-case "ordering"
override). SPD cases are on "recommended" only. The tuning/heldout split is
drawn per class at matrix level (both tracks share it) with a fixed seed.

Usage:
  python tools/bench/fetch_fronttier.py [--fetch] [--budget-mb 1500]
      [--out benchmarks/fixtures/large] [--manifest benchmarks/manifests/fronttier.json]
Run with below-normal priority; matrices are only downloaded with --fetch.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import shutil
import sys
import tarfile
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import fetch_suitesparse as fs  # noqa: E402

SEED = 20261027
HELDOUT_FRACTION = 0.30
ROOT = pathlib.Path(__file__).resolve().parents[2]

# Duff-Hogg-Lopez (RAL-TR-2018-012) hard indefinite set, all 24 matrices
# (.superpowers/sdd/2026-10-27-front-tier-solver/dhl-hard-set.md).
# (group, name, family).
DHL_HARD = [
    ("TSOPF", "TSOPF_FS_b39_c7", "power_flow"), ("TSOPF", "TSOPF_FS_b162_c1", "power_flow"),
    ("QY", "case39", "power_flow"), ("TSOPF", "TSOPF_FS_b39_c19", "power_flow"),
    ("TSOPF", "TSOPF_FS_b39_c30", "power_flow"), ("GHS_indef", "cont-201", "kkt_qp"),
    ("GHS_indef", "stokes128", "kkt_pde"), ("TSOPF", "TSOPF_FS_b162_c3", "power_flow"),
    ("TSOPF", "TSOPF_FS_b162_c4", "power_flow"), ("GHS_indef", "ncvxqp1", "kkt_qp"),
    ("GHS_indef", "darcy003", "kkt_pde"), ("GHS_indef", "cont-300", "kkt_qp"),
    ("GHS_indef", "bratu3d", "indef_pde"), ("GHS_indef", "cvxqp3", "kkt_qp"),
    ("TSOPF", "TSOPF_FS_b300", "power_flow"), ("TSOPF", "TSOPF_FS_b300_c1", "power_flow"),
    ("GHS_indef", "d_pretok", "kkt_pde"), ("GHS_indef", "turon_m", "kkt_pde"),
    ("TSOPF", "TSOPF_FS_b300_c2", "power_flow"), ("TSOPF", "TSOPF_FS_b300_c3", "power_flow"),
    ("GHS_indef", "ncvxqp5", "kkt_qp"), ("GHS_indef", "ncvxqp3", "kkt_qp"),
    ("GHS_indef", "ncvxqp7", "kkt_qp"), ("Schenk_IBMNA", "c-big", "indef_opt"),
]
# Singular matrices: kept for robustness reporting, excluded from accuracy gates.
REPORT_ONLY = {"d_pretok", "turon_m"}


LABEL_MAP = {"train": "tuning", "test": "heldout", "tuning": "tuning", "heldout": "heldout"}


def assign_splits(ids: list[str], seed: int = SEED, fraction: float = HELDOUT_FRACTION,
                  fixed: dict[str, str] | None = None) -> dict[str, str]:
    """Deterministic per-class split of about `fraction` held out (>= 1).

    `fixed` maps matrix_id -> pre-existing label ("tuning"/"heldout"); those are
    kept and only the unlabeled ids are drawn by the seed to reach the target.
    """
    ids = sorted(set(ids))
    fixed = {k: v for k, v in (fixed or {}).items() if k in ids}
    free = [i for i in ids if i not in fixed]
    rng = random.Random(seed)
    order = free[:]
    rng.shuffle(order)
    target = min(max(1, round(fraction * len(ids))), len(ids)) if ids else 0
    have = sum(v == "heldout" for v in fixed.values())
    k = min(len(order), max(0, target - have))
    held = set(order[:k])
    out = dict(fixed)
    out.update({i: ("heldout" if i in held else "tuning") for i in free})
    return out


def _load(name: str) -> dict:
    return json.loads((ROOT / "benchmarks" / "manifests" / f"{name}.json")
                      .read_text(encoding="utf-8"))


def archive_size(group: str, name: str) -> int | None:
    for tmpl in fs.MIRRORS:
        url = tmpl.format(group=group, name=name)
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "smf-bench/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                n = r.headers.get("Content-Length")
                if n:
                    return int(n)
        except Exception:
            continue
    return None


def fetch_new(out: pathlib.Path, wanted: list[tuple[str, str, str]], budget_mb: float,
              do_fetch: bool) -> tuple[dict[str, dict], dict[str, str], float]:
    """Return ({name: case-core}, {name: failure reason}, downloaded_mb)."""
    got: dict[str, dict] = {}
    why: dict[str, str] = {}
    spent = 0.0
    for group, name, family in wanted:
        mdir = out / name
        mtx = mdir / f"{name}.mtx"
        if not mtx.exists():
            if not do_fetch:
                why[name] = "not on disk: download deferred (slow mirror at low priority); run fetch_fronttier.py --fetch"
                continue
            size = archive_size(group, name)
            mb = (size or 0) / 1e6
            if size is None:
                print(f"warning: no Content-Length for {group}/{name}", flush=True)
            if spent + mb > budget_mb:
                why[name] = (f"deferred: {mb:.0f} MB would exceed the "
                             f"{budget_mb:.0f} MB download budget")
                continue
            if mdir.exists():
                shutil.rmtree(mdir)
            mdir.mkdir(parents=True)
            archive = mdir / f"{name}.tar.gz"
            try:
                url = fs.download(group, name, archive)
                with tarfile.open(archive, "r:gz") as tf:
                    member = next(m for m in tf.getmembers()
                                  if m.isfile() and m.name.endswith(f"/{name}.mtx"))
                    member.name = f"{name}.mtx"
                    tf.extract(member, mdir, filter="data")
                archive.unlink()
                spent += mb
                print(f"fetched {group}/{name} ({mb:.1f} MB) from {url}", flush=True)
            except Exception as exc:
                why[name] = f"download failed: {exc}"
                if mdir.exists():
                    shutil.rmtree(mdir)
                continue
        banner, rows, cols, nnz = fs.read_header(mtx)
        if "coordinate" not in banner or "symmetric" not in banner or rows != cols:
            why[name] = f"unexpected header '{banner}' {rows}x{cols}"
            continue
        got[name] = {
            "matrix_id": name, "sequence_id": name, "family_id": family,
            "source": f"SuiteSparse {group}/{name}",
            "mtx_path": mtx.relative_to(ROOT).as_posix(),
            "matrix_sha256": fs.sha256_file(mtx), "n": rows, "nnz_lower": nnz,
            "matrix_type": "indefinite", "mode": "cold",
            "expected_outcome": "Success",
        }
    return got, why, spent


def _exists(case: dict) -> bool:
    for key in ("mtx_path", "rhs_path", "seq_values_path"):
        if case.get(key) and not (ROOT / case[key]).exists():
            return False
    return True


def _fill_dims(case: dict) -> dict:
    if "n" not in case or "nnz_lower" not in case:
        _, rows, _, nnz = fs.read_header(ROOT / case["mtx_path"])
        case["n"], case["nnz_lower"] = rows, nnz
    return case


def collect(out: pathlib.Path, do_fetch: bool, budget_mb: float) -> dict:
    large, optim20 = _load("large"), _load("optim20")
    ipopt_kkt, ipopt_hard = _load("ipopt_kkt"), _load("ipopt_hard")
    large_by = {c["matrix_id"]: c for c in large["cases"]}

    base: list[tuple[str, dict]] = []   # (class, case without track/split)
    missing: list[dict] = []
    prelabel: dict[str, dict[str, str]] = {}   # class -> matrix_id -> existing label

    def add(cls: str, case: dict, source_manifest: str) -> None:
        if not _exists(case):
            missing.append({"matrix_id": case["matrix_id"], "class": cls,
                            "source": source_manifest,
                            "reason": "fixture file(s) absent on disk"})
            return
        c = {k: v for k, v in case.items() if k not in ("split", "track")}
        if source_manifest == "ipopt_hard.json" and case.get("split") in LABEL_MAP:
            prelabel.setdefault(cls, {})[case["matrix_id"]] = LABEL_MAP[case["split"]]
        base.append((cls, _fill_dims(c)))

    # hard_indef: DHL names (reuse large.json entry when present) + all large GHS_indef
    to_fetch = []
    seen = set()
    for group, name, family in DHL_HARD:
        if name in large_by:
            add("hard_indef", large_by[name], "large.json")
            seen.add(name)
        else:
            to_fetch.append((group, name, family))
    got, why, spent = fetch_new(out, to_fetch, budget_mb, do_fetch)
    for group, name, family in to_fetch:
        if name in got:
            base.append(("hard_indef", got[name]))
            seen.add(name)
        else:
            missing.append({"matrix_id": name, "class": "hard_indef",
                            "source": f"SuiteSparse {group}/{name}", "reason": why[name]})
    for c in large["cases"]:
        if c["matrix_type"] == "indefinite" and c["matrix_id"] not in seen:
            add("hard_indef", c, "large.json")

    for c in optim20["cases"]:
        if c.get("family_id", "").startswith("kkt_"):
            add("small_kkt", c, "optim20.json")
    for c in ipopt_hard["cases"]:
        if c["matrix_id"].startswith("pglib_"):
            add("small_kkt", c, "ipopt_hard.json")
        elif c["matrix_id"].startswith("cops_"):
            add("large_kkt", c, "ipopt_hard.json")
    for c in ipopt_kkt["cases"]:
        add("large_kkt", c, "ipopt_kkt.json")
    for c in large["cases"]:
        if c["matrix_type"] == "spd":
            add("spd", c, "large.json")

    # split per class at matrix level
    splits: dict[str, dict[str, str]] = {}
    for cls in {c for c, _ in base}:
        splits[cls] = assign_splits([x["matrix_id"] for k, x in base if k == cls],
                                   fixed=prelabel.get(cls))

    cases = []
    for cls, c in base:
        tracks = ["recommended"] if c["matrix_type"] == "spd" else ["recommended", "ipopt"]
        for track in tracks:
            e = dict(c)
            e["class"] = cls
            e["split"] = splits[cls][c["matrix_id"]]
            e["track"] = track
            if c["matrix_id"] in REPORT_ONLY:
                e["report_only"] = True   # singular: robustness reporting only
            if track == "ipopt":
                e["ordering"] = "matching"   # run.py -> bench_replay --ordering
            cases.append(e)
    cases.sort(key=lambda e: (["small_kkt", "large_kkt", "hard_indef", "spd"].index(e["class"]),
                              e["n"], e["matrix_id"], e["track"]))
    return {
        "schema_version": 1,
        "suite": "fronttier",
        "seed": SEED,
        "heldout_fraction": HELDOUT_FRACTION,
        "tracks": {
            "recommended": "pivot u = 0.01, default ordering",
            "ipopt": "pivot u = 1e-8 (bench_replay track), ordering = matching (per-case override)",
        },
        "solvers": ["smf", "ma27", "mumps", "mkl_pardiso", "cholmod", "ma57", "ma97", "ssids"],
        "required_baselines": [],
        "cases": cases,
        "missing": missing,
        "_downloaded_mb": round(spent, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/fixtures/large")
    ap.add_argument("--manifest", default="benchmarks/manifests/fronttier.json")
    ap.add_argument("--fetch", action="store_true", help="download missing matrices")
    ap.add_argument("--budget-mb", type=float, default=1500.0)
    args = ap.parse_args()
    manifest = collect(ROOT / args.out, args.fetch, args.budget_mb)
    dl = manifest.pop("_downloaded_mb")
    (ROOT / args.manifest).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.manifest}: {len(manifest['cases'])} cases, "
          f"{len(manifest['missing'])} missing, downloaded {dl} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
