#!/usr/bin/env python3
"""Select and fetch a sealed public SuiteSparse benchmark corpus.

Usage:
  python tools/bench/fetch_public_suitesparse.py [--fetch]
      [--ssstats path/to/ssstats.csv]
      [--fronttier benchmarks/manifests/fronttier.json]
      [--out benchmarks/fixtures/public_suitesparse]
      [--manifest benchmarks/manifests/public_suitesparse.json]
      [--budget-mb 2500]

Without --fetch, prints the deterministic 60-matrix selection summary table
without downloading archives or writing the manifest.
"""

from __future__ import annotations

import argparse
import csv
import io
import itertools
import json
import math
import pathlib
import re
import shutil
import sys
import tarfile
import urllib.request

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import fetch_suitesparse as fs  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
SSSTATS_URL = "https://www.cise.ufl.edu/research/sparse/matrices/ssstats.csv"

SERIES_PREFIXES = (
    "TSOPF_FS_",
    "af_shell",
    "thermal",
    "ncvxqp",
    "cvxqp",
    "cont-",
    "boyd",
    "aug3d",
    "cfd",
    "Dubcova",
    "apache",
    "stokes",
    "mario",
    "darcy",
    "bratu",
    "helm",
    "copter",
    "dawson",
    "blockqp",
)
ND_SERIES_MARKER = "nd<digits>k"
ND_SERIES_RE = re.compile(r"^nd\d+k$")

EXCLUDED_KINDS = (
    "graph",
    "combinatorial",
    "directed",
    "undirected",
    "weighted",
    "bipartite",
    "duplicate",
)
KKT_KINDS = (
    "optim",
    "linear programming",
    "power network",
    "economic",
    "circuit",
)

MIN_N = 20_000
MAX_N = 600_000
MAX_NNZ = 30_000_000

CLASS_ORDER = ("public_indef_kkt", "public_indef_pde", "public_spd")
TARGETS: dict[str, tuple[int, int]] = {
    "public_indef_kkt": (20, 10),
    "public_indef_pde": (15, 3),
    "public_spd": (25, 3),
}
SOLVERS = [
    "smf",
    "ma57",
    "ma97",
    "mkl_pardiso",
    "panua_pardiso",
    "mumps",
    "cholmod",
    "ssids",
    "cudss",
]

_MTX_ROW_DTYPE = np.dtype([("r", np.int32), ("c", np.int32), ("v", np.float64)])
_CHUNK_ROWS = 1_000_000


def fronttier_exclusion_stems(manifest: dict) -> tuple[set[str], set[str]]:
    """Return (excluded_names, excluded_stems) from SuiteSparse entries in manifest."""
    excluded_names: set[str] = set()
    excluded_stems: set[str] = set()
    entries = list(manifest.get("cases", [])) + list(manifest.get("missing", []))
    for entry in entries:
        if "SuiteSparse" not in entry.get("source", ""):
            continue
        name = entry.get("matrix_id", "")
        if not name:
            continue
        excluded_names.add(name)
        for prefix in SERIES_PREFIXES:
            if name.startswith(prefix):
                excluded_stems.add(prefix)
        if ND_SERIES_RE.match(name):
            excluded_stems.add(ND_SERIES_MARKER)
    return excluded_names, excluded_stems


def is_excluded(name: str, excluded_names: set[str], excluded_stems: set[str]) -> bool:
    """Return True if `name` matches an excluded exact name or series stem."""
    if name in excluded_names:
        return True
    for stem in excluded_stems:
        if stem == ND_SERIES_MARKER:
            if ND_SERIES_RE.match(name):
                return True
        elif name.startswith(stem):
            return True
    return False


def parse_ssstats_meta(csv_text: str) -> dict[str, str]:
    """Extract optional count/date header metadata from ssstats.csv text."""
    lines = csv_text.splitlines()
    if len(lines) >= 2 and lines[0].strip().isdigit():
        return {"count": lines[0].strip(), "date": lines[1].strip()}
    return {}


def parse_ssstats(csv_text: str) -> list[dict]:
    """Parse SuiteSparse ssstats.csv text into a list of entry dicts."""
    lines = csv_text.splitlines()
    start = 0
    if len(lines) >= 2 and lines[0].strip().isdigit():
        start = 2
    elif lines and lines[0].strip().lower().startswith("group,name,"):
        start = 1

    entries: list[dict] = []
    reader = csv.reader(io.StringIO("\n".join(lines[start:])))
    for row in reader:
        if not row or all(not cell.strip() for cell in row):
            continue
        if len(row) < 12:
            continue
        kind = ",".join(row[11:]).strip()
        entries.append({
            "group": row[0].strip(),
            "name": row[1].strip(),
            "nrows": int(row[2]),
            "ncols": int(row[3]),
            "nnz": int(row[4]),
            "is_real": int(row[5]),
            "is_binary": int(row[6]),
            "is_nd": int(row[7]),
            "posdef": int(row[8]),
            "psym": float(row[9]),
            "nsym": float(row[10]),
            "kind": kind,
        })
    return entries


def classify_entry(entry: dict) -> str | None:
    """Return benchmark class for an eligible ssstats entry, or None if ineligible."""
    if entry["is_real"] != 1 or entry["is_binary"] != 0:
        return None
    if entry["nrows"] != entry["ncols"] or entry["nsym"] != 1.0:
        return None
    if not (MIN_N <= entry["nrows"] <= MAX_N) or not (0 <= entry["nnz"] <= MAX_NNZ):
        return None
    kind_lower = entry["kind"].lower()
    if any(bad in kind_lower for bad in EXCLUDED_KINDS):
        return None
    if entry["posdef"] == 1:
        return "public_spd"
    if entry["posdef"] == 0:
        if any(kkt in kind_lower for kkt in KKT_KINDS):
            return "public_indef_kkt"
        return "public_indef_pde"
    return None


def _pick_log_spaced(items: list[dict], cap: int) -> list[dict]:
    """Sort group entries by (nrows, name) and pick up to `cap` evenly spaced in log(nrows)."""
    ordered = sorted(items, key=lambda e: (e["nrows"], e["name"]))
    m = len(ordered)
    if cap <= 0:
        return []
    if m <= cap:
        return ordered
    if cap == 1:
        return [ordered[m // 2]]

    logs = [math.log(e["nrows"]) for e in ordered]
    lo, hi = logs[0], logs[-1]
    if lo == hi:
        indices = [round(i * (m - 1) / (cap - 1)) for i in range(cap)]
        return [ordered[i] for i in indices]

    chosen: set[int] = set()
    for j in range(cap):
        target = lo + j * (hi - lo) / (cap - 1)
        best_idx = min(
            (i for i in range(m) if i not in chosen),
            key=lambda i: (abs(logs[i] - target), i),
        )
        chosen.add(best_idx)
    return [ordered[i] for i in sorted(chosen)]


def rank_candidates(
    entries: list[dict],
    excluded_names: set[str],
    excluded_stems: set[str],
    targets: dict[str, tuple[int, int]] = TARGETS,
) -> dict[str, list[dict]]:
    """Return full deterministic per-group-capped round-robin candidate lists per class."""
    by_cls_group: dict[str, dict[str, list[dict]]] = {cls: {} for cls in targets}
    for entry in entries:
        if is_excluded(entry["name"], excluded_names, excluded_stems):
            continue
        cls = classify_entry(entry)
        if cls is None or cls not in targets:
            continue
        by_cls_group[cls].setdefault(entry["group"], []).append(entry)

    ranked: dict[str, list[dict]] = {}
    for cls, (_target_count, cap) in targets.items():
        groups = sorted(by_cls_group[cls])
        picked = {grp: _pick_log_spaced(by_cls_group[cls][grp], cap) for grp in groups}
        round_robin: list[dict] = []
        for slot in range(cap):
            for grp in groups:
                if slot < len(picked[grp]):
                    round_robin.append(picked[grp][slot])
        ranked[cls] = round_robin
    return ranked


def select_candidates(
    entries: list[dict],
    excluded_names: set[str],
    excluded_stems: set[str],
    targets: dict[str, tuple[int, int]] = TARGETS,
) -> dict[str, list[dict]]:
    """Select up to target_count deterministic candidates per class."""
    ranked = rank_candidates(entries, excluded_names, excluded_stems, targets=targets)
    return {cls: ranked[cls][: targets[cls][0]] for cls in targets}


def _header_singularity_notice(comments: list[str]) -> bool:
    for line in comments:
        lower = line.lower()
        cleaned = re.sub(r"\b(?:non-?singular|not\s+singular)\b", "", lower)
        if re.search(r"\b(?:singular|rank[- ]deficient)\b", cleaned):
            return True
    return False


def _read_mtx_preamble(f) -> tuple[str, list[str], int, int, int]:
    banner = f.readline().strip()
    comments: list[str] = []
    line = f.readline()
    while line and (line.startswith("%") or not line.strip()):
        if line.startswith("%"):
            comments.append(line)
        line = f.readline()
    if not line:
        raise ValueError("missing Matrix Market dimension line")
    rows, cols, nnz = (int(t) for t in line.split()[:3])
    return banner, comments, rows, cols, nnz


def _iter_mtx_chunks(f, symmetric: bool):
    while True:
        raw = list(itertools.islice(f, _CHUNK_ROWS))
        if not raw:
            break
        chunk = [ln for ln in raw if ln.strip() and not ln.lstrip().startswith("%")]
        if not chunk:
            continue
        data = np.atleast_1d(np.loadtxt(chunk, dtype=_MTX_ROW_DTYPE))
        if data.size == 0:
            continue
        v = data["v"]
        nz = v != 0.0
        if not np.any(nz):
            continue
        r = data["r"][nz] - 1
        c = data["c"][nz] - 1
        v = v[nz]
        off = (r != c) if symmetric else np.zeros(r.shape, dtype=bool)
        yield r, c, v, off


def check_algebraic_singularity(mtx_path: pathlib.Path) -> tuple[bool, str]:
    """Check whether a real symmetric coordinate .mtx matrix is algebraically singular.

    Streams coordinates in chunks so large matrices (nnz <= 15M) run in low RAM.
    Returns (is_singular, reason).
    """
    with mtx_path.open("r", encoding="ascii", errors="replace") as f:
        banner, comments, rows, cols, _nnz = _read_mtx_preamble(f)
        if _header_singularity_notice(comments):
            return True, "header comment indicates singular or rank-deficient matrix"
        if rows == 0 or cols == 0:
            return False, ""
        symmetric = "symmetric" in banner.lower()
        row_abs_sum = np.zeros(rows, dtype=np.float64)
        diag = np.zeros(rows, dtype=np.float64)

        for r, c, v, off in _iter_mtx_chunks(f, symmetric=symmetric):
            np.add.at(row_abs_sum, r, np.abs(v))
            if symmetric and np.any(off):
                np.add.at(row_abs_sum, c[off], np.abs(v[off]))
            on_diag = r == c
            if np.any(on_diag):
                np.add.at(diag, r[on_diag], v[on_diag])

    if np.any(row_abs_sum == 0.0):
        return True, "zero row after dropping exact zeros"

    z = diag == 0.0
    if not (z.any() and (~z).any()):
        return False, ""

    col_sum_z = np.zeros(rows, dtype=np.float64)
    with mtx_path.open("r", encoding="ascii", errors="replace") as f:
        banner, _comments, _rows, _cols, _nnz = _read_mtx_preamble(f)
        symmetric = "symmetric" in banner.lower()
        for r, c, v, off in _iter_mtx_chunks(f, symmetric=symmetric):
            if np.any(z[r] & z[c]):
                return False, ""
            m1 = off & z[c]
            if np.any(m1):
                np.add.at(col_sum_z, r[m1], v[m1])
            m2 = off & z[r]
            if np.any(m2):
                np.add.at(col_sum_z, c[m2], v[m2])

    if float(np.max(np.abs(col_sum_z))) == 0.0:
        return True, "exact constant-vector nullspace on zero-diagonal rows (inconsistent with b=1)"
    return False, ""


def build_manifest_cases(fetched_items: list[dict]) -> list[dict]:
    """Build sorted sealed_public manifest cases from fetched matrix metadata."""
    cases: list[dict] = []
    for item in fetched_items:
        tracks = ["recommended"] if item["matrix_type"] == "spd" else ["ipopt", "recommended"]
        for track in tracks:
            case = {
                "matrix_id": item["name"],
                "sequence_id": item["name"],
                "family_id": item["cls"],
                "source": f"SuiteSparse {item['group']}/{item['name']}",
                "mtx_path": item["mtx_path"],
                "matrix_sha256": item["matrix_sha256"],
                "n": item["n"],
                "nnz_lower": item["nnz_lower"],
                "matrix_type": item["matrix_type"],
                "class": item["cls"],
                "split": "sealed_public",
                "mode": "cold",
                "track": track,
                "expected_outcome": "Success",
            }
            if item.get("report_only"):
                case["report_only"] = True
            cases.append(case)

    cases.sort(
        key=lambda c: (
            CLASS_ORDER.index(c["class"]) if c["class"] in CLASS_ORDER else len(CLASS_ORDER),
            c["n"],
            c["matrix_id"],
            c["track"],
        )
    )
    return cases


def _archive_size(group: str, name: str) -> int | None:
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


def _load_ssstats_text(ssstats_arg: str | None) -> tuple[str, str]:
    if ssstats_arg:
        p = pathlib.Path(ssstats_arg)
        return p.read_text(encoding="utf-8", errors="replace"), str(p)
    req = urllib.request.Request(SSSTATS_URL, headers={"User-Agent": "smf-bench/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read().decode("utf-8", errors="replace"), SSSTATS_URL


def _resolve_path(path_str: str) -> pathlib.Path:
    p = pathlib.Path(path_str)
    return p if p.is_absolute() else (ROOT / p)


def _fetch_class_matrices(
    cls: str,
    candidates: list[dict],
    target_count: int,
    out_dir: pathlib.Path,
    budget_mb: float,
    spent_mb: float,
) -> tuple[list[dict], list[dict], float]:
    fetched: list[dict] = []
    skipped: list[dict] = []
    mtype = "spd" if cls == "public_spd" else "indefinite"

    for cand in candidates:
        if len(fetched) >= target_count:
            break
        group, name = cand["group"], cand["name"]
        mdir = out_dir / name
        mtx = mdir / f"{name}.mtx"

        if not mtx.exists():
            size = _archive_size(group, name)
            mb = (size or 0) / 1e6
            if size is not None and spent_mb + mb > budget_mb:
                skipped.append({
                    "group": group,
                    "name": name,
                    "class": cls,
                    "reason": f"archive {mb:.1f} MB would exceed budget {budget_mb:.1f} MB",
                })
                continue
            if mdir.exists():
                shutil.rmtree(mdir)
            mdir.mkdir(parents=True, exist_ok=True)
            archive = mdir / f"{name}.tar.gz"
            try:
                url = fs.download(group, name, archive)
                actual_mb = archive.stat().st_size / 1e6
                if size is None and spent_mb + actual_mb > budget_mb:
                    archive.unlink(missing_ok=True)
                    shutil.rmtree(mdir, ignore_errors=True)
                    skipped.append({
                        "group": group,
                        "name": name,
                        "class": cls,
                        "reason": f"downloaded archive {actual_mb:.1f} MB exceeded budget {budget_mb:.1f} MB",
                    })
                    continue
                with tarfile.open(archive, "r:gz") as tf:
                    member = next(
                        m for m in tf.getmembers()
                        if m.isfile() and (m.name == f"{name}.mtx" or m.name.endswith(f"/{name}.mtx"))
                    )
                    member.name = f"{name}.mtx"
                    tf.extract(member, mdir, filter="data")
                archive.unlink()
                spent_mb += mb if size is not None else actual_mb
                print(f"fetched {cls} {group}/{name} ({(mb if size is not None else actual_mb):.1f} MB) from {url}", flush=True)
            except Exception as exc:
                shutil.rmtree(mdir, ignore_errors=True)
                skipped.append({"group": group, "name": name, "class": cls, "reason": f"download/extract failed: {exc}"})
                continue

        try:
            banner, rows, cols, nnz = fs.read_header(mtx)
        except Exception as exc:
            skipped.append({"group": group, "name": name, "class": cls, "reason": f"header read failed: {exc}"})
            continue

        banner_lower = banner.lower()
        if "coordinate" not in banner_lower or "real" not in banner_lower or "symmetric" not in banner_lower or rows != cols:
            shutil.rmtree(mdir, ignore_errors=True)
            skipped.append({"group": group, "name": name, "class": cls, "reason": f"unexpected header '{banner}' {rows}x{cols}"})
            continue

        is_singular, reason = check_algebraic_singularity(mtx)
        if is_singular:
            print(f"  SINGULAR (report_only): {group}/{name} - {reason}", flush=True)

        mtx_rel = mtx.relative_to(ROOT).as_posix() if mtx.is_relative_to(ROOT) else mtx.as_posix()
        fetched.append({
            "group": group,
            "name": name,
            "cls": cls,
            "matrix_type": mtype,
            "mtx_path": mtx_rel,
            "matrix_sha256": fs.sha256_file(mtx),
            "n": rows,
            "nnz_lower": nnz,
            "report_only": is_singular,
            "singular_reason": reason,
        })

    return fetched, skipped, spent_mb


def _print_summary(selected: dict[str, list[dict]], ranked: dict[str, list[dict]]) -> None:
    total = sum(len(selected[c]) for c in CLASS_ORDER if c in selected)
    print(f"Dry-run summary: {total} matrices selected across {len(CLASS_ORDER)} classes")
    for cls in CLASS_ORDER:
        items = selected.get(cls, [])
        overflow = max(0, len(ranked.get(cls, [])) - len(items))
        print(f"\n[{cls}] selected={len(items)}/{TARGETS[cls][0]} (cap={TARGETS[cls][1]}, overflow={overflow})")
        for e in items:
            print(
                f"  {e['group']:16s} {e['name']:22s} n={e['nrows']:7d} "
                f"nnz={e['nnz']:10d} {e['kind']}"
            )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true", help="download matrices and write manifest")
    ap.add_argument("--ssstats", default=None, help="local ssstats.csv path (default: download from UF)")
    ap.add_argument("--fronttier", default="benchmarks/manifests/fronttier.json")
    ap.add_argument("--out", default="benchmarks/fixtures/public_suitesparse")
    ap.add_argument("--manifest", default="benchmarks/manifests/public_suitesparse.json")
    ap.add_argument("--budget-mb", type=float, default=2500.0)
    args = ap.parse_args(argv)

    fronttier_path = _resolve_path(args.fronttier)
    fronttier_manifest = json.loads(fronttier_path.read_text(encoding="utf-8"))
    excluded_names, excluded_stems = fronttier_exclusion_stems(fronttier_manifest)

    csv_text, ssstats_source = _load_ssstats_text(args.ssstats)
    ssstats_meta = parse_ssstats_meta(csv_text)
    entries = parse_ssstats(csv_text)
    ranked = rank_candidates(entries, excluded_names, excluded_stems, targets=TARGETS)
    selected = {cls: ranked[cls][: TARGETS[cls][0]] for cls in CLASS_ORDER}

    if not args.fetch:
        _print_summary(selected, ranked)
        return 0

    out_dir = _resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _resolve_path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    all_fetched: list[dict] = []
    all_skipped: list[dict] = []
    spent_mb = 0.0
    shortfalls: list[str] = []

    for cls in CLASS_ORDER:
        target_count, _cap = TARGETS[cls]
        got, skipped, spent_mb = _fetch_class_matrices(
            cls, ranked[cls], target_count, out_dir, args.budget_mb, spent_mb
        )
        all_fetched.extend(got)
        all_skipped.extend(skipped)
        if len(got) < target_count:
            shortfalls.append(f"{cls}: got {len(got)}/{target_count}")

    cases = build_manifest_cases(all_fetched)
    singular_flagged = [
        {"matrix_id": item["name"], "class": item["cls"], "reason": item["singular_reason"]}
        for item in all_fetched
        if item.get("report_only")
    ]
    manifest = {
        "schema_version": 1,
        "suite": "public_suitesparse",
        "selection_policy": {
            "ssstats_source": ssstats_source,
            "ssstats_meta": ssstats_meta,
            "fronttier_exclusion_source": str(args.fronttier),
            "min_n": MIN_N,
            "max_n": MAX_N,
            "max_nnz": MAX_NNZ,
            "excluded_kinds": list(EXCLUDED_KINDS),
            "kkt_kinds": list(KKT_KINDS),
            "series_prefixes": list(SERIES_PREFIXES) + [ND_SERIES_MARKER],
            "targets": {k: {"target": v[0], "per_group_cap": v[1]} for k, v in TARGETS.items()},
            "budget_mb": args.budget_mb,
            "downloaded_mb": round(spent_mb, 1),
            "singular_flagged": singular_flagged,
            "skipped_replacements": all_skipped,
        },
        "tracks": {
            "recommended": "pivot u = 0.01, default ordering",
            "ipopt": "pivot u = 1e-8 (bench_replay track), smf default ordering (AutoParallel + Predicted matching, as the plugin)",
        },
        "solvers": SOLVERS,
        "required_baselines": [],
        "cases": cases,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {manifest_path}: {len(all_fetched)} matrices, {len(cases)} case-tracks, "
        f"downloaded {spent_mb:.1f} MB"
    )
    if shortfalls:
        print("shortfalls: " + ", ".join(shortfalls), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
