#!/usr/bin/env python3
"""Fetch a curated set of large SuiteSparse matrices and write a benchmark manifest.

The matrices (n ~ 10^4 - 10^6) are the symmetric sets sparse direct solver
papers benchmark on: GHS_indef optimization / saddle-point KKT systems and
classic SPD problems. They are too large to commit, so this script is the
reproducible source: each archive is downloaded into its own directory under
benchmarks/fixtures/large/, extracted with tarfile's 'data' filter, checked
for a real symmetric Matrix Market header, and recorded in the manifest with
its sha256.

Usage:
  python tools/bench/fetch_suitesparse.py [--out benchmarks/fixtures/large]
      [--manifest benchmarks/manifests/large.json] [--only name,name]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys
import tarfile
import urllib.request

# (group, name, matrix_type, family). Types follow the collection's
# "positive definite" flag; families group results in the report.
CATALOG = [
    # Optimization KKT / saddle point (symmetric indefinite).
    ("GHS_indef", "cvxqp3", "indefinite", "kkt_qp"),
    ("GHS_indef", "ncvxqp9", "indefinite", "kkt_qp"),
    ("GHS_indef", "ncvxqp5", "indefinite", "kkt_qp"),
    ("GHS_indef", "ncvxqp3", "indefinite", "kkt_qp"),
    ("GHS_indef", "aug3dcqp", "indefinite", "kkt_qp"),
    ("GHS_indef", "blockqp1", "indefinite", "kkt_qp"),
    ("GHS_indef", "boyd1", "indefinite", "kkt_qp"),
    ("GHS_indef", "boyd2", "indefinite", "kkt_qp"),
    ("GHS_indef", "cont-201", "indefinite", "kkt_qp"),
    ("GHS_indef", "cont-300", "indefinite", "kkt_qp"),
    ("GHS_indef", "d_pretok", "indefinite", "kkt_pde"),
    ("GHS_indef", "turon_m", "indefinite", "kkt_pde"),
    ("GHS_indef", "stokes128", "indefinite", "kkt_pde"),
    ("GHS_indef", "mario002", "indefinite", "kkt_pde"),
    # Other symmetric indefinite.
    ("GHS_indef", "helm3d01", "indefinite", "indef_pde"),
    ("GHS_indef", "bratu3d", "indefinite", "indef_pde"),
    ("GHS_indef", "copter2", "indefinite", "indef_pde"),
    ("GHS_indef", "dawson5", "indefinite", "indef_pde"),
    # Symmetric positive definite.
    ("Mulvey", "finan512", "spd", "spd_opt"),
    ("GHS_psdef", "oilpan", "spd", "spd_struct"),
    ("Schmid", "thermal1", "spd", "spd_pde"),
    ("Rothberg", "cfd2", "spd", "spd_pde"),
    ("UTEP", "Dubcova3", "spd", "spd_pde"),
    ("GHS_psdef", "bmwcra_1", "spd", "spd_struct"),
    ("Boeing", "pwtk", "spd", "spd_struct"),
    ("ND", "nd24k", "spd", "spd_dense"),
    ("Schenk_AFE", "af_shell7", "spd", "spd_struct"),
    ("GHS_psdef", "apache2", "spd", "spd_pde"),
    ("Schmid", "thermal2", "spd", "spd_pde"),
    ("AMD", "G3_circuit", "spd", "spd_circuit"),
]

MIRRORS = [
    "https://www.cise.ufl.edu/research/sparse/MM/{group}/{name}.tar.gz",
    "https://sparse.tamu.edu/MM/{group}/{name}.tar.gz",
]


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(group: str, name: str, dest: pathlib.Path) -> str:
    last_err = ""
    for tmpl in MIRRORS:
        url = tmpl.format(group=group, name=name)
        req = urllib.request.Request(url, headers={"User-Agent": "smf-bench/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r, dest.open("wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            with dest.open("rb") as f:
                if f.read(2) != b"\x1f\x8b":
                    raise ValueError("not a gzip archive")
            return url
        except Exception as exc:  # try the next mirror
            last_err = f"{url}: {exc}"
            dest.unlink(missing_ok=True)
    raise RuntimeError(last_err)


def read_header(mtx: pathlib.Path) -> tuple[str, int, int, int]:
    with mtx.open("r", encoding="ascii", errors="replace") as f:
        banner = f.readline().strip()
        line = f.readline()
        while line.startswith("%"):
            line = f.readline()
        rows, cols, nnz = (int(t) for t in line.split()[:3])
    return banner, rows, cols, nnz


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/fixtures/large")
    ap.add_argument("--manifest", default="benchmarks/manifests/large.json")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    only = {s for s in args.only.split(",") if s}
    cases = []
    failures = []
    for group, name, mtype, family in CATALOG:
        if only and name not in only:
            continue
        mdir = out / name
        mtx = mdir / f"{name}.mtx"
        if not mtx.exists():
            # Fresh directory per archive; nothing from it is executed.
            if mdir.exists():
                shutil.rmtree(mdir)
            mdir.mkdir(parents=True)
            archive = mdir / f"{name}.tar.gz"
            try:
                url = download(group, name, archive)
                with tarfile.open(archive, "r:gz") as tf:
                    member = next(m for m in tf.getmembers()
                                  if m.isfile() and m.name.endswith(f"/{name}.mtx"))
                    member.name = f"{name}.mtx"
                    tf.extract(member, mdir, filter="data")
                archive.unlink()
                print(f"fetched {group}/{name} from {url}", flush=True)
            except Exception as exc:
                failures.append(f"{group}/{name}: {exc}")
                print(f"FAILED {group}/{name}: {exc}", flush=True)
                continue
        banner, rows, cols, nnz = read_header(mtx)
        if "coordinate" not in banner or "symmetric" not in banner or rows != cols:
            failures.append(f"{group}/{name}: unexpected header '{banner}' {rows}x{cols}")
            continue
        cases.append({
            "matrix_id": name,
            "sequence_id": name,
            "family_id": family,
            "source": f"SuiteSparse {group}/{name}",
            "mtx_path": mtx.as_posix(),
            "matrix_sha256": sha256_file(mtx),
            "n": rows,
            "nnz_lower": nnz,
            "matrix_type": mtype,
            "split": "test",
            "mode": "cold",
            "track": "recommended",
            "expected_outcome": "Success",
        })
        print(f"  {name:14s} n={rows:9d} nnz={nnz:10d} {mtype}", flush=True)

    cases.sort(key=lambda c: (c["n"], c["matrix_id"]))
    manifest = {
        "schema_version": 1,
        "suite": "large",
        "solvers": ["smf", "ma27", "mumps", "mkl_pardiso", "cholmod", "ma57", "ma97", "ssids"],
        "required_baselines": [],
        "cases": cases,
    }
    pathlib.Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.manifest} with {len(cases)} cases")
    if failures:
        print("failures:\n  " + "\n  ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
