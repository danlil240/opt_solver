#!/usr/bin/env python3
"""Reference inertia for the front-tier cases, computed once offline.

The scorer (score.py --inertia-ref) checks every solver's reported inertia
against this file instead of a vote among the solvers being scored.

Per matrix_id (both tracks factor the same matrix) the reference is the
inertia of the matrix the records report on: the case's matrix for a `cold`
case, the last replayed step of a `warm_sequence` (bench_replay --export-step).

  * spd cases: (n, 0, 0), exact from the manifest's matrix_type.
  * n <= --dense-max-n: dense eigenvalues (numpy eigvalsh).
  * larger: spectrum slicing with strict-pivoting sparse LDL^T. By Sylvester's
    law, neg(A + d I) = #{lambda < -d} and neg(A - d I) = #{lambda < d}, so
        neg  = neg(A + d I),   pos = n - neg(A - d I),   zero = n - pos - neg
    with d = --rel-shift, applied to the symmetrically equilibrated matrix
    D A D (same inertia; every row's max |entry| is 1). Each shifted matrix is factorized by smf
    (pivot_u 0.5, matching scaling) AND by MA57 (pivot_u 0.5, its default MC64
    scaling) through bench_replay; the reference is accepted only when both
    agree on both shifts, report no zero pivots and solve accurately, so it
    never rests on the subject solver alone. Disagreements are written with
    status "disputed" and carry both answers; the scorer skips them.
    Dense cases are sliced too, as a check of the slicing on known spectra.

Dense cases use the same equilibrated matrix and d. "zero" is the count of
eigenvalues of D A D within +-d of 0: a matrix with such a
cluster (e.g. stokes128's pressure modes) has no unique numerical inertia, and
the scorer reports a solver whose (pos, neg) differ from the reference only
inside that cluster as "consistent" rather than "exact".

Usage (MA57 needs SMF_HSL_LIBRARY and its DLL directory on PATH):
  python tools/bench/inertia_ref.py --manifest benchmarks/manifests/fronttier.json \
      --binary build/benchmarks/Release/bench_replay.exe \
      --output benchmarks/manifests/fronttier_inertia.json [--cases a,b] [--threads 1]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np

DEFAULT_REL_SHIFT = 1e-10
DEFAULT_DENSE_MAX_N = 6000
STRICT_PIVOT_U = 0.5


def read_mtx_lower(path: pathlib.Path) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """(n, row, col, val), 0-based, as bench_replay --export-step writes them
    (symmetric coordinate, lower triangle, no duplicates)."""
    with path.open("r", encoding="ascii") as fh:
        header = fh.readline()
        if "symmetric" not in header:
            raise ValueError(f"{path}: not a symmetric Matrix Market file")
        line = fh.readline()
        while line.startswith("%"):
            line = fh.readline()
        n, _, nnz = (int(x) for x in line.split())
        data = np.array(fh.read().split(), dtype=np.float64)
    if data.size != 3 * nnz:
        raise ValueError(f"{path}: expected {nnz} entries, read {data.size // 3}")
    data = data.reshape(nnz, 3)
    return n, data[:, 0].astype(np.int64) - 1, data[:, 1].astype(np.int64) - 1, data[:, 2].copy()


def equilibrate(n: int, row: np.ndarray, col: np.ndarray, val: np.ndarray,
                iters: int = 30) -> np.ndarray:
    """Symmetric Ruiz equilibration: D A D with every row's max |entry| ~ 1.
    A congruence, so the inertia is unchanged; it makes the shift d below
    scale-free (IPOPT KKT matrices carry barrier entries up to 1e12 next to
    O(1) ones, where a shift relative to max|a_ij| would be meaningless)."""
    v = val.copy()
    for _ in range(iters):
        m = np.zeros(n)
        a = np.abs(v)
        np.maximum.at(m, row, a)
        np.maximum.at(m, col, a)
        m[m == 0] = 1.0
        r = 1.0 / np.sqrt(m)
        v *= r[row] * r[col]
        if np.max(np.abs(m - 1.0)) < 1e-3:
            break
    return v


def equilibration_residual(n: int, row: np.ndarray, col: np.ndarray, val: np.ndarray) -> float:
    """max_i |max_j |a_ij| - 1| of an equilibrated matrix (0 = fully converged)."""
    m = np.zeros(n)
    a = np.abs(val)
    np.maximum.at(m, row, a)
    np.maximum.at(m, col, a)
    m[m == 0] = 1.0
    return float(np.max(np.abs(m - 1.0)))


def write_shifted_mtx(path: pathlib.Path, n: int, row: np.ndarray, col: np.ndarray,
                      val: np.ndarray, shift: float) -> None:
    """A + shift I as symmetric Matrix Market (missing diagonal entries added)."""
    diag = row == col
    v = val.copy()
    v[diag] += shift
    have = np.zeros(n, dtype=bool)
    have[row[diag]] = True
    miss = np.flatnonzero(~have)
    r = np.concatenate([row, miss]) + 1
    c = np.concatenate([col, miss]) + 1
    v = np.concatenate([v, np.full(miss.size, shift)])
    with path.open("w", encoding="ascii", newline="\n") as fh:
        fh.write("%%MatrixMarket matrix coordinate real symmetric\n")
        fh.write(f"{n} {n} {r.size}\n")
        np.savetxt(fh, np.column_stack([r, c, v]), fmt=["%d", "%d", "%.17g"])


def dense_counts(n: int, row, col, val, d: float) -> dict[str, Any]:
    A = np.zeros((n, n))
    A[row, col] = val
    A[col, row] = val
    lam = np.linalg.eigvalsh(A)
    neg = int(np.sum(lam < -d))
    pos = int(np.sum(lam > d))
    absl = np.sort(np.abs(lam))
    return {"pos": pos, "neg": neg, "zero": n - pos - neg,
            "min_abs_eig": float(absl[0]),
            # smallest |lambda| outside the zero cluster: how far the answer is from flipping
            "min_abs_eig_outside_cluster": float(absl[n - pos - neg]) if pos + neg else None}


def run_replay(binary: str, mtx: pathlib.Path, solver: str, threads: int,
               timeout: float, work: pathlib.Path) -> dict[str, Any]:
    out = work / f"rec_{solver}.jsonl"
    out.unlink(missing_ok=True)
    cmd = [binary, "--case", "inertia_ref", "--mtx", str(mtx), "--mtype", "indefinite",
           "--solver", solver, "--mode", "cold", "--track", "recommended",
           "--threads", str(threads), "--pivot-u", str(STRICT_PIVOT_U), "--JsonlOut", str(out)]
    if solver == "smf":
        cmd += ["--scaling", "matching"]
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS=str(threads))
    try:
        subprocess.run(cmd, env=env, timeout=timeout, capture_output=True, check=False)
    except subprocess.TimeoutExpired:
        return {"outcome": "Timeout"}
    if not out.exists():
        return {"outcome": "NoRecord"}
    rec = json.loads(out.read_text(encoding="utf-8").splitlines()[-1])
    return {"outcome": rec.get("outcome"), "inertia": rec.get("inertia"),
            "eta_inf": (rec.get("validation") or {}).get("eta_inf")}


def sliced_counts(binary: str, n: int, row, col, val, d: float, threads: int,
                  timeout: float, work: pathlib.Path, solvers: list[str]) -> dict[str, Any]:
    per: dict[str, Any] = {}
    for solver in solvers:
        res = {}
        for name, shift in (("plus", d), ("minus", -d)):
            f = work / f"shift_{name}.mtx"
            if not f.exists():
                write_shifted_mtx(f, n, row, col, val, shift)
            res[name] = run_replay(binary, f, solver, threads, timeout, work)
        ok = all(r.get("outcome") == "Success" and isinstance(r.get("inertia"), dict)
                 and int(r["inertia"]["zero"]) == 0 for r in res.values())
        entry: dict[str, Any] = {"ok": ok, "runs": res}
        if ok:
            neg = int(res["plus"]["inertia"]["neg"])            # #{lambda < -d}
            pos = n - int(res["minus"]["inertia"]["neg"])      # #{lambda > d}
            entry.update(pos=pos, neg=neg, zero=n - pos - neg)
        per[solver] = entry
    for name in ("plus", "minus"):
        (work / f"shift_{name}.mtx").unlink(missing_ok=True)
    return per


def export_step(binary: str, case: dict[str, Any], root: pathlib.Path,
                dest: pathlib.Path, timeout: float) -> None:
    cmd = [binary, "--case", case["matrix_id"], "--mode", case.get("mode", "cold"),
           "--mtype", case["matrix_type"], "--export-step", str(dest)]
    if case.get("mtx_path"):
        cmd += ["--mtx", str(root / case["mtx_path"])]
    if case.get("seq_values_path"):
        cmd += ["--seq-values", str(root / case["seq_values_path"])]
    subprocess.run(cmd, timeout=timeout, capture_output=True, check=True)


def reference_for(case: dict[str, Any], args, root: pathlib.Path) -> dict[str, Any]:
    n = int(case["n"])
    if case["matrix_type"] == "spd":
        return {"status": "ok", "method": "spd (manifest matrix_type)",
                "pos": n, "neg": 0, "zero": 0, "n": n}
    with tempfile.TemporaryDirectory(dir=args.workdir) as td:
        work = pathlib.Path(td)
        step = work / "step.mtx"
        export_step(args.binary, case, root, step, args.timeout)
        n2, row, col, val = read_mtx_lower(step)
        step.unlink()
        amax = float(np.max(np.abs(val))) if val.size else 0.0
        val = equilibrate(n2, row, col, val)
        d = args.rel_shift  # the equilibrated matrix has max |entry| 1 per row
        out: dict[str, Any] = {"n": n2, "mode": case.get("mode", "cold"),
                               "max_abs_entry": amax, "shift": d,
                               "equilibration_residual": equilibration_residual(n2, row, col, val)}
        solvers = [s for s in args.slice_solvers.split(",") if s]
        sl = sliced_counts(args.binary, n2, row, col, val, d, args.threads, args.timeout,
                           work, solvers)
        out["slices"] = sl
        good = [(e["pos"], e["neg"], e["zero"]) for e in sl.values() if e["ok"]]
        if n2 <= args.dense_max_n:
            dn = dense_counts(n2, row, col, val, d)
            out["dense"] = dn
            out.update(status="ok", method="dense eigvalsh",
                       pos=dn["pos"], neg=dn["neg"], zero=dn["zero"])
            out["slices_agree_with_dense"] = all(
                g == (dn["pos"], dn["neg"], dn["zero"]) for g in good) and len(good) == len(sl)
        elif len(sl) >= 2 and len(good) == len(sl) and len(set(good)) == 1:
            out.update(status="ok", method="spectrum slicing (" + " + ".join(sl) + ", u=0.5)",
                       pos=good[0][0], neg=good[0][1], zero=good[0][2])
        else:
            out.update(status="disputed", method="spectrum slicing: solvers disagree or failed")
        return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--binary", required=True, help="bench_replay with --export-step")
    ap.add_argument("--output", required=True, help="JSON written (and updated in place)")
    ap.add_argument("--root", default=".", help="directory the manifest's paths are relative to")
    ap.add_argument("--cases", default="", help="comma-separated matrix_id filter")
    ap.add_argument("--rel-shift", type=float, default=DEFAULT_REL_SHIFT)
    ap.add_argument("--dense-max-n", type=int, default=DEFAULT_DENSE_MAX_N)
    ap.add_argument("--slice-solvers", default="smf,ma57")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args(argv)

    root = pathlib.Path(args.root)
    manifest = json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))
    want = {c for c in args.cases.split(",") if c}
    outp = pathlib.Path(args.output)
    ref = json.loads(outp.read_text(encoding="utf-8")) if outp.exists() else {}
    ref.update(schema_version=1, manifest=pathlib.Path(args.manifest).name,
               rel_shift=args.rel_shift, strict_pivot_u=STRICT_PIVOT_U,
               generated=datetime.date.today().isoformat())
    cases = ref.setdefault("cases", {})
    seen: dict[str, dict[str, Any]] = {}
    for c in manifest["cases"]:
        mid = c["matrix_id"]
        if mid in seen:
            # One reference per matrix_id: every track must replay the same matrix.
            a = seen[mid]
            if (a.get("mode"), a.get("mtx_path"), a.get("seq_values_path")) != (
                    c.get("mode"), c.get("mtx_path"), c.get("seq_values_path")):
                raise SystemExit(f"error: {mid} replays different matrices on different tracks")
            continue
        seen[mid] = c
        if want and mid not in want:
            continue
        try:
            cases[mid] = reference_for(c, args, root)
        except Exception as e:  # keep going; the case is recorded as failed
            cases[mid] = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
        r = cases[mid]
        print(f"{mid}: {r.get('status')} {r.get('method', '')} "
              f"{(r.get('pos'), r.get('neg'), r.get('zero'))}", flush=True)
        outp.write_text(json.dumps(ref, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
