#!/usr/bin/env python3
"""Fetch PGLib-OPF v23.07 MATPOWER case files used by the hard-model corpus.

Source: https://github.com/power-grid-lib/pglib-opf at tag v23.07. The .m
files are data only (never executed; Egret's MATPOWER parser reads them). Each
file is downloaded into benchmarks/fixtures/pglib/ (gitignored) and verified
against the sha256 frozen in CASES. Re-running is a no-op when the files exist
with matching hashes; a file with a wrong hash is re-downloaded and, if it
still mismatches, the script fails.

Usage:
  python tools/bench/fetch_pglib.py [--out benchmarks/fixtures/pglib]
      [--only name,name] [--print-hashes]
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys
import urllib.request

TAG = "v23.07"
RAW = "https://raw.githubusercontent.com/power-grid-lib/pglib-opf/" + TAG + "/{path}.m"

# case name -> (subdirectory in the repo, sha256 of the .m file).
CASES = {
    "pglib_opf_case118_ieee": ("", "b1af0833849040c04babc3700631cff0d9afa66b79c5d3e13ae79bdf516cec78"),
    "pglib_opf_case300_ieee": ("", "7ecf056d5942135765200ad7ae8791c28f0d35fb1dc888ba2c32dfc950f3c2f5"),
    "pglib_opf_case500_goc__api": ("api", "25b7f775f74f4220409d9a562da0dbf3b78eaa7c31931b4f9da0b9d65e1bf9ae"),
    "pglib_opf_case1354_pegase": ("", "cd6d27dff4a56684f1e4f82cfa346b36d84c4e90733228aa88331cd550e17652"),
    "pglib_opf_case2000_goc__sad": ("sad", "f42600f66cf97b03d9f114a2680ead3a3f6e2e5f1030ba44cfd4cc9552e919d0"),
    "pglib_opf_case2869_pegase": ("", "6c8e80fba6fc2fa78d65fce64cf4801425b01a0aa093661caf581b6551d4a7ac"),
    "pglib_opf_case4661_sdet": ("", "d24d1f61bdd2c7b9ecde796a88cbfcf178247e04dc7048ae0e043ec893a70c84"),
    "pglib_opf_case9241_pegase": ("", "d55833986cc4e3e417cad93458f39810e1ed6a54aaa2a979200b0232f271474d"),
}


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name: str, subdir: str, dest: pathlib.Path) -> str:
    url = RAW.format(path=f"{subdir}/{name}" if subdir else name)
    req = urllib.request.Request(url, headers={"User-Agent": "smf-bench/1.0"})
    tmp = dest.with_suffix(".m.part")
    try:
        with urllib.request.urlopen(req, timeout=120) as r, tmp.open("wb") as f:
            f.write(r.read())
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return url


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/fixtures/pglib")
    ap.add_argument("--only", default="")
    ap.add_argument("--print-hashes", action="store_true",
                    help="print the sha256 of each file instead of verifying (used to freeze CASES)")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    only = {s for s in args.only.split(",") if s}
    failures = []
    for name, (subdir, want) in CASES.items():
        if only and name not in only:
            continue
        dest = out / f"{name}.m"
        if args.print_hashes:
            if not dest.exists():
                download(name, subdir, dest)
            print(f'    "{name}": ("{subdir}", "{sha256_file(dest)}"),')
            continue
        if dest.exists() and sha256_file(dest) == want:
            print(f"ok      {name}", flush=True)
            continue
        try:
            url = download(name, subdir, dest)
            got = sha256_file(dest)
            if got != want:
                dest.unlink(missing_ok=True)
                raise ValueError(f"sha256 mismatch: got {got}, want {want}")
            print(f"fetched {name} from {url}", flush=True)
        except Exception as exc:
            failures.append(name)
            print(f"FAILED  {name}: {exc}", file=sys.stderr, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
