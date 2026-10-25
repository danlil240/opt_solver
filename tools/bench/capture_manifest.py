"""Build a frozen benchmark manifest from captured KKT fixtures or Matrix Market directories."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any, Dict, List, Optional


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(
    capture_dir: pathlib.Path,
    family_id: str = "captured_kkt",
    split: str = "train",
) -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    if capture_dir.exists():
        for mtx in sorted(capture_dir.glob("*.mtx")):
            stem = mtx.stem
            rhs = mtx.with_suffix(".rhs")
            cases.append(
                {
                    "matrix_id": stem,
                    "sequence_id": f"{family_id}_seq",
                    "family_id": family_id,
                    "mtx_path": str(mtx),
                    "matrix_sha256": sha256_file(mtx),
                    "rhs_path": str(rhs) if rhs.exists() else None,
                    "rhs_sha256": sha256_file(rhs) if rhs.exists() else None,
                    "split": split,
                    "mode": "warm_sequence",
                    "track": "recommended",
                    "expected_outcome": "Success",
                }
            )
    return {
        "schema_version": 1,
        "suite": "captured",
        "solvers": ["smf", "ma27", "ma57", "ma86", "ma97"],
        "required_baselines": ["ma57", "ma97"],
        "cases": cases,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate benchmark manifest from capture directory.")
    parser.add_argument("--input-dir", required=True, help="Directory with captured .mtx/.rhs files")
    parser.add_argument("--output", required=True, help="Output manifest JSON path")
    parser.add_argument("--family", default="captured_kkt", help="Application family ID")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    args = parser.parse_args(argv)

    manifest = build_manifest(pathlib.Path(args.input_dir), args.family, args.split)
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
