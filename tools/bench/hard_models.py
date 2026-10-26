"""Hard nonconvex NLP corpus for IPOPT (IPM-mode work).

MODELS maps name -> (builder, arg) in the same shape as ipopt_kkt.MODELS;
ipopt_kkt.build_nl calls builder(arg), scaling arg only when it is numeric.

PGLib-OPF AC-OPF cases (v23.07) are built with Egret's polar-power-voltage
formulation (egret.models.acopf.create_psv_acopf_model) from the MATPOWER .m
files fetched by tools/bench/fetch_pglib.py into benchmarks/fixtures/pglib/.

Dependency: gridx-egret 0.6.2 (pip install gridx-egret into the smfbench env).
"""

from __future__ import annotations

import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
PGLIB_DIR = REPO / "benchmarks" / "fixtures" / "pglib"


def acopf_pglib(case_name: str):
    """Egret PSV AC-OPF Pyomo model for a PGLib-OPF case (e.g. pglib_opf_case118_ieee)."""
    from egret.data.model_data import ModelData
    from egret.models.acopf import create_psv_acopf_model

    path = PGLIB_DIR / f"{case_name}.m"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run tools/bench/fetch_pglib.py")
    model_data = ModelData.read(str(path))
    model, _md = create_psv_acopf_model(model_data)
    return model


PGLIB_CASES = [
    "pglib_opf_case118_ieee",
    "pglib_opf_case300_ieee",
    "pglib_opf_case500_goc__api",
    "pglib_opf_case1354_pegase",
    "pglib_opf_case2000_goc__sad",
    "pglib_opf_case2869_pegase",
    "pglib_opf_case4661_sdet",
    "pglib_opf_case9241_pegase",
]

MODELS = {name: (acopf_pglib, name) for name in PGLIB_CASES}
