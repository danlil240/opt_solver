"""Python distribution helper for the smf sparse symmetric direct solver.

Provides library discovery, ctypes metadata inspection, and ready-to-use
option dictionaries for native IPOPT (linear_solver="smf"), cyipopt, and CasADi.
"""

from __future__ import annotations

import ctypes
import os
import pathlib
import sys
from typing import Any, Dict, Optional

__all__ = [
    "c_library_path",
    "ma97_plugin_path",
    "abi_version",
    "version",
    "blas_backend",
    "ipopt_options",
    "casadi_options",
]


def _lib_names(stem: str) -> list[str]:
    if sys.platform == "win32":
        return [f"{stem}.dll", f"lib{stem}.dll"]
    if sys.platform == "darwin":
        return [f"lib{stem}.dylib", f"lib{stem}.1.dylib"]
    return [f"lib{stem}.so", f"lib{stem}.so.1"]


def _search_dirs() -> list[pathlib.Path]:
    dirs: list[pathlib.Path] = []
    env_dir = os.environ.get("SMF_LIB_DIR")
    if env_dir:
        dirs.append(pathlib.Path(env_dir))
    pkg_dir = pathlib.Path(__file__).resolve().parent
    dirs.append(pkg_dir / "_libs")
    # Conda / system prefix fallbacks
    prefix = pathlib.Path(sys.prefix)
    if sys.platform == "win32":
        dirs.append(prefix / "Library" / "bin")
    else:
        dirs.append(prefix / "lib")
        dirs.append(prefix / "lib64")
    return dirs


def _find_library(stem: str, env_override: str) -> str:
    explicit = os.environ.get(env_override)
    if explicit:
        return explicit
    for d in _search_dirs():
        for name in _lib_names(stem):
            candidate = d / name
            if candidate.is_file():
                return str(candidate)
    # Fall back to bare library name so dynamic linker search path can resolve it.
    return _lib_names(stem)[0]


def c_library_path() -> str:
    """Return the path to libsmf_c (versioned C ABI for native linear_solver=smf)."""
    return _find_library("smf_c", "SMF_C_LIB")


def ma97_plugin_path() -> str:
    """Return the path to libsmf_ma97 (drop-in HSL MA97 shim for stock IPOPT)."""
    return _find_library("smf_ma97", "SMF_MA97_LIB")


def _load_c_lib() -> ctypes.CDLL:
    path = c_library_path()
    dir_name = os.path.dirname(path)
    if (
        sys.platform == "win32"
        and dir_name
        and hasattr(os, "add_dll_directory")
        and os.path.isdir(dir_name)
    ):
        with os.add_dll_directory(os.path.abspath(dir_name)):
            lib = ctypes.CDLL(path)
    else:
        lib = ctypes.CDLL(path)
    lib.smf_c_abi_version.restype = ctypes.c_int
    lib.smf_c_abi_version.argtypes = []
    lib.smf_version.restype = ctypes.c_char_p
    lib.smf_version.argtypes = []
    lib.smf_blas_backend.restype = ctypes.c_char_p
    lib.smf_blas_backend.argtypes = []
    return lib


def abi_version() -> int:
    """Return SMF_C_ABI_VERSION exported by libsmf_c."""
    return int(_load_c_lib().smf_c_abi_version())


def version() -> str:
    """Return the smf version string exported by libsmf_c."""
    return _load_c_lib().smf_version().decode("utf-8")


def blas_backend() -> str:
    """Return the BLAS backend string (e.g. 'openblas', 'mkl') of libsmf_c."""
    return _load_c_lib().smf_blas_backend().decode("utf-8")


def ipopt_options(
    *,
    native: bool = True,
    num_threads: int = 0,
    scaling: str = "auto",
    ordering: str = "auto",
    deterministic: bool = False,
    pass_lib_path: bool = True,
) -> Dict[str, Any]:
    """Build an IPOPT / cyipopt options dict configured for smf.

    Args:
        native: True for IPOPT built with native linear_solver=smf; False for
            stock IPOPT wheels (uses linear_solver='ma97' + hsllib=libsmf_ma97).
        num_threads: 0 for auto (SMF_NUM_THREADS / OMP_NUM_THREADS / P-cores).
        scaling: 'auto', 'dynamic', 'none', 'matching', or 'equilibration'.
        ordering: 'auto', 'amd', 'metis', or 'matching'.
        deterministic: True for bitwise-reproducible parallel solves.
        pass_lib_path: When native=True, include 'smflib' pointing to libsmf_c
            (set False if IPOPT is compile-time linked with IPOPT_HAS_SMF).
    """
    if native:
        opts: Dict[str, Any] = {
            "linear_solver": "smf",
            "smf_num_threads": int(num_threads),
            "smf_scaling": scaling,
            "smf_ordering": ordering,
            "smf_deterministic": "yes" if deterministic else "no",
        }
        if pass_lib_path:
            opts["smflib"] = c_library_path()
        return opts

    return {
        "linear_solver": "ma97",
        "hsllib": ma97_plugin_path(),
    }


def casadi_options(
    *,
    native: bool = True,
    num_threads: int = 0,
    scaling: str = "auto",
    ordering: str = "auto",
    deterministic: bool = False,
    pass_lib_path: bool = True,
    extra_ipopt: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a CasADi nlpsol('...', 'ipopt', nlp, opts) dictionary for smf."""
    ipopt_opts = ipopt_options(
        native=native,
        num_threads=num_threads,
        scaling=scaling,
        ordering=ordering,
        deterministic=deterministic,
        pass_lib_path=pass_lib_path,
    )
    if extra_ipopt:
        ipopt_opts.update(extra_ipopt)
    return {"ipopt": ipopt_opts}
