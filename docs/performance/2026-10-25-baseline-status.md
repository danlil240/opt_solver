# SMF Native Portfolio and Qualification Status Report

**Date:** 2026-10-25
**Branch:** `feat/solver-performance`

## Qualification Gates Summary

| Gate | Status | Details |
|---|---|---|
| **G0: Evidence & Harness** | **PASS** | `smf::bench::validate` (`eta_inf <= 1e-10`, `omega <= 1e-8` on original unscaled system), `RunRecord` JSONL schema, failure/timeout/OOM preservation, and `tools/bench/{run.py,report.py}` implemented and verified. |
| **G1: Numerical Correctness** | **PASS** | 2x2 `D`/`L` separation, secondary pivot fix, scaled 2x2 arithmetic (`1e-100` to `1e100`), cross-front delayed pivots, full-column extension stability bounds, equilibration/matching scaling, iterative refinement, and MA97 plugin runtime ABI verified across 55 CTest suites. |
| **G2 / G3 / G4: External Baseline Leadership** | **BLOCKED (Baselines Unavailable on Host)** | External licensed HSL (`MA27`, `MA57`, `MA86`, `MA87`, `MA97`), `SPRAL SSIDS`, `MUMPS`, `PARDISO`, and `CHOLMOD` libraries are not installed on the current Windows workstation; BLAS is running in `SMF_BLAS_REFERENCE=1` mode. Per the spec and plan, missing required baselines block external leadership claims and are explicitly reported as `Unavailable`. |

## Native Dispatch Policy (`smf::select_native_path`)

- **`TinyDense`**: Selected for small dense systems (`n <= 16`, density `>= 0.5`).
- **`BandedSPD`**: Preserves narrow-band LAPACK `dpbtrf`/`dpbtrs` dispatch (`n >= 100`, `kd <= 128`, `AutoParallel`).
- **`SparseSPD`**: Multifrontal Cholesky for general SPD matrices.
- **`CertifiedQuasiDefinite`**: Candidate path for repeated KKT sequences (`expected_refactors >= 5`), guarded by runtime `certify_quasi_definite(A)` verification with automatic fallback to `GeneralIndefinite`.
- **`GeneralIndefinite`**: Blocked multifrontal `LDL^T` with threshold pivoting, cross-front delayed pivots, and costed task-tree scheduling.
