# SMF Native Portfolio and Qualification Status Report

**Date:** 2026-10-25
**Branch:** `feat/solver-performance`

## Qualification Gates Summary

| Gate | Status | Details |
|---|---|---|
| **G0: Evidence & Harness** | **PASS** | `smf::bench::validate` (`eta_inf <= 1e-10`, `omega <= 1e-8` on original unscaled system), `RunRecord` JSONL schema, failure/timeout/OOM preservation, and `tools/bench/{run.py,report.py}` implemented and verified. |
| **G1: Numerical Correctness** | **PASS** | 2x2 `D`/`L` separation, secondary pivot fix, scaled 2x2 arithmetic (`1e-100` to `1e100`), cross-front delayed pivots, full-column extension stability bounds, equilibration/matching scaling, iterative refinement, and MA97 plugin runtime ABI verified across 55 CTest suites. |
| **G2 / G3 / G4: External Baseline Leadership** | **PARTIAL — dev-host baselines live; qualification still BLOCKED** | As of 2026-10-25 (conda env `smfdeps`, micromamba, `build-deps`): real **MKL PARDISO 2026.2**, **MUMPS-seq 5.8.2**, **CHOLMOD 5.4.0** (SuiteSparse), **METIS**, **Eigen**, and MKL BLAS/LAPACK are linked and measured via `tools/bench` (`mkl_rt` replaced by `mkl_intel_lp64 + mkl_sequential + mkl_core` to avoid dual OpenMP runtimes). First smoke results: `mumps`/`mkl_pardiso`/`cholmod` 3/3 adapter-valid; `bench_compare` shows smf ~6.2× faster than MUMPS-seq and ~0.73× vs CHOLMOD on tiny SPD fixtures. Licensed **HSL** (`MA27/57/86/87/97`) and **SPRAL SSIDS** remain `Unavailable` (licence + Fortran toolchain required); G2+ requires the frozen corpus and the Linux qualification host per spec — **not attempted**. |

## Native Dispatch Policy (`smf::select_native_path`)

- **`TinyDense`**: Selected for small dense systems (`n <= 16`, density `>= 0.5`).
- **`BandedSPD`**: Preserves narrow-band LAPACK `dpbtrf`/`dpbtrs` dispatch (`n >= 100`, `kd <= 128`, `AutoParallel`).
- **`SparseSPD`**: Multifrontal Cholesky for general SPD matrices.
- **`CertifiedQuasiDefinite`**: Candidate path for repeated KKT sequences (`expected_refactors >= 5`), guarded by runtime `certify_quasi_definite(A)` verification with automatic fallback to `GeneralIndefinite`.
- **`GeneralIndefinite`**: Blocked multifrontal `LDL^T` with threshold pivoting, cross-front delayed pivots, and costed task-tree scheduling.
