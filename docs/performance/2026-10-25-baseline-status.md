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

## 2026-10-25 (evening) — Optimization batch + first KKT/symmetric corpus results

Build: `build/` (MSVC Release, MKL sequential auto-detected from `smfdeps` env, METIS/AMD/CHOLMOD/MUMPS linked). **56/56 CTest pass.**

### Front-kernel blocked update (`bench_front_kernel`, 50 reps, MKL `cblas_dgemm`)

| Front (f x p) | reference | blocked best | speedup |
|---|---:|---:|---:|
| 96 x 64   | 0.049 ms | 0.046 ms (nb=8)  | 1.06x |
| 384 x 256 | 1.607 ms | 0.949 ms (nb=16) | 1.69x |
| 768 x 512 | 12.43 ms | 5.740 ms (nb=16) | 2.17x |

`ctrl.indef_panel_size` default changed 32 -> 16 (best measured on this host).

### Frozen manifests populated

- `benchmarks/manifests/kkt.json`: 10 warm_sequence indefinite KKT fixtures (trajectory-opt/control families, n=320-420) in `benchmarks/fixtures/kkt_ocp/` with `.rhs` files + sha256.
- `benchmarks/manifests/symmetric.json`: 10 cold cases (SPD poisson/banded/tridiag + 2 indefinite banded) in `benchmarks/fixtures/symmetric/`.

### KKT suite — `tools/bench/run.py` (10 trials/case, warm_sequence)

**geomean speedup 1.705x (95% CI 1.648-1.764), win rate 10/10, coverage 1.0**, `eta_inf`/`omega` all within gates. Fastest baseline on every case: `mkl_pardiso`; `mumps` also 10/10 valid.

### Symmetric suite — cold mode (10 trials/case)

geomean 1.073x. smf wins indefinite banded (**2.57-3.10x** vs mumps/mkl_pardiso), ties/wins banded+small poisson (0.98-1.08x vs cholmod), loses tridiag (0.68-0.71x) and poisson2d_50 (0.66x) to CHOLMOD's specialized paths.

### Ordering policy refinement

- `use_identity_ordering` now requires `dense_band` (`(bw+1)*n <= 2*nnz_lower`) **or** `n <= 2048`, so large sparse-band matrices (e.g. 2D Poisson, bw=63, ~5 nnz/col) take the fill-reducing backend instead of identity.
- `AutoParallel` uses METIS only for `n > 8192`; below that AMD is cheaper and better fill-wise on measured workloads (Poisson2D_4000 factor: AMD 0.86 vs METIS 1.11 vs identity 1.56 ms).

### Fixes during optimization batch

- `filled_col_counts` (supernode detection): per-row etree path-walk with per-i mark dedup — exact `|L*w|` in O(nnz + path lengths); replaces both a wrong-direction ancestor walk and an O(nnz_L)-memory set-merge.
- `predicted_factor_entries` semantic reverted to sum of front sizes (test-pinned).
- Stale-exe hazard noted: `scratch_driver.exe` locked by zombie process masked earlier PARDISO debugging.

### Still blocked

HSL MA27/57/86/87/97, Panua PARDISO, SPRAL SSIDS — `Unavailable` (licence + Fortran, Linux host per spec). G2 not attempted: needs frozen-corpus qualification run on the Linux host with required baselines.
