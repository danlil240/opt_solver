# SMF Native Portfolio and Qualification Status Report

**Date:** 2026-10-25
**Branch:** `feat/solver-performance`

## Qualification Gates Summary

| Gate | Status | Details |
|---|---|---|
| **G0: Evidence & Harness** | **PASS** | `smf::bench::validate` (`eta_inf <= 1e-10`, `omega <= 1e-8` on original unscaled system), `RunRecord` JSONL schema, failure/timeout/OOM preservation, and `tools/bench/{run.py,report.py}` implemented and verified. |
| **G1: Numerical Correctness** | **PASS** | 2x2 `D`/`L` separation, secondary pivot fix, scaled 2x2 arithmetic (`1e-100` to `1e100`), cross-front delayed pivots, full-column extension stability bounds, equilibration/matching scaling, iterative refinement, and MA97 plugin runtime ABI verified across 55 CTest suites. |
| **G2 / G3 / G4: External Baseline Leadership** | **PARTIAL — dev-host baselines live; qualification still BLOCKED** | As of 2026-10-25 (conda env `smfdeps`, micromamba, `build-deps`): real **MKL PARDISO 2026.2**, **MUMPS-seq 5.8.2**, **CHOLMOD 5.4.0** (SuiteSparse), **METIS**, **Eigen**, and MKL BLAS/LAPACK are linked and measured via `tools/bench` (`mkl_rt` replaced by `mkl_intel_lp64 + mkl_sequential + mkl_core` to avoid dual OpenMP runtimes). First smoke results: `mumps`/`mkl_pardiso`/`cholmod` 3/3 adapter-valid; `bench_compare` shows smf ~6.2× faster than MUMPS-seq and ~0.73× vs CHOLMOD on tiny SPD fixtures. **CoinHSL MA27 added later same day (see below)** — now the strongest baseline and beating smf on the current corpus. Commercial **HSL** (`MA57/77/86/87/97`) and **SPRAL SSIDS** remain `Unavailable` (licence + Fortran toolchain required); G2+ requires the frozen corpus and the Linux qualification host per spec — **not attempted**. |

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

HSL MA57/77/86/87/97, Panua PARDISO, SPRAL SSIDS — `Unavailable` (licence + Fortran, Linux host per spec). G2 not attempted: needs frozen-corpus qualification run on the Linux host with required baselines.

## 2026-10-25 (late) — CoinHSL MA27 baseline live; leaderboard corrected

`C:\Users\daniel\Downloads\CoinHSL-archive.v2024.5.15.x86_64-w64-mingw32-libgfortran5` supplies the free legacy HSL codes (MA27/MA28/MA30/MC19–24; **no MA57/77/86/87/97**). MSVC cannot consume the `.dll.a` import libs, so `lib/coinhsl.lib` was generated via `lib.exe /DEF:coinhsl_msvc.def /MACHINE:X64` (all 100 exports). CMake gained `COINHSL_ROOT` discovery, `SMF_HAS_MA27_IMPL` propagates through `smf_bench_baselines` (so `bench_replay`/`run.py` see it, not just `bench_compare`), and `smf_stage_runtime_dlls` stages the MinGW runtime stack (`libcoinhsl.dll`, `libopenblas.dll`, `libgfortran-5.dll`, `libgomp-1.dll`, ...). `hsl.cpp` now has a real `Ma27Adapter` (analyse=`ma27ad_`, factor=`ma27bd_` with O(nnz) value rescatter since MA27 overwrites `A[]` in place, solve=`ma27cd_` per RHS).

`INFO` layout verified empirically (`scratch` probe, since deleted): `INFO(1)`=0 ok / <0 arg-or-workspace error / **3 = singular detected**; `INFO(15)` = #negative eigenvalues → on success `pos = n − neg`, `zero = 0`. Singular `diag(2,0)` probe returns `info(1)=3` — honest singularity detection, unlike MKL PARDISO's silent regularization.

**Result: MA27 is the strongest baseline in the pool and currently beats smf.**

- **KKT suite** (10 warm_sequence indefinite, n=320–420): geomean speedup dropped to **0.348× (CI 0.341–0.354), win 0/10** — MA27 wins every case (wall ~0.14ms vs smf ~0.40ms on `kkt_ocp_lqr`; per-step factor 0.098ms vs 0.319ms). Prior 1.705× figure was vs mkl_pardiso/mumps only.
- **Symmetric suite** (10 cold): geomean **0.60×**, MA27 wins 9/10 (smf ties `poisson2d_20` only), beating CHOLMOD on banded/tridiag too.
- All 100 MA27 records + 500 per-update validations pass `eta_inf ≤ 1e-10`, `omega ≤ 1e-8`. Inertia matches smf exactly (240/120/0 on `kkt_ocp_lqr`).
- `bench_kkt_fixture_compare` (in-process): MA27 factor 0.047ms vs smf 0.162ms — ~3.4×. MA27 uses 317 tiny steps (maxfront=5); smf's 40 supernode steps carry larger per-node bookkeeping (FrontResult moves, workspace machinery) that dominates at small n.
- `test_benchmark_adapters` passes (56/56 ctest).

Implication: the "performance leadership" claim must now be evaluated against MA27 — the real HSL-class small-n multifrontal baseline — not just MUMPS/PARDISO/CHOLMOD. The smf factor path is overhead-bound at n≈360 (25 kflop in ~0.06ms/step); closing the gap means attacking per-call fixed costs (workspace prep, front construction, pivot bookkeeping), not flop throughput.
