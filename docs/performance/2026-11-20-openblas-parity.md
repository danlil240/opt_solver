# OpenBLAS Performance Parity Gate (1T & 8T vs MKL)

Date: 2026-11-20
Branch: `perf/openblas-parity` (base: `main` @ `865dc3c`, post-PR #79)
Hardware: AMD Ryzen 9 9950X3D (16C/32T, Windows 11 host for MKL 2026.2 + MSVC 2022; WSL2 Ubuntu GCC 15.2.0 + OpenBLAS 0.3.32 pthread for OpenBLAS)

## Summary

Prior to this change, OpenBLAS had never been benchmarked and suffered three major issues on Linux/GCC:

1. **Slow `dgemmt` fallback (1T & 8T):** Non-MKL builds fell back to a serial `kBlock = 64` strip loop of `cblas_dgemm` calls, wasting upper-triangle flops and limiting GEMM panel width. (Note: OpenBLAS 0.3.32 exports `cblas_dgemmt`, but microbenchmarking showed its implementation is 4.0–4.8x slower than OpenBLAS `cblas_dgemm` on AVX-512/AVX2, so calling OpenBLAS `cblas_dgemmt` regressed 1T by 1.5–2.3x.)
2. **Process-wide `openblas_set_num_threads` races (8T & tests):** OpenBLAS has no thread-local thread control (`MKL_Set_Num_Threads_Local`). `SolvePool::worker`'s `BlasSerialGuard` and concurrent `TopScheduler` `BlasThreadGuard`/`BlasShareScope` calls raced on the process-wide thread count, causing oversubscription, 2.5–14x 8T slowdowns, and failures in `ParallelDeterminism` and `ThreadPlan`.
3. **Missing intra-front parallelism on non-MKL + `libgomp` per-thread pool contention:** PR #79's column-strip `deferred_update` and `schur_update` were gated `#if SMF_USE_MKL`, and SPD fronts relied on threaded BLAS. Invoking `#pragma omp parallel` from `SolvePool`'s `std::thread` workers under GCC `libgomp` spawned a separate OpenMP team per worker thread (up to 64 spinning threads on 8 cores), regressing medium-front cases by 2–3x.

## Changes

- **Recursive triangle-split `dgemmt`** (`solver/src/dense_kernel_lapack.cpp`): Replaced the `kBlock = 64` loop with recursive bisection (`dgemmt_lower_nt_rec` / `dgemmt_lower_nn_rec`, `kLeaf = 48`) dispatching square sub-blocks to full-width `cblas_dgemm`. Faster than both `kBlock = 64` and OpenBLAS `cblas_dgemmt`, and portable to Apple Accelerate / BLIS / Netlib.
- **Process-wide serial BLAS pinning** (`solver/include/smf/blas_thread_guard.hpp`, `solver/src/parallel/blas_thread_guard.cpp`, `solver/src/parallel/solve_pool.cpp`): Added `BlasSerialPinScope` (no-op on MKL; pins process-wide BLAS to 1 thread once on `SolvePool::run` entry and restores once on exit) and `BlasWorkerSerialScope` (accounting-only on non-MKL). `BlasThreadGuard` and `blas_follow_share` skip global `openblas_set_num_threads` calls while pinned, keeping thread-local `blas_current_share()` accurate and lock-free.
- **Cooperative `StripJob` dispatch on `TopScheduler`** (`solver/src/parallel/tile_tasks.{hpp,cpp}`, `solver/src/parallel/top_scheduler.{hpp,cpp}`, `solver/src/multifrontal_serial.cpp`, `solver/src/ldlt_kernel.cpp`, `solver/src/front_assembly.cpp`): Extended `TopScheduler` with a generic `HelperJob` interface and lock-free stack-allocated `StripJob` (`parallel_for_helpers`). On non-MKL builds, top-front `deferred_update`, `schur_update`, SPD Cholesky/TRSM/SYRK (`posdef_factor_front_lapack`), and front assembly/contribution copy all run cooperatively on the existing `SolvePool` threads with zero OpenMP thread creation. Works even with `OPENBLAS_NUM_THREADS=1`, `libopenblas-serial`, or Apple Accelerate (`plan_blas_parallel` updated in `thread_plan.cpp`).
- **Linux/GCC 15 & plugin fixes:** Added `<memory>` and fixed token-pasting in `gtest_fallback`, added non-Windows `Sampler` stub, and captured `delay_budget`/`ipm_ctrl` per `SmfAkeep` so `Ma97PluginAbi` passes under glibc `DF_1_NODELETE`.

## Benchmark Results

All runs: 3 measured trials + 1 warmup (median reported), idle host, 100% `Success` outcome.
Columns: `MKL` = Windows MKL 2026.2 (`main`), `OB0` = Linux OpenBLAS (`main`), `OB3` = Linux OpenBLAS (this branch). Times are `wall / factor` in ms.

### Long-Refactor KKT Suite (`benchmarks/manifests/probe_refactor8t.json`, `--seq-steps all`)

#### 1 Thread

| Case | MKL (wall / factor ms) | OB0 (wall / factor ms) | OB3 (wall / factor ms) | OB3 / MKL (wall / factor) | Factor speedup (OB0 / OB3) |
|---|---:|---:|---:|---:|---:|
| `acopf_grid_160` | 816.6 / 500.2 | 727.5 / 387.0 | 654.2 / 348.5 | 0.801x / 0.697x | 1.11x |
| `acopf_grid_80` | 114.2 / 85.1 | 105.8 / 72.6 | 90.3 / 60.5 | 0.791x / 0.711x | 1.20x |
| `obstacle_surface_200` | 77.2 / 59.6 | 71.6 / 50.7 | 60.9 / 40.8 | 0.789x / 0.686x | 1.24x |
| `pde_control_120` | 69.0 / 50.0 | 63.1 / 41.3 | 54.6 / 34.9 | 0.791x / 0.698x | 1.18x |
| `pde_control_250` | 564.4 / 398.6 | 492.7 / 297.4 | 405.3 / 261.4 | 0.718x / 0.656x | 1.14x |
| `pglib_opf_case1354_pegase` | 321.4 / 208.6 | 385.1 / 242.6 | 305.4 / 188.0 | 0.950x / 0.901x | 1.29x |
| `pglib_opf_case2000_goc__sad` | 1081.9 / 734.2 | 1190.6 / 781.8 | 941.7 / 613.9 | 0.870x / 0.836x | 1.27x |
| `portfolio_20k` | 55.4 / 33.7 | 61.4 / 38.3 | 50.6 / 29.8 | 0.915x / 0.885x | 1.28x |
| **Geomean (8)** | — | — | — | **0.825x / 0.753x** | **1.21x** |

#### 8 Threads

| Case | MKL (wall / factor ms) | OB0 (wall / factor ms) | OB3 (wall / factor ms) | OB3 / MKL (wall / factor) | Factor speedup (OB0 / OB3) |
|---|---:|---:|---:|---:|---:|
| `acopf_grid_160` | 267.8 / 130.5 | 576.2 / 329.8 | 254.4 / 107.3 | 0.950x / 0.823x | 3.07x |
| `acopf_grid_80` | 51.3 / 29.6 | 133.9 / 98.1 | 62.1 / 39.8 | 1.209x / 1.344x | 2.46x |
| `obstacle_surface_200` | 37.7 / 23.5 | 155.9 / 128.7 | 44.0 / 28.6 | 1.167x / 1.217x | 4.49x |
| `pde_control_120` | 52.5 / 31.1 | 66.8 / 33.9 | 49.6 / 28.8 | 0.945x / 0.927x | 1.18x |
| `pde_control_250` | 217.1 / 119.5 | 626.8 / 493.2 | 207.4 / 108.3 | 0.955x / 0.907x | 4.55x |
| `pglib_opf_case1354_pegase` | 196.3 / 102.6 | 215.4 / 110.5 | 172.2 / 85.0 | 0.877x / 0.829x | 1.30x |
| `pglib_opf_case2000_goc__sad` | 529.9 / 308.0 | 600.3 / 324.2 | 459.1 / 236.6 | 0.866x / 0.768x | 1.37x |
| `portfolio_20k` | 41.3 / 24.6 | 64.5 / 32.9 | 45.9 / 26.9 | 1.112x / 1.092x | 1.22x |
| **Geomean (8)** | — | — | — | **1.003x / 0.970x** | **2.11x** |

### Large-Front SPD & Hard Indefinite Suite (`G3_circuit, bratu3d, cfd2, helm3d01, ncvxqp9`)

#### 8 Threads

| Case (track) | MKL (wall / factor ms) | OB0 (wall / factor ms) | OB3 (wall / factor ms) | OB3 / MKL (wall / factor) | Factor speedup (OB0 / OB3) |
|---|---:|---:|---:|---:|---:|
| `G3_circuit` (rec) | 5498.3 / 3933.9 | 2441.3 / 1120.8 | 2554.2 / 1093.9 | 0.465x / 0.278x | 1.02x |
| `bratu3d` (ipopt) | 57.2 / 41.4 | 113.2 / 83.4 | 57.3 / 40.0 | 1.002x / 0.965x | 2.09x |
| `bratu3d` (rec) | 62.4 / 44.9 | 679.2 / 589.7 | 57.6 / 40.7 | 0.924x / 0.907x | **14.49x** |
| `cfd2` (rec) | 473.1 / 134.7 | 1115.8 / 267.3 | 475.1 / 100.8 | 1.004x / 0.749x | 2.65x |
| `helm3d01` (ipopt) | 92.4 / 52.4 | 532.0 / 415.2 | 86.9 / 44.7 | 0.941x / 0.853x | **9.28x** |
| `helm3d01` (rec) | 84.7 / 50.3 | 404.3 / 310.9 | 81.7 / 44.8 | 0.965x / 0.890x | **6.94x** |
| `ncvxqp9` (ipopt) | 9.5 / 5.3 | 77.2 / 71.3 | 10.7 / 6.3 | 1.128x / 1.175x | **11.38x** |
| `ncvxqp9` (rec) | 25.9 / 18.3 | 125.4 / 116.3 | 27.0 / 19.8 | 1.046x / 1.082x | **5.86x** |

## Verification

- **OpenBLAS (Linux/GCC 15.2.0):** `ctest` **83/83 passed** (including `ParallelDeterminism`, `ThreadPlan`, `Ma97PluginAbi`, `BlasThreadGuard`, `BlasWrap`, `TopScheduler`, `FrontAssembly`).
- **MKL (Windows/MSVC 2022):** `ctest` **83/83 passed** (MKL fast paths unchanged).
- **`OPENBLAS_NUM_THREADS=1` at 8T:** Verified full 8T scaling on `acopf_grid_160` and `pde_control_250`.
