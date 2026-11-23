# Blind Sealed Public SuiteSparse Benchmark Board (9 Rivals: CPU, Panua, cuDSS GPU, WSL2 SSIDS)

**Date:** 2026-11-22 (initial blind run completed 2026-11-23; Panua PARDISO adapter fix & `panua_pardiso`-only `1T`/`2T` re-run completed 2026-11-23)  
**Branch:** `bench/sealed-public-suitesparse`  
**Pre-registration sealed manifest commit:** `590eb19` (`benchmarks/manifests/public_suitesparse.json`, 60 matrices / 95 case-tracks)  
**Initial blind run commit:** `86e743a` (zero changes to `solver/src/`, `solver/include/`, `benchmarks/manifests/public_suitesparse.json`, or `benchmarks/manifests/fronttier.json`; all `smf`, `ma57`, `ma97`, `mkl_pardiso`, `mumps`, `cholmod`, `ssids`, and `cudss` records remain untouched from `86e743a`)  
**Panua adapter fairness fix:** Only `PanuaPardisoSolverAdapter` (`solver/benchmarks/adapters/pardiso.cpp`) was updated to insert explicit zero diagonal entries in 1-based upper-CSR and set Panua's symmetric-indefinite KKT defaults (`iparm[10]=1, iparm[12]=2, iparm[20]=3`), and only `panua_pardiso` (`T = 1` and `T = 2`) was re-run on the identical sealed manifest  
**Post-seal inertia reference:** `benchmarks/manifests/public_suitesparse_inertia.json` (generated strictly after all initial benchmark sweeps finished)

---

## 1. Provenance, Protocol & Fairness Constraints

### Hardware & Operating Systems (Single Physical Silicon)

Every Windows CPU, GPU, and WSL2 Linux run executed sequentially (one process at a time, idle host) on the same workstation:

- **CPU:** AMD Ryzen 9 9950X3D (16 cores / 32 threads), 61.6 GiB physical RAM (`66,155,405,312` bytes; WSL2 VM capped at 30 GiB)
- **GPU:** NVIDIA GeForce RTX 5080 (16,303 MiB VRAM, driver `621.14`)
- **Windows Host OS & Compiler:** Windows 11 (`10.0.26300-SP0`), MSVC `19.44.35228.0` (VS 2022 BuildTools `14.44.35207`)
  - CPU binary (`build-mtt/benchmarks/Release/bench_replay.exe`): `SMF_USE_MKL=ON`, `SMF_MKL_THREADING=intel`, Intel oneAPI MKL `2026.2-Product Build 20260915`
  - GPU binary (`build/benchmarks/Release/bench_replay.exe`): `SMF_BENCH_CUDSS=ON`, `SMF_MKL_THREADING=sequential`, NVIDIA cuDSS `0.8.0` (`libcudss-windows-x86_64-0.8.0.10_cuda13-archive`)
- **WSL2 Guest OS & Compiler:** Ubuntu Linux (`6.18.58.1-microsoft-standard-WSL2`), GCC / GFortran `15.2.0-16ubuntu1`, OpenBLAS `0.3.32+ds-5`, METIS `5.1.0.dfsg-8`
  - `ssids` driver (`~/ssids_bench/ssids_replay`): SPRAL SSIDS `2025.09.18` (CPU, OpenMP, `libspral.so`) with `OMP_CANCELLATION=TRUE`, `OMP_PROC_BIND=TRUE`, `OMP_THREAD_LIMIT=<T>`, `OPENBLAS_NUM_THREADS=1`
  - `smf` Linux parity binary (`~/opt_solver_wsl/build/benchmarks/bench_replay`): `SMF_USE_MKL=OFF` (OpenBLAS)

### Third-Party Rival Versions & Thread-Fairness Protocol

| Solver ID | Version / Library | Thread Participation | Notes |
|---|---|---|---|
| `smf` | `0.1.0` (in-tree `libsmf`, MKL on Win; OpenBLAS in WSL2) | `T = 1, 2, 8` | Default `Control` (`AutoSerial` on indefinite, `AutoParallel` promoted on `recommended` + SPD per `hsl.cpp:313-317`); `ipopt` track sets `pivot_u = 1e-8` + 3 IR steps |
| `ma57` | CoinHSL `2024.5.15` (`libcoinhsl.dll`, MinGW/gfortran5) | `T = 1` only | Serial-only (`--serial-solvers ma57`), excluded from `T = 8` |
| `ma97` | CoinHSL `2024.5.15` (`libcoinhsl.dll`, MinGW/gfortran5) | `T = 1, 8` | Full CoinHSL library via `SMF_HSL_LIBRARY` |
| `mkl_pardiso` | Intel oneAPI MKL `2026.2` (`mkl_rt`) | `T = 1, 8` | `build-mtt` with `SMF_MKL_THREADING=intel` |
| `panua_pardiso` | Panua PARDISO `20240630` (`libpardiso.dll`) | `T = 1`, matched `T = 2` | Valid academic trial license (`type 1`, **capped at 2 cores**, exp. `2026-12-22`). Fixed adapter (`iparm[10]=1, iparm[12]=2, iparm[20]=3` for `mtype=-2` + explicit zero diagonals in 1-based upper-CSR). Never run/scored at `T = 8`; raced head-to-head vs `smf` at matched `T = 2` |
| `mumps` | MUMPS `5.8.2` (`dmumps.lib`) | `T = 1, 8` | Built with MKL BLAS (`build-mtt`) |
| `cholmod` | SuiteSparse CHOLMOD `5.4.0` (`cholmod.lib`) | `T = 1, 8` | SPD-only (`Unavailable` on all 70 indefinite tracks); multi-threaded MKL BLAS at `T = 8` |
| `ssids` | SPRAL SSIDS `2025.09.18` (`libspral.so`, WSL2) | `T = 1, 8` | Cross-OS in main board + same-OS/same-BLAS parity view vs Linux OpenBLAS `smf` |
| `cudss` | NVIDIA cuDSS `0.8.0` (`cudss64`, RTX 5080) | `T = 1, 8` | `threads` controls host CPU analysis/ordering; exempt from CPU GFLOP/s plausibility cap (`--include-gpu`) |

### Corpus, Timing & Gates

- **Manifest:** `benchmarks/manifests/public_suitesparse.json` — **60 matrices / 95 case-tracks**, `mode: "cold"`, `split: "sealed_public"`, strictly zero name or generator-stem overlap with `benchmarks/manifests/fronttier.json`:
  - `public_indef_kkt`: 20 matrices × 2 tracks (`recommended`, `ipopt`) = **40 case-tracks**
  - `public_indef_pde`: 15 matrices × 2 tracks (`recommended`, `ipopt`) = 30 total (**24 scored**, 6 pre-tagged `report_only: true` for exact zero rows: `bcsstm36`, `k1_san`, `cop20k_A`)
  - `public_spd`: 25 matrices × 1 track (`recommended`) = **25 case-tracks**
  - **Scored total:** **89 case-tracks** (`35` indefinite × 2 + `25` SPD - `6` `report_only`)
- **Measurement:** 1 untimed warmup + 3 measured trials per `(case, track, solver, T)`, 300 s subprocess timeout, shuffled solver order (`seed=42`).
- **Accuracy & Plausibility Gates (`tools/bench/score.py`):** `eta_inf <= 1e-10` and `omega <= 1e-8`; CPU plausibility cap `165 GF/s * T` (**0 implausible records** detected across all runs). Metric: shifted geometric mean of `wall_ms(smf) / wall_ms(rival)` with `shift = 10 ms` over accurate records (`< 1.0` favors `smf`).

---

## 2. Executive Headline Summary

All ratios are `smf / rival` shifted 10 ms geometric means (`< 1.000` means `smf` is faster; `> 1.000` means `smf` is slower). Unfiltered over all 89 scored case-tracks.

| View | Threads | `smf` Solved | Common Cohort `n` | Shifted Geomean (`10 ms`) | `smf` #1 Overall | Rival / Comparison Scope |
|---|---:|---:|---:|---:|---:|---|
| **CPU-only (`smf` vs best accurate CPU)** | `T = 1` | **82 / 89** | 82 | **1.046** | 36 / 82 | All 7 CPU rivals (`ma57, ma97, mkl_pardiso, panua_pardiso, mumps, cholmod, ssids`) |
| **CPU-only (`smf` vs best accurate CPU)** | `T = 8` | **81 / 89** | 81 | **0.650** | **68 / 81** | 5 multi-core CPU rivals (`ma97, mkl_pardiso, mumps, cholmod, ssids`) |
| **Matched `T = 2` vs Panua PARDISO** | `T = 2` | **82 / 89** | 80 | **0.424** | **71 / 80** | Panua solved **`84 / 89`** (`5` inaccurate, `0` crashes, `0` timeouts) |
| **GPU head-to-head (`smf` vs `cudss` RTX 5080)** | `T = 1` | **82 / 89** | 64 | **0.851** | 35 / 64 | `cudss` solved `66 / 89` (`23` inaccurate) |
| **GPU head-to-head (`smf` vs `cudss` RTX 5080)** | `T = 8` | **81 / 89** | 64 | **0.416** | **57 / 64** | `cudss` solved `66 / 89` (`23` inaccurate) |
| **All rivals incl. GPU (`smf` vs best CPU+GPU)** | `T = 1` | **82 / 89** | 82 | **1.381** | 26 / 82 | 7 CPU rivals + RTX 5080 `cudss` |
| **All rivals incl. GPU (`smf` vs best CPU+GPU)** | `T = 8` | **81 / 89** | 81 | **0.672** | **66 / 81** | 5 multi-core CPU rivals + RTX 5080 `cudss` |
| **WSL2 OS-parity (`smf` OpenBLAS vs `ssids`)** | `T = 1` | **82 / 89** | 82 | **0.417** | 74 / 82 | Same WSL2 Ubuntu guest + GCC 15 + OpenBLAS |
| **WSL2 OS-parity (`smf` OpenBLAS vs `ssids`)** | `T = 8` | **82 / 89** | 82 | **0.352** | **79 / 82** | Same WSL2 Ubuntu guest + GCC 15 + OpenBLAS |
| **OS/BLAS parity (`wsl_smf` OpenBLAS / `win_smf` MKL)** | `T = 1` / `T = 8` | 82 / 82 (`wsl`) | 82 / 81 | **0.848** (`1T`) / **1.023** (`8T`) | — | Linux OpenBLAS `smf` is 15% faster at 1T, parity (+2.3%) at 8T |

### Best-Rival Counts in CPU-Only & All-Rivals Comparisons

- **`T = 1` CPU-only (82 pairs):** Fastest accurate CPU rival per case-track: `ma97` (26), `ma57` (20), `ssids` (15), `cholmod` (10), `mkl_pardiso` (7), `panua_pardiso` (4).  
- **`T = 1` Incl. GPU (82 pairs):** Fastest accurate overall rival: `cudss` (29), `ma97` (20), `ma57` (19), `cholmod` (8), `mkl_pardiso` (4), `panua_pardiso` (2).  
- **`T = 8` CPU-only (81 pairs):** Fastest accurate CPU rival: `mkl_pardiso` (52), `ma97` (20), `ssids` (6), `cholmod` (2), `mumps` (1).  
- **`T = 8` Incl. GPU (81 pairs):** Fastest accurate overall rival: `mkl_pardiso` (44), `ma97` (17), `cudss` (13), `ssids` (4), `cholmod` (2), `mumps` (1).

---

## 3. Per-Class & Per-Track Breakdown

### Cross-Track Per-Class Summary (`T = 1`, `T = 8`, Matched `T = 2`, WSL2 Parity)

| Class (Scored Cases) | View / Threads | `smf` Solved | Rival / Best Cohort `n` | CPU-Only Geomean | GPU H2H Geomean | Incl. GPU Geomean |
|---|---|---:|---:|---:|---:|---:|
| **`public_indef_kkt`** (40) | Main `T = 1` | 35 / 40 | 35 (GPU: 19) | **1.034** | **0.337** | **1.067** |
| **`public_indef_kkt`** (40) | Main `T = 8` | 34 / 40 | 34 (GPU: 19) | **0.496** | **0.257** | **0.499** |
| **`public_indef_kkt`** (40) | Matched `T = 2` vs Panua | 35 / 40 | 34 | **0.271** | — | — |
| **`public_indef_kkt`** (40) | WSL2 Parity `T = 1` / `T = 8` | 35 / 40 | 35 / 35 | **0.321** / **0.309** | — | — |
| **`public_indef_pde`** (24) | Main `T = 1` | 22 / 24 | 22 (GPU: 20) | **1.250** | **2.196** | **2.282** |
| **`public_indef_pde`** (24) | Main `T = 8` | 22 / 24 | 22 (GPU: 20) | **0.968** | **0.819** | **1.062** |
| **`public_indef_pde`** (24) | Matched `T = 2` vs Panua | 22 / 24 | 21 | **0.764** | — | — |
| **`public_indef_pde`** (24) | WSL2 Parity `T = 1` / `T = 8` | 22 / 24 | 22 / 22 | **0.719** / **0.562** | — | — |
| **`public_spd`** (25) | Main `T = 1` | 25 / 25 | 25 (GPU: 25) | **0.908** | **0.805** | **1.274** |
| **`public_spd`** (25) | Main `T = 8` | 25 / 25 | 25 (GPU: 25) | **0.663** | **0.349** | **0.673** |
| **`public_spd`** (25) | Matched `T = 2` vs Panua | 25 / 25 | 25 | **0.476** | — | — |
| **`public_spd`** (25) | WSL2 Parity `T = 1` / `T = 8` | 25 / 25 | 25 / 25 | **0.373** / **0.280** | — | — |

### Individual `(class, track, threads)` Cell Scores (`score.py`)

| Class | Track | `T` | `smf` Solved | CPU-Only (`n`) | GPU H2H (`n`) | Incl. GPU (`n`) | WSL2 Parity (`n`) | Matched `T=2` vs Panua (`n`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `public_indef_kkt` | `recommended` | `1` | 17 / 20* | 1.083 (17) | 0.424 (3) | 1.083 (17) | 0.390 (17) | — |
| `public_indef_kkt` | `recommended` | `8` | 16 / 20* | 0.566 (16) | 0.333 (3) | 0.566 (16) | 0.367 (17) | — |
| `public_indef_kkt` | `recommended` | `2` | 17 / 20 | — | — | — | — | 0.331 (16) |
| `public_indef_kkt` | `ipopt` | `1` | 18 / 20 | 0.990 (18) | 0.323 (16) | 1.053 (18) | 0.267 (18) | — |
| `public_indef_kkt` | `ipopt` | `8` | 18 / 20 | 0.440 (18) | 0.245 (16) | 0.445 (18) | 0.263 (18) | — |
| `public_indef_kkt` | `ipopt` | `2` | 18 / 20 | — | — | — | — | 0.228 (18) |
| `public_indef_pde` | `recommended` | `1` | 11 / 12 | 1.236 (11) | 2.204 (10) | 2.275 (11) | 0.726 (11) | — |
| `public_indef_pde` | `recommended` | `8` | 11 / 12 | 0.902 (11) | 0.780 (10) | 0.992 (11) | 0.506 (11) | — |
| `public_indef_pde` | `recommended` | `2` | 11 / 12 | — | — | — | — | 0.776 (10) |
| `public_indef_pde` | `ipopt` | `1` | 11 / 12 | 1.263 (11) | 2.187 (10) | 2.288 (11) | 0.711 (11) | — |
| `public_indef_pde` | `ipopt` | `8` | 11 / 12 | 1.038 (11) | 0.860 (10) | 1.137 (11) | 0.625 (11) | — |
| `public_indef_pde` | `ipopt` | `2` | 11 / 12 | — | — | — | — | 0.753 (11) |
| `public_spd` | `recommended` | `1` | 25 / 25 | 0.908 (25) | 0.805 (25) | 1.274 (25) | 0.373 (25) | — |
| `public_spd` | `recommended` | `8` | 25 / 25 | 0.663 (25) | 0.349 (25) | 0.673 (25) | 0.280 (25) | — |
| `public_spd` | `recommended` | `2` | 25 / 25 | — | — | — | — | 0.476 (25) |

*\*Scorer denominator note:* When a warmup attempt hits the 300 s subprocess timeout (`dtoc` `recommended` for `smf` at `T = 1, 8`; `Si34H36` for `ma97`/`mumps` at `T = 1`), `run.py` skips measured trials and writes no non-warmup record, so `score.py`'s table denominator appears as `19`, `17`, `16`, or `11`. All denominators in this report are normalized to the true **89 scored case-tracks** (40 KKT + 24 PDE + 25 SPD). Under the fixed adapter, `panua_pardiso` completed all 95 case-tracks with zero timeouts and zero crashes.

---

## 4. Head-to-Head Matrix (`smf` vs Each Individual Rival)

Shifted 10 ms geometric mean of `smf / rival` on the **common accurate cohort** (`n_common`). `W / T / L (1.1x)` counts raw wall-time wins (`smf < rival / 1.1`), ties (`within 1.1x`), and losses (`smf > 1.1 * rival`).

### `T = 1` Head-to-Head

| Rival | `n_common` | Overall Geomean | `W / T / L (1.1x)` | Raw `W / L` | `public_indef_kkt` (`n`) | `public_indef_pde` (`n`) | `public_spd` (`n`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`ma57`** | 81 | **0.713** | 46 / 12 / 23 | 48 / 33 | 0.741 (34) | 1.015 (22) | 0.496 (25) |
| **`ma97`** | 80 | **0.697** | 48 / 15 / 17 | 53 / 27 | 0.714 (33) | 0.947 (22) | 0.515 (25) |
| **`mkl_pardiso`** | 75 | **0.531** | 60 / 4 / 11 | 61 / 14 | 0.345 (29) | 0.817 (21) | 0.609 (25) |
| **`panua_pardiso`** | 80 | **0.474** | 65 / 5 / 10 | 69 / 11 | 0.303 (34) | 0.795 (21) | 0.563 (25) |
| **`mumps`** | 79 | **0.327** | 78 / 1 / 0 | 79 / 0 | 0.272 (34) | 0.397 (20) | 0.362 (25) |
| **`cholmod`** *(SPD)* | 25 | **0.759** | 20 / 4 / 1 | 24 / 1 | — | — | 0.759 (25) |
| **`ssids`** *(WSL2)* | 82 | **0.492** | 67 / 4 / 11 | 68 / 14 | 0.353 (35) | 0.904 (22) | 0.459 (25) |
| **`ssids`** *(WSL2 parity)* | 82 | **0.417** | 73 / 1 / 8 | 74 / 8 | 0.321 (35) | 0.719 (22) | 0.373 (25) |
| **`cudss`** *(RTX 5080)* | 64 | **0.851** | 35 / 2 / 27 | 35 / 29 | 0.337 (19) | 2.196 (20) | 0.805 (25) |

### Matched `T = 2` Head-to-Head (`panua_t2`)

| Rival | `n_common` | Overall Geomean | `W / T / L (1.1x)` | Raw `W / L` | `public_indef_kkt` (`n`) | `public_indef_pde` (`n`) | `public_spd` (`n`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`panua_pardiso`** | 80 | **0.424** | 71 / 0 / 9 | 71 / 9 | 0.271 (34) | 0.764 (21) | 0.476 (25) |

### `T = 8` Head-to-Head

| Rival | `n_common` | Overall Geomean | `W / T / L (1.1x)` | Raw `W / L` | `public_indef_kkt` (`n`) | `public_indef_pde` (`n`) | `public_spd` (`n`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`ma97`** | 79 | **0.347** | 75 / 3 / 1 | 77 / 2 | 0.354 (32) | 0.471 (22) | 0.259 (25) |
| **`mkl_pardiso`** | 74 | **0.622** | 63 / 3 / 8 | 63 / 11 | 0.440 (28) | 0.939 (21) | 0.648 (25) |
| **`mumps`** | 77 | **0.203** | 77 / 0 / 0 | 77 / 0 | 0.231 (32) | 0.206 (20) | 0.170 (25) |
| **`cholmod`** *(SPD)* | 25 | **0.445** | 24 / 0 / 1 | 24 / 1 | — | — | 0.445 (25) |
| **`ssids`** *(WSL2)* | 81 | **0.343** | 75 / 3 / 3 | 78 / 3 | 0.298 (34) | 0.564 (22) | 0.270 (25) |
| **`ssids`** *(WSL2 parity)* | 82 | **0.352** | 77 / 2 / 3 | 79 / 3 | 0.309 (35) | 0.562 (22) | 0.280 (25) |
| **`cudss`** *(RTX 5080)* | 64 | **0.416** | 54 / 4 / 6 | 57 / 7 | 0.257 (19) | 0.819 (20) | 0.349 (25) |

---

## 5. True Accuracy, Failure, OOM, Timeout & Crash Audit (All 89 Scored Cases)

Every solver's true outcome across all **89 scored case-tracks** (no hidden warmup timeouts or missing crash rows):

| Solver | `T` | Accurate (`ok`) | Inaccurate | Hard Fail (`Singular` / `OOM`) | Timeout (300 s) | Crash (`NoRecord`) | Unavail | Notes on Non-`ok` Scored Cases |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **`smf`** *(Win MKL)* | `1` | **82 / 89** | 3 | 3 (`1` Singular, `2` OOM) | 1 (warmup) | 0 | 0 | `dtoc` (`ipopt` Singular, `rec` Timeout); `Si34H36` (`2` OOM); `OPF_10000` (`ipopt`), `HTC_336_4438` (`rec`), `HTC_336_9129` (`rec`) |
| **`smf`** *(Win MKL)* | `8` | **81 / 89** | 4 | 3 (`1` Singular, `2` OOM) | 1 (warmup) | 0 | 0 | Same 7 + `c-54` (`rec`: trial 0 `omega=1.055e-8 > 1e-8`, trials 1–2 `Success`) |
| **`smf`** *(Win MKL)* | `2` | **82 / 89** | 3 | 4 (`2` Singular, `2` OOM) | 0 | 0 | 0 | Same 7 (`dtoc` `rec` completed in 141.9 s as `Singular`) |
| **`smf`** *(WSL2 OpenBLAS)* | `1, 8` | **82 / 89** | 3 | 3 (`1` Singular, `2` OOM) | 1 (warmup) | 0 | 0 | Identical 7 non-`ok` cases at both `1T` and `8T`; `c-54` (`rec`) passed at `8T` |
| **`ma57`** | `1` | **86 / 89** | 3 | 0 | 0 | 0 | 0 | `dtoc` (`2`), `c-54` (`rec`: `omega=1.58e-8`) |
| **`ma97`** | `1` | **81 / 89** | 6 | 0 | 2 (warmup) | 0 | 0 | `dtoc` (`2`), `c-62` (`rec`), `c-64` (`rec`), `HTC_336_4438` (`rec`), `HTC_336_9129` (`rec`), `Si34H36` (`2` Timeout) |
| **`ma97`** | `8` | **83 / 89** | 6 | 0 | 0 | 0 | 0 | Same 6 inaccurate; `Si34H36` completed in 63.2 s |
| **`mkl_pardiso`** | `1, 8` | **78 / 89** | 11 | 0 | 0 | 0 | 0 | `dtoc` (`2`), `OPF_6000` (`rec`), `OPF_10000` (`rec`), `c-62` (`rec`), `c-64` (`rec`), `c-67` (`rec`), `HTC_336_4438` (`2`), `HTC_336_9129` (`rec`), `qa8fk` (`rec`) |
| **`panua_pardiso`** | `1` | **84 / 89** | 5 | 0 | 0 | 0 | 0 | `dtoc` (`2`), `OPF_10000` (`2`), `qa8fk` (`rec`: `eta=omega=1.48e-10`) |
| **`panua_pardiso`** | `2` | **84 / 89** | 5 | 0 | 0 | 0 | 0 | Identical 5 inaccurate cases (`dtoc` ×2, `OPF_10000` ×2, `qa8fk` `rec`) |
| **`mumps`** | `1` | **82 / 89** | 1 | 4 (`4` Singular) | 2 (warmup) | 0 | 0 | `dtoc` (`2` Singular), `CurlCurl_1` (`2` Singular), `c-64` (`rec`), `Si34H36` (`2` Timeout) |
| **`mumps`** | `8` | **83 / 89** | 2 | 4 (`4` Singular) | 0 | 0 | 0 | `dtoc` (`2` Singular), `CurlCurl_1` (`2` Singular), `c-62` (`rec`), `c-64` (`rec`); `Si34H36` solved in 117.2 s |
| **`cholmod`** | `1, 8` | **25 / 25 SPD** | 0 | 0 | 0 | 0 | 64 | `Unavailable` on all 64 scored indefinite tracks (expected) |
| **`ssids`** *(WSL2)* | `1, 8` | **87 / 89** | 2 | 0 | 0 | 0 | 0 | `dtoc` (`2` Inaccurate, `eta=0.25, omega=1.0`) |
| **`cudss`** *(RTX 5080)* | `1, 8` | **66 / 89** | 23 | 0 | 0 | 0 | 0 | Inaccurate on 16 `rec` KKT + `dtoc` (`2`) + `brainpc2` (`ipopt`) + `OPF_6000` (`ipopt`) + `OPF_10000` (`ipopt`) + `tuma1` (`2`) |

### Detailed Audit of Every `smf` Non-`ok` Scored Case

| Case (Class / Track) | `n` | Threads / Runner | Outcome & Status | `eta_inf` | `omega` | Reported Inertia vs Ref | Root Cause Observation (No Code Edits) |
|---|---:|---|---|---|---|---|---|
| **`dtoc`** (`kkt` / `ipopt`) | 24,993 | All (`1, 2, 8`, Win & WSL) | `Singular` (`wall ~131–134 s`, `112.6 s` at 2T) | — | — | `(9997, 9997, 4999)` vs `(9997, 9997, 4999)` exact | Algebraically singular matrix with **4,999 zero eigenvalues** (lacks an all-zero row or constant-1 nullspace, so it was not caught by `fetch_public_suitesparse.py`'s pre-filter). Triggers a 12.48M-delay / 166.9 GFLOP cascade (`L = 75.08M`) before `smf` detects rank deficiency (`Singular`). Every solver failed `dtoc`. |
| **`dtoc`** (`kkt` / `recommended`) | 24,993 | `Win 1T, 8T`; `WSL 1T, 8T` | `Timeout` (>300 s warmup) | — | — | — | Same rank-4999 singular matrix; `T = 2` completed in `141.9 s` (`Singular`). |
| **`Si34H36`** (`pde` / `recommended`, `ipopt`) | 97,569 | All (`1, 2, 8`, Win & WSL) | `OutOfMemory` (`wall ~0.33–0.36 s`, `271 MB` RSS) | — | — | `(0, 0, 0)` (ref disputed) | `check_analysis_storage_bounds(keep)` (`factor_indef.cpp:402`) aborted because predicted panel storage exceeded `kMaxFactorStorageEntries = 2^29` (`536,870,912` entries, ~4 GiB cap). `hsl.cpp:314` left `ctrl.ordering = AutoSerial` (AMD only, never METIS) on indefinite matrices, whereas METIS-enabled rivals stayed just below the cap (`ssids 480M`, `panua 482M`, `ma57 530M`, `ma97 531M`; `mumps` used `1.35B` / `20.3 GB` RSS). |
| **`OPF_10000`** (`kkt` / `ipopt`) | 43,887 | All (`1, 2, 8`, Win & WSL) | `Inaccurate` (`wall ~20–25 ms`) | `1.1e-17..1.9e-17` | `3.2e-8..4.2e-7` | `(21392, 22495, 0)` vs `(21392, 22494, 1)` consistent | Matrix has 1 near-zero eigenvalue. On `ipopt` (`u = 1e-8 <= skip_trial_max_pivot_u`), `skip_trial_gate` skipped the MC64 matching trial (`matching=0.49 ms` vs `12.1 ms` on `recommended`), inflating delays from `45 -> 1004` and raising `omega` past `1e-8` despite machine-precision `eta_inf`. On `recommended`, `smf` ran MC64 and passed (`omega = 5.2e-11`). |
| **`HTC_336_4438`** (`kkt` / `recommended`) | 226,340 | All (`1, 2, 8`, Win & WSL) | `Inaccurate` (`wall ~496–614 ms`) | `3.2e-18..4.3e-18` | `1.8e-7..3.1e-7` | `(105781, 120559, 0)` exact | `eta_inf` is machine precision (`~3e-18`), failing only `OMEGA_GATE = 1e-8` with `ir_steps = 0`. Shared by `ma97` (`5.0e-7`), `mkl_pardiso` (`0.076`), and `cudss` (`0.012`). On `ipopt` (`ir_steps = 3`), `smf` passes at `omega <= 5.2e-15` across Win & WSL (`<= 5.0e-15` Win, `5.16e-15` WSL `8T`) in `63–68 ms`. |
| **`HTC_336_9129`** (`kkt` / `recommended`) | 226,340 | All (`1, 2, 8`, Win & WSL) | `Inaccurate` (`wall ~91–139 ms`) | `9.1e-18..1.4e-17` | `7.1e-7..1.2e-6` | `(105781, 120559, 0)` exact | Same pattern as `HTC_336_4438`: `eta_inf ~ 1e-17`, no refinement on `recommended` (`ma97 1.4e-6`, `mkl_pardiso 6.5e-6`, `cudss 7.6e-3`). On `ipopt` (`ir_steps = 3`), `smf` passes at `omega <= 5.5e-15` in `63–73 ms`. |
| **`c-54`** (`kkt` / `recommended`) | 31,793 | `Win 8T` only | `Inaccurate` (1/3 trials; `100.6 ms`) | `5.8e-19..1.2e-18` | `9.14e-9..1.055e-8` | `(17664, 14129, 0)` exact | Non-deterministic parallel tree reduction straddles `OMEGA_GATE = 1e-8`: trial 0 hit `1.055e-8`, trials 1–2 passed (`9.96e-9`, `9.14e-9`). Passed at `Win 1T` (`7.29e-9`), `Win 2T`, and `WSL2 1T/8T`, and passed `ipopt` at `omega = 7.1e-16`. |

---

## 6. Post-Seal Inertia Verification (`public_suitesparse_inertia.json`)

Post-seal spectrum slicing (`smf` + `ma57` at `pivot_u = 0.5` with Ruiz equilibration, `shift = 1e-10`, `--timeout 120`) confirmed **59 / 60 matrices** (`34/35` indefinite + `25/25` SPD). Only **`Si34H36`** (`n = 97,569`) was marked `disputed` (`smf` hit `kMaxFactorStorageEntries` OOM; `ma57` timed out at `> 120 s` per slice).

### Inertia Agreement Across All 9 Solvers (Scored Cases)

- **Zero wrong inertias (`0 FAIL`) across all checked cases and tracks:**
  - **`smf`** (Win 1T, 2T, 8T; WSL 1T, 8T): **75 exact / 10 consistent / 0 wrong** of 85 checked (`PASS`; 4 unchecked by `score.py`: `Si34H36` x2 `disputed`/`OOM` plus `dtoc` x2 `Singular`/`Timeout`, though on `dtoc` `ipopt` and `2T rec` `smf` still reported exact `(9997, 9997, 4999)`).
  - **`ma97`** (1T & 8T): **77 exact / 10 consistent / 0 wrong** of 87 checked (`PASS`)
  - **`ssids`** (1T & 8T): **77 exact / 10 consistent / 0 wrong** of 87 checked (`PASS`)
  - **`mumps`** (1T & 8T): **75 exact / 8 consistent / 0 wrong** of 83 checked (`PASS`)
  - **`cholmod`** (SPD 1T & 8T): **25 exact / 0 consistent / 0 wrong** of 25 checked (`PASS`)
- **Consistent near-zero-cluster splits (`10` per full-coverage solver):** `OPF_10000` (2 tracks; ref `(21392, 22494, 1)`), `CurlCurl_1` (2 tracks; ref `(196876, 0, 29575)`), `F2` (2 tracks; ref `(71499, 0, 6)`), `bcsstk35` (2 tracks; ref `(30231, 0, 6)`), `qa8fk` (2 tracks; ref `(66126, 0, 1)`). All accurate Bunch-Kaufman solvers report 0 zero pivots by splitting the zero cluster cleanly into `(pos, neg)`.
- **Solvers with wrong inertias:**
  - **`ma57`** (`T = 1`): **75 exact / 10 consistent / 2 wrong** of 87 (`FAIL`): reported `(89399, 1, 0)` on SPD `denormal` (ref `(89400, 0, 0)`) and `(102157, 1, 0)` on SPD `thermomech_TK` (ref `(102158, 0, 0)`), even though both passed backward error (`eta_inf ~ 3e-16`).
  - **`mkl_pardiso`** (`T = 1, 8`): **73 exact / 12 consistent / 2 wrong** of 87 (`FAIL`): `HTC_336_4438` (`ipopt` & `rec`) reported `(105782, 120558, 0)` vs `(105781, 120559, 0)` (inaccurate).
  - **`panua_pardiso`** (`T = 1, 2`): **75 exact / 10 consistent / 2 wrong** of 87 (`FAIL`): `dtoc` (`ipopt` & `rec`) reported `(24993, 0, 0)` vs `(9997, 9997, 4999)` (inaccurate; perturbed 4,999 zero + 9,997 negative pivots positive). On `OPF_10000`, the fixed adapter reported `(21392, 22495, 0)` (consistent with `(21392, 22494, 1)`).
  - **`cudss`** (`T = 1, 8`): **71 exact / 10 consistent / 6 wrong** of 87 (`FAIL`): `brainpc2` (`2` tracks: `(13805, 13802, 0)` vs `(13807, 13800, 0)`), `OPF_6000` (`2` tracks: `(14305..14311, 15591..15597, 0)` vs `(14274, 15628, 0)`), `OPF_10000` (`2` tracks: `(21455..21463, 22424..22432, 0)` vs `(21392, 22494, 1)`) — all inaccurate static-pivoting sign flips.

---

## 7. Pre-Tagged `report_only` Singular Cases (6 Case-Tracks)

Matrices `bcsstm36` (`n = 23,052`), `k1_san` (`n = 67,759`), and `cop20k_A` (`n = 121,192`) were tagged `report_only: true` in `public_suitesparse.json` before execution due to exact zero rows. Excluded from ratios/gates:

| Matrix (`n`) | Inertias: `ref` / `smf` | `smf` (`1T`/`8T`) | `ma57` (`1T`) / `ma97` (`1T`,`8T`) | `mkl_pardiso` (`1T`,`8T`) | `panua_pardiso` (`1T` / `2T`) | `mumps` (`1T`,`8T`) | `ssids` (`1T`,`8T`) | `cudss` (`1T`,`8T`) |
|---|---|---|---|---|---|---|---|---|
| `bcsstm36` (23,052) | `(12172, 0, 10880)` | `Singular` (`12–15 ms`) | `Inaccurate` (`omega=1`) | `Inaccurate` (`omega=1`, inertia `(23052,0,0)`) | `Inaccurate` (`omega=1`, inertia `(23052,0,0)`, `74.2`/`77.5 ms`) | `Singular` | `Inaccurate` (`omega=1`) | `Inaccurate` (`omega=1`, inertia `(23052,0,0)`) |
| `k1_san` (67,759) | `(46954, 20804, 1)` | `Singular` (`32–61 ms`) | `Inaccurate` (`omega=1`) | `Inaccurate` (`omega=1`, inertia `(46955,20804,0)`) | `Inaccurate` (`omega=1`, inertia `(46955,20804,0)`, `295.3`/`259.0 ms`) | `Singular` | `Inaccurate` (`omega=1`) | `Inaccurate` (`omega=1`, inertia `(46955,20804,0)`) |
| `cop20k_A` (121,192) | `(98566, 1277, 21349)` | `Singular` (`216–695 ms`) | `Inaccurate` (`omega=1`) | `Inaccurate` (`omega=1`, inertia `(119915,1277,0)`) | `Inaccurate` (`omega=1`, inertia `(119915,1277,0)`, `1025.3`/`830.6 ms`) | `Singular` | `Inaccurate` (`omega=1`) | `Inaccurate` (`omega=1`, inertia `(119915,1277,0)`) |

Notice `smf`, `ma57`, `ma97`, `mumps`, and `ssids` all identified exact algebraic nullities (`10880`, `1`, `21349`), whereas `mkl_pardiso`, `panua_pardiso`, and `cudss` perturbed zero pivots and reported full-rank `(pos, neg, 0)` inertias.

---

## 8. Complete Unfiltered Loss List (`smf > 1.1x` Rival Wall Time) & Root-Cause Observations

### A. Root-Cause Findings Observed in the Blind Run (No Solver Code Modified)

1. **`AutoSerial` Lock-In on Indefinite Matrices (`solver/benchmarks/adapters/hsl.cpp:313-317`, `solver/include/smf/control.hpp:13`):**  
   `Control::ordering` defaults to `OrderingMethod::AutoSerial`. `SmfSolverAdapter::configure()` promotes `AutoSerial -> AutoParallel` **only** for `track == "recommended"` and `MatrixType::RealSymmetricPositiveDefinite`. Consequently, **every indefinite case on both `recommended` and `ipopt` ran under `AutoSerial`**, which never runs METIS (`symbolic_analysis.cpp:903-908`) and always picks AMD. On 3D PDE / FEM / semiconductor indefinite meshes (`SiO`, `Lin`, `CurlCurl_1`, `gas_sensor`, `qa8fk`, `F1`), AMD produced **1.6x–3.2x more factor entries** (e.g. `SiO`: `263.4M` vs `ssids` `83.4M`, `1.51` vs `0.32` TFLOPs; `gas_sensor`: `69.5M` vs `panua` `23.1M`) and pushed `Si34H36` past `kMaxFactorStorageEntries = 2^29` (`OutOfMemory`). By contrast, on `public_spd` (where `AutoParallel` fired), `smf` scored **0.910 at 1T** and **0.663 at 8T** (`25/25` solved, 0 PDE-style fill blowups).
2. **`SiO` `ipopt` vs `recommended` at `T = 8` (`8.84 s` vs `4.22 s` factor):**  
   At `T = 1`, `SiO` took `21.8 s` on both tracks; at `T = 8`, `recommended` scaled to `4.22 s` factor (`4.37 s` wall), whereas `ipopt` took `8.84 s` (`9.05 s` wall) on Windows MKL (and `9.54 s` vs `3.19 s` on WSL2 OpenBLAS) despite identical `L = 263,374,073`, `flops = 1.511e12`, `order_hash`, and `delays = 0`.
3. **Fixed Analysis/Matching Overhead on Fast KKT & Diagonal Cases at `1T`:**  
   On small/sparse KKT matrices (`c-49`..`c-67b`, `qpband`, `a5esindl`, `OPF_6000`), `smf`'s 18–52 ms analysis/matching overhead loses to `ma57`/`ma97` at `T = 1` (and on diagonal `bcsstm39` to `panua_pardiso`). Every single KKT speed loss disappears at `T = 8` (`0` KKT losses > 1.1x at `8T`).

### B. Main Board CPU-Only Losses `> 1.1x` at `T = 8` (All **9** of 81)

| Ratio | Matrix (`class` / `track`) | `n` | `nnz_L` (`smf`) | `smf` wall (ana / fac / sol ms) | Best Accurate CPU Rival | Rival wall (ana / fac / sol ms) | Observed Cause |
|---:|---|---:|---:|---:|---|---:|---|
| **4.74x** | `SiO` (`pde` / `ipopt`) | 33,401 | 263,374,073 | 9047.1 (84.8 / 8838.7 / 122.1) | `mkl_pardiso` | 1910.4 (202.0 / 1634.6 / 51.8) | AMD 3.16x fill (`ssids` METIS `83.4M`) + 8T `ipopt` scaling gap |
| **2.64x** | `Lin` (`pde` / `ipopt`) | 256,000 | 214,969,542 | 3722.0 (138.3 / 3479.7 / 106.6) | `mkl_pardiso` | 1407.4 (312.3 / 994.5 / 76.0) | AMD 2.25x fill (`panua`/`mkl` METIS `~95.7M`) |
| **2.32x** | `SiO` (`pde` / `rec`) | 33,401 | 263,374,073 | 4368.4 (86.1 / 4221.1 / 60.2) | `mkl_pardiso` | 1885.1 (199.7 / 1626.2 / 24.5) | AMD 3.16x fill (`AutoSerial` skipped METIS) |
| **2.04x** | `gas_sensor` (`pde` / `rec`) | 66,917 | 69,470,759 | 513.7 (56.4 / 437.6 / 18.3) | `mkl_pardiso` | 252.3 (154.0 / 80.6 / 9.5) | AMD 3.01x fill (`panua`/`mkl` METIS `23.1M`) |
| **2.00x** | `gas_sensor` (`pde` / `ipopt`) | 66,917 | 69,470,759 | 519.6 (57.0 / 440.6 / 21.1) | `mkl_pardiso` | 260.3 (159.4 / 79.7 / 11.9) | AMD 3.01x fill |
| **1.99x** | `bcsstm39` (`spd` / `rec`) | 46,772 | 46,772 | 4.36 (1.8 / 2.3 / 0.3) | `cholmod` | 2.19 (1.1 / 0.6 / 0.2) | Pure diagonal matrix; 2 ms fixed setup overhead |
| **1.92x** | `Lin` (`pde` / `rec`) | 256,000 | 214,969,542 | 2600.5 (138.5 / 2411.2 / 50.7) | `mkl_pardiso` | 1351.9 (293.7 / 996.0 / 34.5) | AMD 2.25x fill |
| **1.85x** | `CurlCurl_1` (`pde` / `ipopt`) | 226,451 | 184,931,635 | 1761.0 (124.0 / 1585.5 / 53.7) | `mkl_pardiso` | 954.4 (353.4 / 508.9 / 73.6) | AMD 2.41x fill (`panua` `76.9M`, `ssids` `82.1M`) |
| **1.69x** | `CurlCurl_1` (`pde` / `rec`) | 226,451 | 184,938,037 | 1609.4 (128.1 / 1431.5 / 48.6) | `mkl_pardiso` | 950.1 (352.1 / 502.7 / 69.9) | AMD 2.41x fill |

*(At `T = 8`, `smf` has **0 losses > 1.1x** in `public_indef_kkt` and **only 1 trivial 2 ms loss** on diagonal `bcsstm39` in `public_spd`.)*

### C. Main Board CPU-Only Losses `> 1.1x` at `T = 1` (All **33** of 82)

| Ratio | Matrix (`class` / `track`) | `smf` wall ms (`ana`/`fac`/`sol`) | Best CPU Rival | Rival wall ms (`ana`/`fac`/`sol`) | `smf` / Rival `L` entries |
|---:|---|---:|---|---:|---|
| **3.61x** | `c-54` (`kkt` / `rec`) | 126.17 (106.9 / 18.3 / 0.9) | `ma97` | 34.90 (17.0 / 14.9 / 2.1) | 903K / 1.38M |
| **3.53x** | `qpband` (`kkt` / `ipopt`) | 4.79 (3.3 / 1.1 / 0.3) | `ma57` | 1.36 (0.5 / 0.4 / 0.3) | 80K / 50K |
| **3.51x** | `bcsstm39` (`spd` / `rec`) | 4.60 (2.0 / 2.2 / 0.3) | `panua_pardiso` | 1.31 (0.0 / 0.1 / 0.2) | 46.8K / 46.8K |
| **3.46x** | `SiO` (`pde` / `rec`) | 21951.47 (82.8 / 21801.7 / 67.7) | `ssids` | 6349.11 (308.7 / 5891.9 / 30.2) | 263.4M / 83.4M |
| **3.32x** | `SiO` (`pde` / `ipopt`) | 21988.32 (82.2 / 21765.6 / 139.8) | `ssids` | 6629.47 (317.7 / 6124.4 / 32.5) | 263.4M / 83.4M |
| **2.63x** | `c-67` (`kkt` / `rec`) | 83.95 (40.4 / 42.2 / 0.8) | `ma97` | 31.87 (16.7 / 12.1 / 2.3) | 718K / 1.19M |
| **2.35x** | `gas_sensor` (`pde` / `rec`) | 2055.63 (56.3 / 1974.6 / 25.0) | `mkl_pardiso` | 873.35 (378.3 / 474.3 / 15.7) | 69.5M / — |
| **2.34x** | `gas_sensor` (`pde` / `ipopt`) | 2054.00 (56.4 / 1969.1 / 28.0) | `ssids` | 876.13 (412.5 / 430.4 / 15.4) | 69.5M / 25.4M |
| **2.34x** | `a5esindl` (`kkt` / `ipopt`) | 20.33 (16.1 / 3.0 / 1.2) | `ma57` | 8.68 (4.0 / 2.7 / 1.3) | 575K / 310K |
| **2.10x** | `CurlCurl_1` (`pde` / `ipopt`) | 6842.66 (125.5 / 6647.1 / 71.4) | `ssids` | 3254.51 (1282.2 / 1865.6 / 46.7) | 184.9M / 82.1M |
| **2.08x** | `CurlCurl_1` (`pde` / `rec`) | 6839.65 (122.5 / 6646.5 / 67.4) | `ssids` | 3284.73 (1284.6 / 1906.5 / 42.9) | 184.9M / 82.1M |
| **1.77x** | `c-67b` (`kkt` / `rec`) | 49.64 (40.1 / 8.7 / 0.8) | `ma97` | 28.03 (16.7 / 8.8 / 1.9) | 726K / 1.01M |
| **1.71x** | `Lin` (`pde` / `ipopt`) | 10857.08 (134.3 / 10573.1 / 148.1) | `ssids` | 6361.94 (1163.5 / 4983.9 / 122.3) | 215.0M / 123.3M |
| **1.70x** | `Lin` (`pde` / `rec`) | 10785.32 (134.2 / 10581.2 / 70.7) | `ssids` | 6358.80 (1172.6 / 5015.2 / 59.8) | 215.0M / 123.3M |
| **1.58x** | `c-52` (`kkt` / `rec`) | 28.68 (23.1 / 5.0 / 0.5) | `ma97` | 18.19 (11.3 / 5.5 / 1.0) | 466K / 597K |
| **1.41x** | `c-64b` (`kkt` / `rec`) | 54.01 (40.4 / 12.8 / 0.8) | `ma97` | 38.19 (18.3 / 16.4 / 2.7) | 1.12M / 1.73M |
| **1.38x** | `c-49` (`kkt` / `rec`) | 23.31 (18.1 / 4.7 / 0.4) | `ma97` | 16.84 (9.8 / 5.7 / 0.9) | 483K / 573K |
| **1.37x** | `c-67b` (`kkt` / `ipopt`) | 31.42 (20.5 / 7.4 / 3.6) | `ma57` | 22.91 (12.8 / 6.5 / 2.5) | 708K / 737K |
| **1.33x** | `qa8fk` (`pde` / `rec`) | 1144.35 (53.3 / 1071.6 / 19.1) | `ssids` | 860.20 (387.0 / 435.5 / 12.7) | 50.0M / 24.5M |
| **1.30x** | `c-67` (`kkt` / `ipopt`) | 31.59 (20.6 / 7.2 / 3.7) | `ma57` | 24.25 (12.8 / 6.4 / 4.0) | 697K / 705K |
| **1.25x** | `qa8fk` (`pde` / `ipopt`) | 1149.03 (55.0 / 1073.0 / 22.8) | `ssids` | 919.35 (396.6 / 480.4 / 15.3) | 50.0M / 24.5M |
| **1.25x** | `c-63` (`kkt` / `rec`) | 93.00 (51.1 / 39.8 / 1.8) | `ma97` | 74.57 (26.1 / 43.7 / 3.5) | 3.47M / 3.28M |
| **1.21x** | `HTC_336_9129` (`kkt` / `ipopt`) | 71.29 (30.5 / 31.2 / 9.4) | `ma57` | 59.07 (26.2 / 20.2 / 10.5) | 1.99M / 1.97M |
| **1.20x** | `OPF_6000` (`kkt` / `ipopt`) | 12.30 (7.5 / 2.5 / 2.2) | `ma57` | 10.26 (3.9 / 3.5 / 2.4) | 339K / 402K |
| **1.19x** | `c-49` (`kkt` / `ipopt`) | 15.45 (9.7 / 4.3 / 1.3) | `ma57` | 13.00 (5.8 / 5.2 / 1.5) | 456K / 471K |
| **1.18x** | `c-63` (`kkt` / `ipopt`) | 92.06 (51.9 / 35.4 / 4.8) | `ma97` | 77.84 (25.9 / 43.0 / 7.8) | 3.18M / 3.26M |
| **1.17x** | `qpband` (`kkt` / `rec`) | 4.62 (3.4 / 1.1 / 0.1) | `ma57` | 3.96 (0.5 / 3.1 / 0.1) | 80K / 50K |
| **1.15x** | `bcsstk35` (`pde` / `ipopt`) | 44.07 (18.5 / 21.2 / 4.3) | `ma57` | 38.18 (8.5 / 22.9 / 4.3) | 3.24M / 2.95M |
| **1.15x** | `F1` (`pde` / `ipopt`) | 8015.47 (448.2 / 7420.6 / 157.2) | `ma57` | 6980.33 (3029.7 / 3751.9 / 126.8) | 271.7M / 171.1M |
| **1.14x** | `HTC_336_4438` (`kkt` / `ipopt`) | 68.25 (30.6 / 27.7 / 9.5) | `ma57` | 60.00 (27.7 / 19.5 / 10.6) | 1.85M / 1.91M |
| **1.12x** | `rail_20209` (`pde` / `ipopt`) | 9.90 (6.1 / 3.0 / 0.8) | `ma57` | 8.82 (3.5 / 4.2 / 0.8) | 466K / 429K |
| **1.11x** | `qa8fm` (`spd` / `rec`) | 763.69 (345.3 / 405.8 / 11.9) | `mkl_pardiso` | 690.04 (271.0 / 400.8 / 12.6) | 27.1M / — |
| **1.10x** | `af_2_k101` (`spd` / `rec`) | 2579.87 (186.5 / 2327.1 / 67.1) | `ma97` | 2337.61 (939.5 / 1305.6 / 68.2) | 137.3M / 100.6M |

### D. Matched `T = 2` Losses vs Panua PARDISO (All **9** of 80)

| Ratio | Matrix (`class` / `track`) | `smf` wall ms (`ana`/`fac`/`sol`) | `panua_pardiso` wall ms (`ana`/`fac`/`sol`) | `smf` / `panua` `L` |
|---:|---|---:|---:|---|
| **2.98x** | `bcsstm39` (`spd` / `rec`) | 4.24 (1.8 / 2.2 / 0.3) | 1.42 (0.0 / 0.1 / 0.1) | 46.8K / 46.8K |
| **2.28x** | `SiO` (`pde` / `rec`) | 18250.15 (83.4 / 18103.3 / 63.2) | 8008.53 (411.5 / 7521.4 / 52.8) | 263.4M / 84.3M |
| **2.25x** | `SiO` (`pde` / `ipopt`) | 19250.84 (82.0 / 19037.0 / 130.3) | 8553.98 (415.7 / 8044.4 / 61.2) | 263.4M / 84.3M |
| **1.93x** | `Lin` (`pde` / `rec`) | 9844.99 (134.7 / 9646.8 / 57.9) | 5108.67 (1010.4 / 3982.0 / 71.8) | 215.0M / 95.7M |
| **1.84x** | `CurlCurl_1` (`pde` / `rec`) | 5704.87 (111.0 / 5538.0 / 57.0) | 3099.71 (1019.3 / 1927.2 / 137.7) | 184.9M / 76.9M |
| **1.73x** | `CurlCurl_1` (`pde` / `ipopt`) | 5620.63 (113.5 / 5438.2 / 61.9) | 3250.04 (1024.6 / 2027.8 / 171.1) | 184.9M / 76.9M |
| **1.73x** | `Lin` (`pde` / `ipopt`) | 9358.23 (138.8 / 9103.7 / 125.3) | 5419.01 (1045.3 / 4210.8 / 150.9) | 215.0M / 95.7M |
| **1.72x** | `gas_sensor` (`pde` / `rec`) | 1221.08 (52.7 / 1150.4 / 17.8) | 708.73 (396.0 / 283.0 / 14.5) | 69.5M / 23.1M |
| **1.56x** | `gas_sensor` (`pde` / `ipopt`) | 1113.13 (52.5 / 1038.1 / 20.6) | 712.64 (400.3 / 283.5 / 16.0) | 69.5M / 23.1M |

### E. WSL2 OS-Parity Losses (`wsl_smf` OpenBLAS vs `wsl_ssids`)

- **`T = 1` (All 8 of 82):** `SiO` `rec` **2.77x** (`17614.9` vs `6349.1` ms), `SiO` `ipopt` **2.66x** (`17653.6` vs `6629.5`), `gas_sensor` `ipopt` **1.64x** (`1438.4` vs `876.1`), `gas_sensor` `rec` **1.60x** (`1419.5` vs `887.3`), `CurlCurl_1` `ipopt` **1.48x** (`4827.7` vs `3254.5`), `CurlCurl_1` `rec` **1.47x** (`4829.2` vs `3284.7`), `Lin` `ipopt` **1.29x** (`8216.7` vs `6361.9`), `Lin` `rec` **1.28x** (`8123.4` vs `6358.8`).
- **`T = 8` (All 3 of 82):** `SiO` `ipopt` **4.49x** (`9753.8` vs `2170.4` ms), `SiO` `rec` **1.53x** (`3343.5` vs `2182.8`), `Lin` `ipopt` **1.33x** (`3835.9` vs `2888.2`). All 3 are `AutoSerial` AMD fill blowups (plus `SiO` `ipopt` 8T gap).

### F. GPU Head-to-Head Losses vs `cudss` on RTX 5080

- **`T = 8` (All 6 of 64):** `SiO` `ipopt` **7.64x** (`9047.1` vs `1184.4` ms), `SiO` `rec` **3.74x** (`4368.4` vs `1169.4`), `Lin` `ipopt` **2.03x** (`3722.0` vs `1829.4`), `Lin` `rec` **1.43x** (`2600.5` vs `1813.8`), `F1` `rec` **1.18x** (`2567.7` vs `2173.0`), `F1` `ipopt` **1.10x** (`2459.5` vs `2232.2`).
- **`T = 1` (All 27 of 64):** `SiO` `rec` **18.80x**, `SiO` `ipopt` **18.57x**, `Lin` `rec` **5.95x**, `Lin` `ipopt` **5.95x**, `CurlCurl_1` `rec` **4.18x**, `CurlCurl_1` `ipopt` **4.14x**, `gas_sensor` `ipopt` **3.79x**, `gas_sensor` `rec` **3.78x**, `F1` `rec` **3.65x**, `F1` `ipopt` **3.50x**, `thread` **3.00x**, `af_2_k101` **2.81x**, `af_0_k101` **2.81x**, `boneS01` **2.27x**, `qa8fk` `rec` **2.15x**, `qa8fk` `ipopt` **2.14x**, `BenElechi1` **1.97x**, `s3dkq4m2` **1.89x**, `m_t1` **1.88x**, `2cubes_sphere` **1.80x**, `smt` **1.68x**, `c-62` `ipopt` **1.65x**, `offshore` **1.48x**, `qa8fm` **1.47x**, `gsm_106857` `rec` **1.46x**, `gsm_106857` `ipopt` **1.42x**, `msdoor` **1.32x**.
