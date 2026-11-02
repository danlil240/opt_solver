# GPU Mixed-Precision Factorization Implementation Plan (Milestone 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find out, on CPU first, whether an FP32 factorization of the large root fronts plus FP64 GMRES-based refinement reaches smf's accuracy gate on the front-tier KKT and SPD cases. If it does, put that root-front factorization on the RTX 5080 behind a build option and measure it against smf's own CPU times and against cuDSS through `bench_replay`.

**Architecture:**
- A new **root-static path** factors every root front of at least `Control::root_static_min_rows` rows with in-order 1x1 static pivots and no delays. That is the quasidefinite mode `IpmSession` already uses, applied to the roots only.
- The path runs in FP64 (the existing in-order kernel), in FP32 on the CPU (a new kernel), or in FP32 on the GPU (cuBLAS, Tasks 5–7).
- The FP32 kernels work on a float copy of the assembled front. They write the factor back as doubles, in `ldlt_factor_front`'s layout, so the solve, `FactorKeep` and the tree code are unchanged. The solve is FP64 with an FP32-accurate factor: GMRES-IR with u_f = single and u = u_p = double.
- Accuracy is recovered by the existing `refine()` (`RefineMethod::Auto`, which escalates to FGMRES). The benchmark adapter falls back to an FP64 refactor when refinement misses the gate.
- Milestone 1 ends at a go/no-go gate (Task 4) and a measured GPU root offload (Task 7). Milestones 2–5 are outlined at the end.

**Tech Stack:** C++20, MSVC 2022, CMake, MKL/OpenBLAS (`cblas_s*` for FP32), CUDA runtime and cuBLAS from NVIDIA's redistributable archives (no nvcc in M1), GoogleTest, Python 3.12 bench tools (`tools/bench/ab_run.py`, `ab_compare.py`, `score.py`, `run.py`).

**Spec:** this doc's recommendation at `https://claude.ai/artifact/Dvh52kmg9CwrigE8smPqUL`, plus:
- `docs/roadmap.md`, Horizon 2.
- `research_notes/Front tier sparse LDLT solvers/refinement_mixed_precision.md`, sections 1, 2, 4 and 5. Its numbers:
  - FP32 GMRES-IR has a guaranteed backward regime of kappa << about 5e7. It is observed to work up to about 1e10.
  - Late-IPM kappa_eff is at least 1e20.
  - FTZ/DAZ is mandatory.
  - An FP32 factor effectively static-pivots at about 6e-8·max|A|. CuClarabel needed sqrt(eps32) = 3.45e-4.
  - Escalation ladder: classical IR, then FGMRES, then an FP64 refactor, then report failure.
- `research_notes/Front tier sparse LDLT solvers/kernels_parallelism_gpu.md`, section 7. Its numbers:
  - RTX 5080 FP64 runs at about 1/64 of its FP32 rate.
  - cuDSS 1.0 has FP32 factor downcast and FP64 emulation.
  - GPUs lose below about 1e4–1e5 unknowns.
- `docs/performance/2026-10-29-path-to-number-1.md`: cuDSS is accurate on only 4 of 21 hard_indef cases and 6–9 of 13 small_kkt cases, so accuracy is the differentiator.
- `docs/performance/2026-10-30-branches-and-scoreboard.md`: the cuDSS reference column.
- `docs/performance/2026-10-26-ipm-mode.md`: the regularized/in-order static mode and why it is off by default.

## Global Constraints

- Windows, MSVC 2022, CMake. cmake and ctest are not on PATH. Their directory is `"/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/"`.
- Build directory: the worktree's `build/`. Build with `--config Release --parallel 4` at below-normal priority (`nice -n 10`). The user games on this machine.
- The CUDA build uses a second build directory, `build-cuda/`, configured with `-DSMF_ENABLE_CUDA=ON -DSMF_BENCH_CUDSS=ON -DSMF_CUDSS_ROOT=... -DSMF_CUDA_RUNTIME_ROOT=...`. These are the same redist roots the cuDSS adapter already uses; see the comment above `option(SMF_BENCH_CUDSS ...)` in `solver/CMakeLists.txt`. The runtime root must also contain libcublas (`lib/x64/cublas.lib`, `bin/x64/cublas64_*.dll`, `cublasLt64_*.dll`).
- **No nvcc in Milestone 1.** GPU code is host C++ calling the CUDA runtime and cuBLAS only.
- `SMF_ENABLE_CUDA` is OFF by default. A build without it must compile and pass the full `ctest -j 4` unchanged.
- **Default behaviour is unchanged:**
  - `Control::root_static_min_rows = 0` (off);
  - with it at 0, factors, inertia and solutions are bit-identical to main;
  - the `ParallelDeterminism`, `TreeParallel`, `Refinement.ClassicalPathUnchanged` and `StaticPivoting.*` tests pass unchanged.
- Threads are `std::thread` only, never OpenMP. MSVC's OpenMP runtime next to MKL aborts.
- The accuracy gate is eta ≤ 1e-10 and omega ≤ 1e-8 (`score.py` `ETA_GATE`/`OMEGA_GATE`, `refinement.cpp` `kTargetEtaInf`/`kTargetOmega`). The gate is never loosened for a low-precision variant.
- cuDSS stays a reference column, never a competitor in `score.py` (ruling R43, `competitor_set`).
- Benchmarks:
  - A/B runs are interleaved, from one binary, on one machine (`tools/bench/ab_run.py`; see "Measuring" in `docs/performance/2026-11-01-matching-search-order.md`).
  - Run one at a time, and check that no `bench_replay.exe` is running first.
  - Run at below-normal priority.
  - Announce any run longer than 10 minutes. GPU runs make the machine unusable for games: announce them too.
- Tests that use CoinHSL need `SMF_HSL_LIBRARY=C:/Users/daniel/projects/opt_solver/build-deps/coinhsl-2024.5.15/CoinHSL.v2024.5.15.x86_64-w64-mingw32-libgfortran5/bin/libhsl.dll`, with that `bin` directory on PATH.
- Never push. Never commit CoinHSL files, CUDA/cuDSS archives, `*.log` files, or the `.gitignore` change.
- Code style: 2-space indent in `src/`, 4 in `tests/`. Every new `Control` or `Info` field gets a comment with the evidence for its default.

## Review Focus

1. **Front values outside FP32 range.** Entries above about 3e38 overflow to inf when cast. Entries below 1.2e-38 become subnormals or flush to zero; this is the late-IPM 1e-21 diagonal case. Such a front must take the FP64 kernel, not produce inf/NaN or silently zero pivots, and the solve must still meet the gate.
   - Task 1: `LowPrecFront.UnrepresentableReportsStatus`.
   - Task 2: `RootStatic.HugeEntriesFallBackPerFront`.
2. **An SPD front that is not positive definite in FP32** (kappa about 1e8). It must be refactored in FP64 for that front alone. It must never surface as `NotPositiveDefinite` from a matrix that is SPD in FP64.
   - Task 1: `LowPrecFront.PosdefLossReportsStatus`.
   - Task 2: `RootStatic.IllConditionedSpdFallsBackPerFront`.
3. **Late-IPM ill-conditioning where FGMRES cannot reach the gate.** The adapter must refactor in FP64, stay FP64 for the rest of the sequence, and never record an inaccurate success.
   - Task 3: `BenchmarkAdapters.SmfLowPrecFallbackIsAccurate`.
4. **One workspace refactored many times while the root-static settings and values change** (an IPM sequence, the GPU context grown and reused). Each result must equal a fresh workspace's.
   - Task 2: `RootStatic.WorkspaceReuseMatchesFresh`.
   - Task 6: `GpuRoot.RefactorSequenceMatchesFresh`.
5. **`root_static_device = Gpu` with no usable device.** This covers a CUDA build on a machine without a GPU, the device out of memory for the front, and a cuBLAS error. The front runs on the CPU FP32 kernel, `Info::gpu_fronts` stays 0, and nothing fails.
   - Task 6: `GpuRoot.NoDeviceRunsCpu` and `GpuRoot.FrontTooLargeForDeviceRunsCpu`.

---

### Task 1: FP32 BLAS wrappers and the in-order static front kernel

**Files:**
- Modify: `solver/include/smf/dense_kernel.hpp` and `solver/src/dense_kernel_lapack.cpp`: FP32 wrappers in both the `SMF_BLAS_REFERENCE` branch and the cblas branch.
- Create: `solver/include/smf/ldlt_static.hpp`, `solver/src/ldlt_static.cpp`.
- Modify: `solver/CMakeLists.txt` (add `src/ldlt_static.cpp` to `add_library(smf ...)`) and `solver/tests/CMakeLists.txt`.
- Test: `solver/tests/test_lowprec_front.cpp`, registered as `add_test(NAME LowPrecFront COMMAND test_lowprec_front)`.

**Interfaces:**
- Produces, in `dense_kernel.hpp`. These mirror the double versions' contracts, with `float` element types:
  - `void smf_sgemmt_lower_nt_sub(float *C, int m, int k, const float *A, int lda, const float *B, int ldb, int ldc);`
  - `void smf_strsm_right_lower_transpose_unit(float *B, int n_rows, int k, const float *L, int ldl, int ldb);`
- Produces, in `ldlt_static.hpp`:

```cpp
struct StaticFrontParams {
  double static_pivot = 0.0;   // > 0 (indefinite): |d| < static_pivot or non-finite -> d = sign(d)*static_pivot (sign + for 0/NaN)
  bool posdef = false;         // in-order Cholesky-style: stop at the first d <= 0 / non-finite
  bool static_pivots_as_zero = false;  // as LdltKernelParams
  int nb = 128;                // panel width
};
// In-order LDL^T of the leading num_elig columns of the f x f front F (ld,
// lower triangle). Output layout, tags, inertia, static_pivots and flops are
// exactly those of ldlt_factor_front with prm.static_pivot = min_abs_pivot =
// static_pivot (indefinite) or prm.posdef (SPD); rows are never permuted.
template <class T>
LdltKernelResult ldlt_static_front(T *F, int ld, int f, int num_elig,
                                   const StaticFrontParams &prm, int8_t *tags);

enum class LowPrecStatus { Done, NotRepresentable, NotPositiveDefinite, DeviceFailed };
struct LowPrecFrontResult { LowPrecStatus status = LowPrecStatus::Done; LdltKernelResult kr; };
// Casts F (double, ld = f) to `scratch` (resized to f*f floats), factors it
// with ldlt_static_front<float> under FTZ/DAZ, and only when status == Done
// writes the factor and Schur complement back into F as doubles. Any other
// status leaves F untouched, so the caller can run the FP64 kernel on it.
LowPrecFrontResult factor_front_fp32(double *F, int f, int num_elig,
                                     const StaticFrontParams &prm, int8_t *tags,
                                     std::vector<float> &scratch);
```

- [ ] **Step 1: Write the failing tests** in `test_lowprec_front.cpp`. The helper `random_quasidefinite(int f, int n1, unsigned seed)` builds an f x f lower front: an SPD (1,1) block of size n1, a negative-definite (2,2) block, and a random coupling block.
  - `LowPrecFront.DoubleInstanceMatchesTppInOrder`: f = 300, num_elig = 300, nb = 32. Compare `ldlt_static_front<double>` with `ldlt_factor_front` (`prm.static_pivot = prm.min_abs_pivot = 1e-8`). Expect:
    - max abs difference of the lower triangles ≤ 1e-12 · max|F|;
    - tags all 1 and equal;
    - equal inertia and `static_pivots`.
  - `LowPrecFront.DoubleInstancePartialSchur`: f = 400, num_elig = 250, same comparison. The Schur block F[250:400, 250:400] must agree to 1e-12 relative.
  - `LowPrecFront.PosdefMatchesTpp`: an SPD front with f = 257 (not a multiple of nb), `prm.posdef`, compared with `ldlt_factor_front` with `posdef = true`, same tolerances.
  - `LowPrecFront.Fp32CloseToFp64`: `factor_front_fp32` on f = 600 quasidefinite. Expect:
    - status Done;
    - ||L_fp32 − L_fp64||_max / ||L_fp64||_max ≤ 1e-4;
    - inertia equal (n1 positive, f − n1 negative);
    - `kr.flops` equal.
  - `LowPrecFront.UnrepresentableReportsStatus`:
    - one entry 1e39: status `NotRepresentable`, and F is bitwise unchanged (`memcmp` against a copy);
    - a second front whose whole (1,1) diagonal is 1e-40: also `NotRepresentable`.
  - `LowPrecFront.PosdefLossReportsStatus`: the SPD matrix `Q diag(1, ..., 1e-9) Qᵀ` (f = 200) is positive definite in double. `factor_front_fp32` with `posdef` returns `NotPositiveDefinite` and leaves F unchanged.
  - `LowPrecFront.StaticPivotsRaised`: one zero diagonal entry with an all-zero row and column in an indefinite front, `static_pivot = 1e-6`. Expect `static_pivots == 1`, and |d| at that column equal to 1e-6 in both precisions.
  - `LowPrecFront.FtzRestored`: MXCSR (`_mm_getcsr()`) is the same before and after `factor_front_fp32`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cmake --build build --config Release --target test_lowprec_front` (the binary is under `build/tests/Release/`).
Expected: compile errors, because `ldlt_static.hpp` does not exist.

- [ ] **Step 3: Implement the FP32 wrappers**
  - cblas branch: `cblas_sgemm` per column block, mirroring the non-MKL `smf_dgemmt_lower_nt_sub` loop. Under MKL use `cblas_sgemmt`/`cblas_sgemmtr`, with the same `INTEL_MKL_VERSION` guard. The triangular solve uses `cblas_strsm`.
  - Reference branch: plain loops.

- [ ] **Step 4: Implement `ldlt_static_front<T>` (explicit instantiations for float and double) and `factor_front_fp32`**
  - Right-looking blocked loop: unblocked in-order elimination inside each nb panel, then `W = L_panel·D` and `smf_*gemmt_lower_nt_sub(trailing, ..., L_panel, W)`. For T = double use the d-wrappers.
  - The representability check runs during the cast. Fail when any |a| > 1e38, or when a diagonal is nonzero but below 1e-30 while the front's max|a| ≥ 1, i.e. the value would be flushed.
  - FTZ/DAZ is set by an RAII MXCSR guard around the float factorization only.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `ctest --test-dir build -C Release -R "LowPrecFront|LDLTDense|BlasWrap" --output-on-failure`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add solver/include/smf/dense_kernel.hpp solver/src/dense_kernel_lapack.cpp solver/include/smf/ldlt_static.hpp solver/src/ldlt_static.cpp solver/CMakeLists.txt solver/tests/CMakeLists.txt solver/tests/test_lowprec_front.cpp
git commit -m "feat(kernel): in-order static front kernel in FP32 and FP64"
```

---

### Task 2: Root-static path in the factorization (CPU, FP64 or FP32)

**Files:**
- Modify: `solver/include/smf/types.hpp` (enums), `solver/include/smf/control.hpp` (fields, next to `static_pivot_in_order`), `solver/include/smf/info.hpp`, `solver/include/smf/multifrontal_serial.hpp` (`SerialFactorStats`), `solver/include/smf/factor_posdef.hpp` (`FactorKeep`), `solver/include/smf/workspace.hpp` (`NumericWorkspace`).
- Modify: `solver/src/multifrontal_serial.cpp`:
  - `process_front`, around the kernel dispatch at lines 540–571;
  - the `prm` setup in `multifrontal_factor_serial` at lines 841–900;
  - `add_stats`.
- Modify: `solver/src/factor_indef.cpp` and `solver/src/factor_posdef.cpp` (pass the root static pivot, copy the stats to `Info`/`FactorKeep`), and `solver/src/refinement.cpp` (Auto's early escalation).
- Test: `solver/tests/test_root_static.cpp`, registered as `add_test(NAME RootStatic COMMAND test_root_static)`.

**Interfaces:**
- Consumes: `ldlt_static_front`, `factor_front_fp32`, `StaticFrontParams` and `LowPrecStatus` from Task 1.
- Produces:
  - `enum class FrontPrecision { Fp64, Fp32 };` and `enum class FrontDevice { Cpu, Gpu };` in `types.hpp`.
  - `Control` fields:
    - `int root_static_min_rows = 0;`. 0 is off. Otherwise every root front (`Supernode::parent == -1`) with at least this many rows takes the root-static path.
    - `double root_static_tol = 1e-8;`. This is the static pivot relative to max|A| after scaling, as `kDelayBudgetStaticTol`.
    - `FrontPrecision root_static_precision = FrontPrecision::Fp32;`
    - `FrontDevice root_static_device = FrontDevice::Cpu;`
    - `int root_gpu_panel = 256;`. Task 5 uses it.
  - `Info` and `SerialFactorStats` fields:
    - `int root_static_fronts` (fronts that took the root-static path);
    - `int lowprec_fronts` (of those, factored in FP32);
    - `int lowprec_front_fallbacks` (an FP32 attempt that returned non-Done, then FP64);
    - `int gpu_fronts`;
    - `double root_static_flop_share` (their kernel flops / all kernel flops);
    - `double root_static_seconds` (wall time inside those kernels);
    - `double gpu_seconds`.
  - `FactorKeep::lowprec_fronts` (int).
  - `NumericWorkspace::lowprec_scratch` (`std::vector<float>`).
  - Semantics:
    - An FP64 root-static front uses `ldlt_factor_front` with `prm.static_pivot = prm.min_abs_pivot = root_static_tol·max|A|`. This is the existing in-order mode.
    - An FP32 front uses `factor_front_fp32`. On a non-Done status it falls back to the FP64 root-static kernel on the untouched F.
    - A root-static front is never APTP, tiled or small-front.
    - The SPD path keeps `posdef` and uses no static pivot.
    - `refine()` Auto treats `fkeep.lowprec_fronts > 0` like `static_pivots > 0`: escalate to FGMRES after min(3, cap) classical steps.

- [ ] **Step 1: Write the failing tests** in `test_root_static.cpp`. Use `opf_like_kkt` from `kkt_test_matrices.hpp`, a 3D Poisson SPD matrix (as in `test_poisson_2d.cpp`, extended to 3D, n = 30³), and `Solver::factor_solve`/`refine`.
  - `RootStatic.OffIsBitIdentical`: with `root_static_min_rows = 0`, compare the factor values, inertia and solution against the same matrix factored with a default `Control`. Use `memcmp` equality.
  - `RootStatic.Fp64RootMeetsGate`: `opf_like_kkt(400, 300, 7)`, min rows 64, Fp64. Expect:
    - `info.root_static_fronts >= 1`, `lowprec_fronts == 0`;
    - after `refine`, `eta_inf <= 1e-10` and `omega <= 1e-8`.
  - `RootStatic.Fp32RootMeetsGateWithFgmres`: the same matrix with Fp32. Expect:
    - `lowprec_fronts >= 1`, and `root_static_flop_share > 0` and ≤ 1;
    - `refine` succeeds with `iterations <= 10`.
  - `RootStatic.Fp32SpdRoot`: 3D Poisson, Fp32, min rows 256. Expect `lowprec_fronts == 1`, the refine gate met, and `info.num_neg == 0`.
  - `RootStatic.EveryLargeRootOfAForest`: a block-diagonal matrix of two independent `opf_like_kkt` blocks. Expect `root_static_fronts == 2`.
  - `RootStatic.HugeEntriesFallBackPerFront`: scale one row/column pair of the KKT matrix by 1e20 with `scaling = None`. Expect `lowprec_front_fallbacks == 1`, and the refine gate met.
  - `RootStatic.IllConditionedSpdFallsBackPerFront`: an SPD matrix with a 1e-9-spread root block. Expect a factor status of Success (not NotPositiveDefinite), `lowprec_front_fallbacks == 1`, and the gate met.
  - `RootStatic.WorkspaceReuseMatchesFresh`: on one `NumericWorkspace`, factor the sequence (Fp32, values v1) → (off, v2) → (Fp32, v3) → (Fp64, v1). Each factor must equal a fresh-workspace factor of the same settings and values bit for bit.
  - `RootStatic.ParallelMatchesSerial`: `num_threads = 4` against 1 with Fp32 and the same min rows. Expect equal `root_static_fronts` and `lowprec_fronts`, and both meet the gate.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cmake --build build --config Release --target test_root_static`
Expected: compile errors, because `root_static_min_rows` is not a member of `Control`.

- [ ] **Step 3: Implement**
  - In `multifrontal_factor_serial`, add `FactorContext` members `root_static` (pivot value) and `root_min_rows`.
  - In `process_front`, compute `const bool root_static = is_root && cx.root_min_rows > 0 && f >= cx.root_min_rows;` before choosing `tiled`/`aprm`/`small`, and force those off when it is true. Dispatch as described in Interfaces, and count the stats in `P.st`.
  - The FP32 path's static pivot uses the same `root_static_tol · matrix_max_abs` as the FP64 path, computed in `factor_indef.cpp` next to `budget_static_pivot`.
  - Copy `lowprec_fronts` to `fkeep` in both drivers.
  - In `refinement.cpp`, extend the Auto early-escalation condition.

- [ ] **Step 4: Run the tests to verify they pass, plus the regression set**

Run: `ctest --test-dir build -C Release -R "RootStatic|Refinement|StaticPivoting|ParallelDeterminism|TreeParallel|IpmRegularized" --output-on-failure`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add solver/include/smf solver/src/multifrontal_serial.cpp solver/src/factor_indef.cpp solver/src/factor_posdef.cpp solver/src/refinement.cpp solver/tests/test_root_static.cpp solver/tests/CMakeLists.txt
git commit -m "feat(factor): root-static path (in-order static roots, FP64 or FP32 on CPU)"
```

---

### Task 3: Bench plumbing — env knobs, refine and FP64 fallback in the smf adapter, records, study table

**Files:**
- Modify: `solver/benchmarks/bench_replay.cpp`:
  - env parsing next to `SMF_BENCH_REFINE_METHOD` (about line 620);
  - the per-step loop and the record fill (lines 700–810 and 860–895);
  - the `--print-steps` line.
- Modify: `solver/benchmarks/adapters/adapter.hpp` (`AdapterStats`), `solver/benchmarks/adapters/hsl.cpp` (`SmfSolverAdapter`, lines 268–436), `solver/benchmarks/common/result.hpp` and the JSONL writer that emits its fields.
- Create: `tools/bench/lowprec_study.py`, `tools/bench/test_lowprec_study.py`.
- Test: `solver/tests/test_benchmark_adapters.cpp`.

**Interfaces:**
- Consumes: the `Control` and `Info` fields from Task 2.
- Produces:
  - Environment variables:
    - `SMF_BENCH_ROOT_STATIC` = `off` | `fp64` | `fp32` | `gpu`. `gpu` sets device Gpu with precision Fp32. Anything other than `off` defaults the minimum to 2048 rows.
    - `SMF_BENCH_ROOT_STATIC_MIN_ROWS` (int).
    - `SMF_BENCH_ROOT_STATIC_TOL` (double).
    - `SMF_BENCH_REFINE_MAX_STEPS` (int, `Control::refine_max_steps`).
  - `AdapterStats` fields: `root_static_fronts`, `lowprec_fronts`, `gpu_fronts` (int); `root_static_flop_share`, `gpu_ms` (double); `lowprec_fallbacks` (int, adapter-level FP64 refactors).
  - Record fields (`result.hpp`, -1 = not reported):
    - the same six fields as `AdapterStats`;
    - `int refine_iterations_max` (over all steps of the sequence);
    - `int lowprec_fallback_step` (first step that fell back; -1 none).
  - `SmfSolverAdapter::solve` refines whenever `static_pivots_ > 0 || lowprec_fronts_ > 0 || refine_always_`. When the factor had `lowprec_fronts > 0` and `refine()` returns non-Success:
    1. set `root_static_min_rows = 0` in its `ctrl_` for the rest of the adapter's life (sticky);
    2. refactor the current values;
    3. solve and refine again from the saved right-hand side;
    4. `++lowprec_fallbacks`.

    All of this is inside the timed `solve_ms`.
  - `--print-steps` appends ` rs=%d fp32=%d share=%.3f gpu=%d gpu_ms=%.3f refine_it=%d fb=%d`.
  - `lowprec_study.py --records DIR [DIR...] --manifest M [--split tuning] [--min-share 0.5] --output OUT.md` writes one row per case-track whose `root_static_flop_share >= --min-share` in the first DIR. For every DIR (variant) it gives:
    - accurate (score.py `_validation_ok`);
    - `refine_iterations_max`;
    - `lowprec_fallback_step`;
    - factor ms and wall ms as medians over trials.

    The summary gives, per variant, the zero-fallback fraction, the median `refine_iterations_max` over zero-fallback rows, the accurate count, and a shifted geomean (10 ms) of factor time against the first DIR.

- [ ] **Step 1: Write the failing tests**
  - `BenchmarkAdapters.SmfLowPrecFallbackIsAccurate`. Matrix: `opf_like_kkt(400, 300, 3)`, with the multiplier block's diagonal set to −1e-12 and primal diagonals spread over 1e-10 to 1e10 (a late-IPM shape). Configure `root_static_min_rows = 64`, Fp32, `refine_max_steps = 2`. Then `analyse`, `factor` and `solve` the all-ones right-hand side. Expect:
    - `stats().lowprec_fallbacks == 1`;
    - the solution's eta ≤ 1e-10 and omega ≤ 1e-8, checked with `validation.hpp`;
    - a second `factor` + `solve` reports `lowprec_fronts == 0` (sticky).
  - `BenchmarkAdapters.SmfLowPrecReportsStats`: a well-conditioned case. Expect `lowprec_fronts >= 1`, `lowprec_fallbacks == 0`, and `refine_iterations >= 1`.
  - `test_lowprec_study.py`, `test_summary_counts_fallbacks`: two synthetic variants of three records each. The script must report a zero-fallback fraction of 2/3 and the right median.

- [ ] **Step 2: Run them to verify they fail**

Run: `ctest --test-dir build -C Release -R BenchmarkAdapters --output-on-failure` and `python -m pytest tools/bench/test_lowprec_study.py -q`
Expected: compile error (unknown `lowprec_fallbacks`), and ModuleNotFoundError.

- [ ] **Step 3: Implement** the adapter, `bench_replay`, the record fields and `lowprec_study.py` as specified in Interfaces. `lowprec_study.py` imports `score` the way `ab_compare.py` does.

- [ ] **Step 4: Run the tests to verify they pass**

Same commands as Step 2. Expected: pass.

- [ ] **Step 5: Smoke run.** Run `build/benchmarks/Release/bench_replay.exe --case cops_pinene_400 --track ipopt --print-steps` with `SMF_BENCH_ROOT_STATIC=fp32 SMF_BENCH_ROOT_STATIC_MIN_ROWS=256`.
Expected: step lines with `rs>=1 fp32>=1`, and a record with `lowprec_fronts` set.

- [ ] **Step 6: Commit**

```bash
git add solver/benchmarks tools/bench/lowprec_study.py tools/bench/test_lowprec_study.py solver/tests/test_benchmark_adapters.cpp
git commit -m "bench: root-static knobs, FP64 fallback in the smf adapter, lowprec study table"
```

---

### Task 4: CPU accuracy study and the go/no-go gate for the GPU tasks

**Files:**
- Create: `docs/performance/<run-date>-fp32-root-study.md`. Records go to `benchmarks/results/fp32_root_study/`, which is not committed beyond the summaries, as with earlier result folders.

**Interfaces:**
- Consumes: Task 3's env knobs, records and `lowprec_study.py`.
- Produces:
  - the candidate list: case-tracks with `root_static_flop_share >= 0.5` at 2048 rows;
  - the chosen `root_static_tol` and `root_static_min_rows` for Tasks 6–7;
  - a GO or NO-GO line at the top of the write-up.

- [ ] **Step 1: Candidate scan.** Run `ab_run.py` (1 round, 1 thread) with these settings:
  - classes `large_kkt,hard_indef,spd`;
  - split `tuning`;
  - one variant, `fp32=SMF_BENCH_ROOT_STATIC=fp32`;
  - output `benchmarks/results/fp32_root_study/scan`.

  Then run `lowprec_study.py --min-share 0.5` on it.
  Expected: the candidate list.
  - If fewer than 8 case-tracks qualify, repeat with `SMF_BENCH_ROOT_STATIC_MIN_ROWS=1024` and record that.
  - The excluded inaccurate-everywhere cases (d_pretok, turon_m, robot_arm_8k rec) are listed but never counted.

- [ ] **Step 2: Study run.** Announce first: about 30–60 minutes on one core. Run `ab_run.py` with `--rounds 3 --threads 1 --cases <candidates>` and these variants:
  - `fp64=SMF_BENCH_ROOT_STATIC=off`;
  - `s64=SMF_BENCH_ROOT_STATIC=fp64`;
  - `s32a=SMF_BENCH_ROOT_STATIC=fp32,SMF_BENCH_ROOT_STATIC_TOL=1e-8`;
  - `s32b=...,SMF_BENCH_ROOT_STATIC_TOL=1e-6`;
  - `s32c=...,SMF_BENCH_ROOT_STATIC_TOL=3.45e-4`;
  - `s32d=SMF_BENCH_ROOT_STATIC=fp32,SMF_BENCH_ROOT_STATIC_TOL=<best of a-c>,SMF_BENCH_REFINE_MAX_STEPS=20`.

  Then run `lowprec_study.py --records fp64 s64 s32a s32b s32c s32d`. `s64` against `s32*` separates the static-pivoting error from the FP32 error.

- [ ] **Step 3: Apply the gate** to the best FP32 variant. "Best" means the most zero-fallback rows, with ties broken by the lower median `refine_iterations_max`.
  - **G1:** zero fallbacks on at least 70% of the candidate case-tracks.
  - **G2:** median `refine_iterations_max` ≤ 6 on the zero-fallback rows. Each iteration is one FP64 solve, so this caps the solve cost a GPU factor must win back.
  - **G3:** every case-track that `fp64` solves accurately is accurate in the FP32 variant (after fallback). There are no new failures in `score.py --split tuning` on the variant's records.

  GO means G1, G2 and G3 all hold. On NO-GO, stop after Step 5: Tasks 5–7 are not run, and the write-up recommends the M2-alt route in the outline below. Do not tune thresholds on the held-out split.
- [ ] **Step 4: Validate the choice on held-out.** Run the same two variants (`fp64` and the best FP32) on `--split heldout` for the same classes and candidate rule. Report G1–G3 there too. This is validation only, with no retuning.
- [ ] **Step 5: Write the study up**, using the write-up layout of `docs/performance/2026-11-02-factor-side-delays.md`:
  - the candidate table: case, track, n, root rows, flop share;
  - the per-variant summary;
  - where fallbacks start along each IPOPT sequence (`lowprec_fallback_step` against sequence length), which is the late-IPM question;
  - the FP32 versus FP64 CPU root-kernel time (`root_static_seconds`), as a by-product;
  - the GO/NO-GO line.

  Commit:

```bash
git add docs/performance/*-fp32-root-study.md
git commit -m "docs(performance): FP32 root-front + GMRES-IR accuracy study"
```

---

### Task 5: CUDA build option and the cuBLAS root-front kernel (GO only)

**Files:**
- Modify: `solver/CMakeLists.txt`:
  - hoist the cudart `find_*` calls out of the `SMF_BENCH_CUDSS` block into a shared CUDA block;
  - add `option(SMF_ENABLE_CUDA ...)` and `find_library(SMF_CUBLAS_LIBRARY cublas ...)`;
  - extend `_smf_cuda_dlls` and `smf_stage_runtime_dlls` to the option;
  - add `SMF_HAS_CUDA=1` to the `smf` public compile definitions when it is on;
  - add a configuration summary line.
- Create: `solver/src/gpu/gpu_front.hpp` (internal), `solver/src/gpu/gpu_front_cuda.cpp` (built with the option), `solver/src/gpu/gpu_front_stub.cpp` (built without it).
- Modify: `solver/include/smf/workspace.hpp`: forward declaration plus deleter, and the `NumericWorkspace::gpu` member.
- Test: `solver/tests/test_gpu_front.cpp`, `add_test(NAME GpuFront ...)`, with `smf_stage_runtime_dlls(test_gpu_front)`. Tests include the internal header as `#include "../src/gpu/gpu_front.hpp"`, as `test_ldlt_aptp.cpp` includes `../src/parallel/tile_tasks.hpp`.

**Interfaces:**
- Consumes: `StaticFrontParams`, `LowPrecFrontResult`, `LowPrecStatus` and `ldlt_static_front<float>` from Task 1. The last one factors each nb x nb diagonal block on the host.
- Produces:

```cpp
// workspace.hpp
struct GpuFrontContext;
struct GpuFrontDeleter { void operator()(GpuFrontContext *) const noexcept; };
// NumericWorkspace: std::unique_ptr<GpuFrontContext, GpuFrontDeleter> gpu;  (created lazily)

// gpu/gpu_front.hpp
struct GpuStatus { bool available = false; std::string reason; std::string device; };
const GpuStatus &gpu_status();     // cached; stub: {false, "built without SMF_ENABLE_CUDA"}
std::unique_ptr<GpuFrontContext, GpuFrontDeleter> make_gpu_front_context();  // nullptr if unavailable
std::size_t gpu_context_device_bytes(const GpuFrontContext &);
void set_gpu_disabled_for_test(bool disabled);   // gpu_status().available reads false while set
// FP32 factor of F (double, ld = f) on the device; same contract as
// factor_front_fp32. DeviceFailed (allocation, cuBLAS/CUDA error, or
// f*f*4 bytes > 80% of free device memory) leaves F untouched.
LowPrecFrontResult gpu_factor_front_fp32(GpuFrontContext &ctx, double *F, int f, int num_elig,
                                         const StaticFrontParams &prm, int8_t *tags, int panel,
                                         std::vector<float> &host_scratch);
```

- The algorithm, which the signature does not determine (right-looking, host panel, no nvcc). For each panel k of width `panel`:
  1. Cast F to float on the host. Do the representability check of Task 1 and return `NotRepresentable` from the host side. Upload the lower triangle with one `cudaMemcpyAsync` from a pinned buffer that the context owns and grows.
  2. Download the nb x nb diagonal block. Factor it on the host with `ldlt_static_front<float>` (num_elig = nb, the same static pivot), under FTZ/DAZ. Upload it, together with 1/d.
  3. `cublasStrsm` (right, lower, transpose, unit) on the rows below the panel. Keep the result as W = L·D, then scale with `cublasSdgmm` by 1/d to get L.
  4. Trailing update over all remaining rows, including the Schur rows past num_elig: `cublasSsyrkx(lower, N, m, nb, −1, L, W, 1, C)`.
  5. Download the lower triangle once at the end. Write it back as doubles with D on the diagonal, in `ldlt_factor_front`'s layout. Tags are all 1; inertia and static pivots come from the host blocks; flops are as the CPU kernel's.

  Device buffers and the handle/stream live in the context and are only grown.

- [ ] **Step 1: Write the failing tests.** Every test starts with `if (!gpu_status().available) GTEST_SKIP() << gpu_status().reason;`, except the stub test.
  - `GpuFront.MatchesCpuFp32`: f = 1500 quasidefinite (Task 1 helper, copied), panel 256. Compare against `factor_front_fp32`:
    - lower triangle max abs difference ≤ 1e-4 · max|L|;
    - equal tags, inertia and `static_pivots`.
  - `GpuFront.PartialEliminationSchur`: f = 2000, num_elig = 1200. The Schur block agrees with the CPU FP32 result to 1e-4 relative.
  - `GpuFront.ReusesBuffers`: f = 2000, then f = 1000, on one context. `gpu_context_device_bytes` is unchanged after the second call, and both match the CPU result.
  - `GpuFront.UnrepresentableLeavesFrontUntouched`: a 1e39 entry returns `NotRepresentable`, with F `memcmp`-equal.
  - `GpuFront.StubReportsUnavailable`: compiled only without `SMF_HAS_CUDA`. Expect `!gpu_status().available` and a reason containing "SMF_ENABLE_CUDA".

- [ ] **Step 2: Run them to verify they fail**

Run: `cmake --build build-cuda --config Release --target test_gpu_front`
Expected: compile errors, because `gpu/gpu_front.hpp` does not exist.

- [ ] **Step 3: Implement** the CMake changes, the stub and the CUDA file as specified. CUDA/cuBLAS errors map to `DeviceFailed`, with the call name kept in a context field for logging.

- [ ] **Step 4: Run them to verify they pass**

Run: `ctest --test-dir build-cuda -C Release -R GpuFront --output-on-failure`. Then, in `build/` (no CUDA): `ctest --test-dir build -C Release -j 4`.
Expected: GpuFront passes on the RTX 5080. The default build passes in full, with the stub test running.

- [ ] **Step 5: Commit**

```bash
git add solver/CMakeLists.txt solver/src/gpu solver/include/smf/workspace.hpp solver/tests/test_gpu_front.cpp solver/tests/CMakeLists.txt
git commit -m "feat(gpu): SMF_ENABLE_CUDA option and cuBLAS FP32 root-front kernel"
```

---

### Task 6: GPU device in the root-static path (GO only)

**Files:**
- Modify: `solver/src/multifrontal_serial.cpp` (the root-static dispatch from Task 2), `solver/src/factor_indef.cpp` and `solver/src/factor_posdef.cpp` (copy `gpu_fronts`/`gpu_seconds`), `solver/benchmarks/bench_replay.cpp` (`SMF_BENCH_ROOT_STATIC=gpu` already maps to the device; also print `gpu_status().device` once on stderr).
- Test: `solver/tests/test_root_static.cpp` (GpuRoot cases; they skip without a device except `NoDeviceRunsCpu`).

**Interfaces:**
- Consumes: `gpu_status`, `make_gpu_front_context`, `gpu_factor_front_fp32` and `set_gpu_disabled_for_test` (Task 5); the `Control::root_static_device` and `root_gpu_panel` fields and the stats from Task 2.
- Produces the dispatch for a root-static FP32 front with device Gpu:
  - when `gpu_status().available`, create `ws.gpu` lazily and call `gpu_factor_front_fp32`;
  - `Done` counts `gpu_fronts` and `lowprec_fronts`, and adds the call's wall time to `gpu_seconds`;
  - `DeviceFailed` (or no device) runs `factor_front_fp32` on the CPU;
  - `NotRepresentable`/`NotPositiveDefinite` run the FP64 kernel and count `lowprec_front_fallbacks`;
  - the SPD posdef variant of the device kernel stops at d ≤ 0, exactly as Task 1.

- [ ] **Step 1: Write the failing tests**
  - `GpuRoot.FactorSolveReachesGate`: the 3D Poisson SPD case (n = 40³) and `opf_like_kkt(3000, 2000, 5)`, min rows 1024, device Gpu. Expect `info.gpu_fronts >= 1`, and the refine gate met.
  - `GpuRoot.MatchesCpuFp32Path`: the same matrix with device Cpu. Expect equal inertia and `lowprec_fronts`, and solutions within 1e-9 relative after refinement.
  - `GpuRoot.RefactorSequenceMatchesFresh`: 5 refactors with changing values on one workspace. Each solution agrees with a fresh-workspace GPU factor's to 1e-12 relative, and the context's device bytes are constant after the first.
  - `GpuRoot.NoDeviceRunsCpu`: under `set_gpu_disabled_for_test(true)`, expect `gpu_fronts == 0`, `lowprec_fronts >= 1`, and the gate met. This test runs in every build.
  - `GpuRoot.FrontTooLargeForDeviceRunsCpu`: a test hook that caps the device budget at 1 MB (`set_gpu_memory_cap_for_test(std::size_t)`, added to `gpu_front.hpp`). Expect `gpu_fronts == 0` and `lowprec_fronts >= 1`.

- [ ] **Step 2: Run them to verify they fail**

Run: `ctest --test-dir build-cuda -C Release -R RootStatic --output-on-failure`
Expected: GpuRoot cases fail with `gpu_fronts == 0`.

- [ ] **Step 3: Implement** the dispatch and the memory-cap hook.

- [ ] **Step 4: Run them to verify they pass**, in both build directories, then the full `ctest -j 4` in `build/`.

- [ ] **Step 5: Commit**

```bash
git add solver/src solver/benchmarks/bench_replay.cpp solver/tests/test_root_static.cpp
git commit -m "feat(gpu): root-static fronts on the GPU, CPU FP32 fallback"
```

---

### Task 7: Benchmark against smf CPU and cuDSS; Milestone 1 exit gate (GO only)

**Files:**
- Create: `docs/performance/<run-date>-gpu-root-offload.md`. Records go to `benchmarks/results/gpu_root_m1/`.
- Modify: `docs/roadmap.md`. Add a "Status" bullet under Horizon 2 with the M1 result and links to both write-ups.

**Interfaces:**
- Consumes: the `build-cuda` `bench_replay.exe`, Task 4's candidate list and settings, `ab_run.py`, `run.py`, `score.py`, `ab_compare.py` and `lowprec_study.py`.
- Produces: the M1 exit verdict and the per-phase table that Milestone 2 plans from.

- [ ] **Step 1: smf variants.** Announce first: a GPU run. Run `ab_run.py --binary build-cuda/benchmarks/Release/bench_replay.exe --cases <candidates> --rounds 3` in two passes:
  1. `--threads 1`, variants `cpu=SMF_BENCH_ROOT_STATIC=off`, `cpu32=SMF_BENCH_ROOT_STATIC=fp32,...` and `gpu=SMF_BENCH_ROOT_STATIC=gpu,...`, with Task 4's tolerance and minimum rows;
  2. `--threads 16`, variants `cpu` and `gpu`.
- [ ] **Step 2: cuDSS on the same cases.** Run `python tools/bench/run.py --manifest benchmarks/manifests/fronttier.json --binary build-cuda/benchmarks/Release/bench_replay.exe --solvers cudss --cases <candidates> --trials 3 --allow-short-trials --output benchmarks/results/gpu_root_m1/cudss`.
- [ ] **Step 3: Score.**
  - `ab_compare.py --a .../t16/cpu --b .../t16/gpu --metric wall` and `--metric factor`.
  - `ab_compare.py --a .../t1/cpu --b .../t1/gpu`.
  - For each smf variant, `score.py --records <dir with that variant's records.jsonl plus cudss/records.jsonl> --manifest benchmarks/manifests/fronttier.json`, reading cuDSS's reference column.
  - Per-phase medians from the records: analyse, first factor (`gpu_ms` inside it), refactor, solve + refine.
- [ ] **Step 4: Apply the M1 exit gate** on the candidate case-tracks.
  - **E1, accuracy:** the `gpu` variant is accurate on every case-track that `cpu` is, using `score.py`'s gate.
  - **E2, speed:** the shifted geomean (10 ms) of wall time is ≤ 0.75 for gpu at 16 threads against cpu at 16 threads (smf's best CPU time), and ≤ 0.5 for gpu at 1 thread against cpu at 1 thread.
  - **E3, report only:** gpu against cuDSS: wall-time ratio, accurate counts, and refactor time per step.

  If E1 fails, keep the feature off and report it. If E2 fails, report the phase that ate the gain (transfer, host panel, refinement solves) as the first item of the M2 plan. Do not change thresholds after seeing results.
- [ ] **Step 5: Write up and commit.** Use the layout of the earlier write-ups:
  - the hardware line (RTX 5080, driver, CUDA runtime and cuDSS versions from `gpu_status().device` and the cuDSS adapter's version string);
  - the tables from Step 3 and E1–E3;
  - the roadmap status bullet.

```bash
git add docs/performance/*-gpu-root-offload.md docs/roadmap.md
git commit -m "docs(performance): GPU FP32 root-front offload vs smf CPU and cuDSS (M1)"
```

---

## Later milestones (outline; each gets its own plan after the previous gate)

- **M2: CPU/GPU tree split and device residency.**
  - Move the whole top part (`partition_subtrees`' top fronts above a flop cutoff), not only the roots, to the GPU. Contribution blocks stay on the device. Extend-add runs as device kernels, which introduces nvcc and CMake's CUDA language behind the same option.
  - Overlap the panel factorization with the trailing update (lookahead) and transfers with compute.
  - Store FP32 factors to halve factor memory, and upcast per front in the solve.
  - Gate: E2 on all large_kkt/hard_indef/spd cases with top-part flop share ≥ 0.5.
- **M2-alt (if Task 4 is NO-GO):** use cuDSS's own FP32-downcast and FP64-emulation modes through a `SMF_BENCH_CUDSS_MODE` knob in `cudss_adapter.cpp`, and decide whether smf should call cuDSS for the top part instead of owning GPU kernels.
- **M3: Precision escalation inside the library.**
  - Move the FP32→FP64 ladder from the bench adapter into `Solver`/`IpmSession`. The fallback is sticky, with a retry backoff, as the re-analysis lesson in `2026-10-26-ipm-mode.md` suggests.
  - Report static/FP32 pivots to IPOPT's inertia logic.
  - Gate: IPOPT end to end on the plan-3 public sets, with no new failures and no iteration-count increase above 5%.
- **M4: GPU solve.**
  - Run the triangular solves and the FGMRES vectors on the device when the factor is device-resident.
  - Multi-RHS blocking for refinement.
  - Gate: solve + refine time ≤ smf CPU's at 16 threads on the large replays.
- **M5: Horizon 2 exit.** Run a full front-tier board with smf-GPU and cuDSS (all modes) on the RTX 5080, at the same accuracy gate. Exit criterion from the roadmap: beat smf's best CPU time on the large IPOPT KKT replays and be competitive with cuDSS on the same card.
