# smf

smf is a sparse symmetric multifrontal direct solver written in C++20. It factors positive-definite matrices with Cholesky and indefinite matrices with LDLᵀ, and it is built to be the linear-algebra engine inside an interior-point (barrier) optimizer such as IPOPT. Its design follows HSL_MA97: a symbolic analysis that picks an ordering, an optional matching-based scaling and an assembly tree, then a numeric factorization over that tree with threshold pivoting and delayed pivots, then forward and backward solves with iterative refinement.

What it adds on top of that design:

- An analysis that chooses between AMD and METIS nested dissection per matrix, with a parallel nested dissection for large SPD cases.
- A matching trial during analysis that decides whether max-product matching scaling is worth its cost, and a scale-on-delay rescue that applies the scaling only when an unscaled factor delays too many pivots.
- A cost-model amalgamation of the assembly tree.
- Tree-level parallelism on a persistent worker pool, with tile tasks inside the largest fronts, and a deterministic mode that gives bitwise-identical factors at every thread count.
- An interior-point mode (`smf::IpmSession`) that can reuse pivot sequences and the symbolic analysis across IPOPT iterations. Its features are built and gated but stay off by default.
- A drop-in `libsmf_ma97` plugin that IPOPT loads in place of HSL MA97, an IPOPT `TSymLinearSolver` adapter and a C API.

The code lives in `solver/`. Benchmark tooling is in `tools/bench/`, results in `benchmarks/results/`, and the performance write-ups in `docs/performance/`.

## Building

You need CMake 3.20 or later, a C++20 compiler (MSVC 2022, GCC 11+ or Clang 14+), BLAS and LAPACK (MKL or OpenBLAS), and GoogleTest for the tests. SuiteSparse AMD and METIS 5 are optional but recommended, since the ordering policy relies on both.

```bash
cmake -S solver -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
ctest --test-dir build --output-on-failure
```

The options most people touch:

| Option | Default | What it does |
|---|---|---|
| `SMF_USE_MKL` | platform dependent | Use Intel MKL instead of OpenBLAS |
| `SMF_USE_METIS` | ON | METIS nested dissection ordering |
| `SMF_USE_SUITESPARSE_AMD` | ON | SuiteSparse AMD ordering |
| `SMF_DETERMINISTIC` | OFF | Reproducible factors at every thread count |
| `SMF_BUILD_MA97_PLUGIN` | ON | Build `libsmf_ma97` for IPOPT's dynamic loader |
| `SMF_BUILD_IPOPT_ADAPTER` | OFF | Build the IPOPT `TSymLinearSolver` adapter |
| `SMF_BUILD_C_API` | OFF | Build the C ABI wrapper (`smf_c.h`) |
| `SMF_BUILD_BENCHMARKS` | ON | Build the benchmark drivers in `solver/benchmarks/` |
| `SMF_SANITIZE` | OFF | Build with address and UB sanitizers |

`solver/README.md` has the full option list and dependency notes.

## Using it

The C++ API follows the MA97 split into analyse, factor and solve. The analysis is reused for every refactor with the same pattern.

```cpp
#include "smf/solver.hpp"

smf::Solver solver;
smf::Control ctrl;
smf::Info info;

auto ak = solver.analyse(A, ctrl, info);         // A is lower-triangular CSC
smf::FactorKeep fkeep;
solver.factor(*ak, ctrl, info, fkeep);
solver.solve(fkeep, ctrl, info, b.data(), n);    // b is overwritten with x
```

To use smf from IPOPT, build the plugin and point IPOPT's `hsllib` option at `libsmf_ma97` (`smf_ma97.dll` on Windows) with `linear_solver ma97`.

### Fast mode (opt-in static pivoting)

The default is threshold pivoting (`smf::Pivoting::Threshold`): delayed pivots, the scale-on-delay rescue and exact inertia. That default is the accurate mode, and every number under "Where it stands" is measured with it.

smf also has an opt-in fast mode, `smf::Pivoting::Static`. It does PARDISO-like static pivoting: no pivot is ever delayed, and a pivot smaller than 1e-8 × max|A| is raised to that value (a perturbed pivot). It can be much faster where threshold pivoting cascades delays. On the pde_control KKT systems at 8 threads it takes 0.82-0.89 of PARDISO's time. It gives up guarantees to get there:

- **Accuracy is not guaranteed on hard indefinite systems.** With perturbed pivots the factor is of a perturbed matrix, and pivot growth is not bounded, so a solve can miss the accuracy gate even after refinement. Badly scaled KKT systems (OPF, COPS-like) can fail as singular or inaccurate where the default succeeds.
- **The reported inertia is that of the perturbed matrix** when any pivot was perturbed (`Info::inertia_exact` is then false). That can mislead an interior-point method's inertia correction. The IPOPT plugin keeps its guard in fast mode: a factor with a perturbed pivot is reported to IPOPT as singular (MA97 flag 7), so IPOPT regularizes and refactors. On a KKT system that static pivoting keeps perturbing whatever the regularization, IPOPT can then raise the regularization until it gives up, so in fast mode an inaccurate factor can become a failed IPOPT solve.
- **The delay and scale-on-delay rescue is disabled.** There are no delays to rescue.
- **Solves are always refined.** Each static factor sets `FactorKeep::static_path`. Full solves through `Solver::factor_solve`, `IpmSession`, the C API and the plugin are then refined to a componentwise backward error of 1e-10 (`Control::refine_omega_target`), with at least one step when the pivot growth passes 1e8. After a plain `Solver::factor` and `Solver::solve`, call `smf::refine()` when `smf::refinement_advised(fkeep)` is true.

How to turn it on:

| Interface | Setting |
|---|---|
| C++ | `ctrl.pivoting = smf::Pivoting::Static;` |
| C API | `smf_set_pivoting(analysis, SMF_PIVOTING_STATIC);` before `smf_factor()` |
| IPOPT plugin | environment `SMF_PIVOTING=static`, read once per process (`threshold` or unset: the default) |
| Benchmark harness | `SMF_PIVOTING=static` (or `SMF_BENCH_PIVOTING`) on every track |

The library itself reads no environment variables. Only the plugin and the harness do. Fast-mode benchmark results are reported apart from the headline numbers.

## Where it stands

The reference board is the front-tier manifest: 144 case-tracks (78 matrices) of KKT, hard indefinite and SPD matrices, each replayed with the "recommended" and "ipopt" pivoting settings, split into tuning and held-out sets. Scores are smf's time divided by the other solver's, as a shifted geometric mean with a 10 ms shift, so below 1 means smf is faster. A case only counts when both solvers pass the accuracy gate (eta ≤ 1e-10, omega ≤ 1e-8), so the headline geomeans below cover fewer than 144 case-tracks: per thread count, the 5 that smf fails and the cases where no rival is accurate drop out.

Against the fastest accurate other solver on each case, smf scores **0.72 at 1 thread and 0.61 at 8 threads** (0.718 over 132 of the 144 case-tracks at 1 thread and 0.611 over 125 at 8 threads; the rest are the 5 smf fails plus 7 at 1 thread and 14 at 8 threads where no rival is accurate). That is the November 3-4 board on an i9-12900K, main with PR #36 (parallel nested dissection candidates). All four rivals (MA57 at 1 thread only, MA97, MKL PARDISO and MUMPS) were rerun fresh in the same session as smf, with smf interleaved per case, threaded MKL, and 3 trials plus 1 warmup. SSIDS does not run on that machine. The earlier Ryzen boards, which also include SSIDS and CHOLMOD, gave 0.80 at 1 thread and 0.56 at 16 threads. MA27 and cuDSS are measured as reference columns only.

**The earlier boards flattered smf.** The previous 12900K board (0.63 at 1 thread, 0.45 at 8 threads) ran MKL PARDISO in 1x1-only pivoting mode and MA57 with automatic scaling on the ipopt track. PR #34 fixed both. PARDISO now uses Bunch-Kaufman pivoting, perturbation 1e-8 and MKL's default parallel METIS ordering, and MA57 on the ipopt track uses IPOPT's own defaults (no automatic scaling). The scores of 0.63 and 0.45 should not be compared with the 0.72 and 0.61 above as a regression; the rivals are simply measured correctly now.

Head to head at 1 thread (October 30 board, Ryzen), smf against each solver: MA57 0.60, MA97 0.62, MKL PARDISO 0.58, SSIDS 0.37, MUMPS 0.26. smf also solves more case-tracks accurately than any of them. The one pairing it still loses is MA27 on small KKT systems, where MA27's low overhead wins.

smf fails 5 of the 144 case-tracks at each thread count: d_pretok (both tracks), turon_m (both tracks) and robot_arm_8k (recommended). No other solver solves those 5 accurately either.

Accuracy per thread count, counted on all 144 case-tracks of the board. A case-track counts as accurate for a solver when every measured trial (warmup excluded) passes the accuracy gate; a case-track a solver did not finish (timeout, out of memory, singular) counts as not accurate:

| solver | accurate at 1 thread | accurate at 8 threads |
|---|---:|---:|
| smf | 139 / 144 | 139 / 144 |
| MA57 (1 thread only) | 117 / 144 | not run |
| MUMPS | 105 / 144 | 105 / 144 |
| MA97 | 103 / 144 | 105 / 144 |
| MKL PARDISO | 88 / 144 | 88 / 144 |

Case-tracks that only smf solves accurately: 7 at 1 thread and 14 at 8 threads (21 in total over both thread counts; at 8 threads MA57 is not run, so more cases have no accurate rival). Case-tracks no solver solves accurately: 5 at each thread count (the 5 above).

Inertia against the reference factorisation, as wrong counts per thread count: smf 0 and MUMPS 0 at both; MA57 4 (1 thread); MA97 5 at 1 thread and 4 at 8; PARDISO 10 at each. Every wrong inertia was on an inaccurate run.

What changed since the November 3 board: PR #36 moves the 8-thread analysis of the matching-heavy cases (ncvxqp5 rec 8T 2.30x to 1.78x against PARDISO, c-big ipopt 8T wall 2807 to 1764 ms) and leaves the 1-thread numbers within 0.3%. Fill and accuracy are unchanged.

What still loses by more than 1.1x (fair board with PR #36):

- **At 1 thread:** mainly matching overhead in the analysis against MA57 on small IPOPT KKTs: cops_pinene_400 (2.35x), darcy003 and mario002 (1.77-1.79x), blockqp1 (1.63x) and stokes128 (1.52x). Outside matching, pde_control_250 rec (1.64x against PARDISO) and ncvxqp9 rec (1.33x against MA97) lose on factor time.
- **At 8 threads:** mainly factor speed against PARDISO: pde_control_250 and pde_control_120 rec (2.05x, 1.92x), G3_circuit (1.30x), bratu3d ipopt (1.25x) and acopf_grid_160 (1.19-1.21x). Matching still dominates ncvxqp5 (1.64-1.78x against PARDISO) and cops_pinene_400 ipopt (1.80x against MA97).

The hybrid matcher write-up is `docs/performance/2026-11-02-auction-matching.md`. The latest results are in `docs/performance/2026-11-02-factor-side-delays.md`, and the full Ryzen scoreboard with every rival is in `docs/performance/2026-10-30-branches-and-scoreboard.md`.

## Roadmap

The long-term plan is in `docs/roadmap.md`: win the CPU KKT niche, then a GPU mixed-precision path, then prove it on public end-to-end benchmarks. The next six steps each have an implementation plan in `docs/superpowers/plans/`:

- **Rescue cost** (`2026-11-02-rescue-cost.md`, in progress). The rescue reuses the matcher the analysis already built instead of rebuilding it, and cheapens or skips the unscaled first try after an initial rejection.
- **Auction matching** (`2026-11-02-auction-matching.md`). An epsilon-scaling auction for the analysis' max-product matching, with the same verdicts and accuracy, aimed at the matching-dominated losses above.
- **IPOPT end to end** (`2026-11-02-ipopt-end-to-end.md`). A reproducible public board of IPOPT with smf against MA27, MA57, MA97 and MUMPS on CUTEst, COPS 3.0 and PGLib-OPF, with the instance list and scoring frozen before the first run.
- **Intra-front parallelism** (`2026-11-02-intra-front-parallel.md`). Tile Cholesky and outer-blocked TPP kernels so that large SPD and TPP fronts run as tile tasks and scale past 8 threads.
- **SPD ordering cost** (`2026-11-02-spd-ordering-cost.md`). Cheaper nested dissection on large SPD matrices through METIS options, compression and a thread-safe METIS build, without losing fill.
- **GPU mixed precision** (`2026-11-02-gpu-mixed-precision.md`). Factor the large root fronts in FP32, recover FP64 accuracy with GMRES refinement, and offload them to an RTX 5080, measured against cuDSS. It starts with a CPU go/no-go gate.

## License

MIT, see `LICENSE`. The vendored METIS 5.2.1 and GKlib in `solver/third_party/`
(built into smf by default) keep their own licence, Apache-2.0, plus a few
BSD-style headers. See `solver/third_party/README.md`.
