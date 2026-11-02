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

## Where it stands

The reference board is the front-tier manifest: 137 to 144 case-tracks of KKT, hard indefinite and SPD matrices, each replayed with the "recommended" and "ipopt" pivoting settings, split into tuning and held-out sets. Scores are smf's time divided by the other solver's, as a shifted geometric mean with a 10 ms shift, so below 1 means smf is faster. A case only counts when both solvers pass the accuracy gate (eta ≤ 1e-10, omega ≤ 1e-8).

Against the fastest other solver on each case (MA57, MA97, SPRAL SSIDS, MKL PARDISO, MUMPS, CHOLMOD), smf scores 0.80 at 1 thread and 0.56 at 16 threads. Those boards predate PRs #17 and #18, which cut wall time a further 3 to 4% overall and more on the delay-heavy KKT classes, so the current numbers should be a little better. MA27 and cuDSS are measured as reference columns only.

Head to head at 1 thread (October 30 board), smf against each solver: MA57 0.60, MA97 0.62, MKL PARDISO 0.58, SSIDS 0.37, MUMPS 0.26. smf also solves more case-tracks accurately than any of them. The one pairing it still loses is MA27 on small KKT systems, where MA27's low overhead wins.

What still loses, mostly at 1 thread:

- Cases dominated by the analysis' matching trial: cops_pinene_400, ncvxqp3, stokes128, darcy003, mario002, case39 and the TSOPF family. Removing matching time entirely would take the 1-thread score from 0.80 to 0.61.
- Small factors that pay for the scale-on-delay rescue, such as ncvxqp9 (32 ms against MA97's 16 ms).
- Ordering time on large SPD matrices (cfd2, bmwcra_1, nd24k), where serial METIS costs more than PARDISO's whole analysis.
- Fronts that run serially inside, which keeps nd24k, c-big and ncvxqp3 flat from 8 to 16 threads.

The latest results are in `docs/performance/2026-11-02-factor-side-delays.md`, and the full scoreboard with every rival is in `docs/performance/2026-10-30-branches-and-scoreboard.md`.

## Roadmap

The long-term plan is in `docs/roadmap.md`: win the CPU KKT niche, then a GPU mixed-precision path, then prove it on public end-to-end benchmarks. The next six steps each have an implementation plan in `docs/superpowers/plans/`:

- **Rescue cost** (`2026-11-02-rescue-cost.md`, in progress). The rescue reuses the matcher the analysis already built instead of rebuilding it, and cheapens or skips the unscaled first try after an initial rejection.
- **Auction matching** (`2026-11-02-auction-matching.md`). An epsilon-scaling auction for the analysis' max-product matching, with the same verdicts and accuracy, aimed at the matching-dominated losses above.
- **IPOPT end to end** (`2026-11-02-ipopt-end-to-end.md`). A reproducible public board of IPOPT with smf against MA27, MA57, MA97 and MUMPS on CUTEst, COPS 3.0 and PGLib-OPF, with the instance list and scoring frozen before the first run.
- **Intra-front parallelism** (`2026-11-02-intra-front-parallel.md`). Tile Cholesky and outer-blocked TPP kernels so that large SPD and TPP fronts run as tile tasks and scale past 8 threads.
- **SPD ordering cost** (`2026-11-02-spd-ordering-cost.md`). Cheaper nested dissection on large SPD matrices through METIS options, compression and a thread-safe METIS build, without losing fill.
- **GPU mixed precision** (`2026-11-02-gpu-mixed-precision.md`). Factor the large root fronts in FP32, recover FP64 accuracy with GMRES refinement, and offload them to an RTX 5080, measured against cuDSS. It starts with a CPU go/no-go gate.

## License

MIT, see `LICENSE`.
