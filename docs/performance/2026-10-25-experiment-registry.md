# SMF Advanced Research Experiment Registry

**Baseline Commit:** `b1f211f` (`feat/solver-performance`)
**Entry Prerequisite:** G0 and G1 passed; G2/G3 CPU baseline measured on target Linux host with installed HSL/SSIDS/MUMPS/PARDISO comparators.

## Admitted Experiment Specifications

| Experiment | Time Box | Baseline Commit | Hypothesis & File Scope | Frozen Subset | Accuracy Contract | Stop Condition | Status |
|---|---|---|---|---|---|---|---|
| **1. APTP Tiled Fronts** | 2 weeks | `b1f211f` | A posteriori threshold pivoting with rollback in `solver/src/blocked_indef.cpp` reduces panel synchronization when pivoting is a measured limiter. | `benchmarks/manifests/kkt.json` (`train` split for tuning, `test` split held out) | `eta_inf <= 1e-10`, `omega <= 1e-8`, exact inertia match, zero false successes | Abort and disable if full-sequence gain `< 20%` on target cohort or if G1 regresses. | Gated on G2 CPU profile |
| **2. FP32 Factors + FP64 Refinement** | 1 week | `b1f211f` | Mixed-precision `LDL^T` factors with FP64 residual/refinement in `solver/src/refinement.cpp` reduce bandwidth on memory-bound fronts. | `benchmarks/manifests/kkt.json` | Same external FP64 `eta_inf <= 1e-10`, `omega <= 1e-8` and inertia contract; fallback to FP64 included in wall time | Abort if total gain (including retries/fallback) `< 20%` or any new accuracy/inertia failure occurs. | Gated on G2 CPU profile |
| **3. GPU Large-Front Updates** | 2 weeks | `b1f211f` | Offloading large-front Schur updates when calibrated transfer+compute model predicts net gain. Native GPU kernels reported separately from whole-solver backends (`cuDSS`, `SSIDS`). | Large-front subset (`max_front >= 256`) | Same G1 contract; host-device transfers and setup included in wall time | Abort if full-sequence gain `< 1.5x` vs strongest CPU baseline or device memory budget exceeded. | Gated on G3 CPU profile |
| **4. Explicit Structured KKT Condensation** | 2 weeks | `b1f211f` | Condensing caller-supplied block structure before sparse factorization reduces fill on structured OCP/MPC KKT systems. | Structured KKT subset | Original uncondensed KKT residuals (`eta_inf <= 1e-10`, `omega <= 1e-8`), inertia, and IPOPT convergence preserved | Abort if total gain `< 20%` or application convergence degrades. | Gated on application profile |
| **5. BLR / Low-Rank Front Compression** | 2 weeks | `b1f211f` | Block low-rank compression of large off-diagonal frontal tiles reduces flops and factor memory on 3D PDE/KKT separators. | Large-separator subset | Original-system `eta_inf <= 1e-10`, `omega <= 1e-8` and inertia preserved | Abort if time/memory gain does not survive original-system accuracy and inertia checks. | Gated on separator rank profile |

## Scope Boundaries (Not Claimed in First Release)

The following capabilities require separate project plans and dedicated competitor/resource contracts and are **not** claimed as supported by this release:
- Out-of-core / external-memory factorization (`HSL MA77` track)
- Distributed-memory MPI factorization
- Complex symmetric / Hermitian arithmetic
- Unsymmetric / rectangular linear systems
- External 64-bit CSC index ABI (`Int` remains `int32_t`; internal factor/storage offsets are 64-bit `LongInt`)
