# smf long-term roadmap

**Date:** 2026-10-26
**Input:** `solver_research_report.md` (state of the art in optimization solvers, Oct 2026) and the performance write-ups in `docs/performance/`.

## Positioning

The research report ranks complete solvers like COPT, Gurobi and cuOpt on LP, MIP and NLP. smf is not one of those. It is the sparse symmetric LDLᵀ engine that runs inside an interior-point (barrier) solver. That is a strong place to be: every second-order entry in the report spends most of its time in this kernel. That includes COPT's barrier, cuOpt's barrier and IPOPT/Knitro on the AMPL-NLP set. The report also names cuDSS (NVIDIA's GPU sparse direct solver) as the thing that made GPU barrier methods practical.

**Goal:** be the factorization a barrier solver would choose, and prove it on a public end-to-end benchmark. We are not chasing "#1 overall" or MIP; the report rates MILP optimality as the least contestable target.

## Where we stand (2026-10-26)

- smf beats MA27 by 1.37× geomean on optim20 (`docs/performance/2026-10-25-beating-ma27.md`).
- smf beats CHOLMOD, PARDISO and MUMPS on all 7 IPOPT KKT models at 1 thread, and on 6 of 7 at 8 and 16 threads. It beats MA27 on all 7 IPOPT models (`docs/performance/2026-10-25-large-scale-benchmark.md`).
- Open gaps:
  - backsolve is about 2× slower than MA27's;
  - cvxqp3 and ncvxqp3 time out;
  - d_pretok and turon_m fail accuracy (both are singular);
  - the threading results have not been re-measured after the engine changes;
  - IPOPT end to end has not been re-run with the new plugin;
  - there is no comparison yet against MA57, MA97, SPRAL SSIDS or cuDSS.

## Horizon 1 — Win the CPU KKT niche (~3 months)

1. **Benchmark against the real competition.**
   - MA57 and MA97: apply for the HSL academic licence (free for academic use). The runtime-loaded adapters already exist (`SMF_HSL_LIBRARY`).
   - SPRAL SSIDS: needs Linux with gfortran, so WSL or a Linux host.
   - Until these are in the table, "best open KKT solver" cannot be claimed.
2. **Add an interior-point mode on top of the general solver.** These are features general-purpose codes handle poorly:
   - **Reuse the pivot sequence** from the previous iterate, and fall back to fresh pivoting only if it fails. IPOPT never changes the sparsity pattern, and fresh pivoting is expensive: the first pde_control_250 factorization delayed 26,176 pivots.
   - **Cheaper inertia-correction refactors.** When IPOPT adds δI and refactors, the pattern and ordering are the same, so everything symbolic can be reused and only the numeric phase runs.
   - **Make the regularized ("quasidefinite") path a supported mode** instead of an opt-in (`Control::static_pivot_tol`). A quasidefinite matrix can be factored stably in any symmetric order (Vanderbei), so ordering and memory are fixed at analysis. That is what makes parallel and GPU factorization tractable in Horizon 2.
   - **Status (2026-10-26):** all three are built, behind `smf::IpmSession` (`docs/performance/2026-10-26-ipm-mode.md`). All stay **off by default**, because each failed its gate:
     - **Early abort on wrong inertia:** identical IPOPT logs, but it saves < 5%, since the excess negatives appear only in the root front.
     - **Realized-order re-analysis:** cuts delays 16–18× when adopted, but rejected attempts cost wall time; it needs a retry backoff.
     - **Fallback and Regularized static modes:** new failures, +23% to 2.75× iterations, and up to 4.7× wall time on the held-out models.

     Next levers: re-analysis backoff, ρ applied after scaling, and the case9241 replay-accuracy triage.
3. **Close the backsolve gap.** IPOPT does one solve plus up to three refinement solves per factorization, so a 2× slower solve shows up directly in end-to-end time (robot_arm: 0.44 s vs MA27's 0.20 s). Options: blocked multi-RHS solves and a solve-friendly factor layout.
4. **Keep a held-out test set.** The report's warning about over-tuning (Mittelmann uses hidden instances and random reordering for this) applies to us. The auto-policy thresholds were set on the same matrices we report on: 5% small diagonals, the 1.5× flops cutoff, the 4.5e5-per-row METIS rule. Freeze a hidden set, for example CUTEst KKT captures or PGLib-OPF, and use it only for validation, never for tuning.

**Exit criteria:** smf matches or beats MA57, MA97 and SSIDS on the IPOPT KKT replays and on IPOPT end to end, on the held-out set as well as the tuning set, with no accuracy-gate failures except known singular cases.

## Horizon 2 — GPU with mixed precision (~3–9 months)

- **The competitor is cuDSS.** It is what cuOpt's GPU barrier and current GPU NLP work (report S6, S7) are built on.
- **The development GPU is an RTX 5080: a consumer card whose FP64 rate is a small fraction of its FP32 rate** (about 1/64 on GeForce cards). A straight FP64 port will lose.
- **The plan:** factor in FP32 and recover FP64 accuracy with GMRES-based iterative refinement (Carson–Higham). The static/quasidefinite mode from Horizon 1 is what makes this safe: no dynamic pivoting on the GPU, and refinement repairs the accuracy.
- **Caveat:** late-iteration KKT systems are badly conditioned. Measure where FP32 factors stop converging under refinement and keep an FP64 fallback.
- **Split the work:** small fronts on the CPU, big fronts near the root on the GPU, as CHOLMOD's and SSIDS's GPU modes do. The tree-parallel engine already separates independent subtrees from the top of the tree.
- **Use the existing replays:** the `bench_replay` KKT recordings give a direct head-to-head against cuDSS on the same matrices.
- **The niche:** cuDSS is aimed at datacenter GPUs. A solver that reaches 1e-10 backward error on gaming GPUs would stand out.

**Exit criteria:** on the large IPOPT KKT replays, the GPU path beats smf's best CPU time and is competitive with cuDSS on the same card, at the same accuracy gate.

## Horizon 3 — Prove it publicly (~6–18 months)

- **(a) First: IPOPT+smf on public NLP sets.**
  - Candidates: CUTEst, COPS and PGLib-OPF. AC-OPF is where GPU NLP is currently being contested, and the corpus already has acopf models.
  - On Mittelmann's AMPL-NLP benchmark IPOPT scores 15.3 against COPT's 1.32 (report §1). Part of that gap is linear algebra.
  - Package smf for conda-forge and pip on Linux and Windows. The MIT licence makes adoption easy.
  - Offer it as a linear solver to IPOPT, CasADi and Clarabel users.
- **(b) Later, possibly our own barrier solver:** a convex QP/LP/SOCP interior-point method built on smf. The report shows the LP winners run barrier, PDLP and simplex at the same time, and barrier is the piece we would be best at. Bigger scope, crowded field: decide after (a).
- **Avoid:**
  - MIP, simplex and a PDLP-only race. PDLP's bottleneck is sparse matrix-vector products, so factorization work does not carry over.
  - Chasing "#1 overall": commercial leaders don't publish public results, so it cannot be shown either way.

## Longer bets

- **Block low-rank fronts,** like MUMPS's BLR mode, for large PDE-constrained problems such as pde_control_250.
- **Exploiting the stage structure of optimal-control KKT systems,** with Riccati-style elimination as in HPIPM. No general sparse solver does this, and it is directly useful for MPC.

## Infrastructure

- **A dedicated Linux benchmark host.** Long sweeps on the development PC are disruptive and add timing noise.
- **CI on Linux and Windows.**
- **Every benchmark report states** the shifted geometric mean, solved counts and the accuracy tolerance, as the research report recommends.

## Next step

Apply for the HSL academic licence, and start the interior-point mode: pivot-sequence reuse, cheap δ refactors, and static/quasidefinite mode as the default on the IPOPT track.
