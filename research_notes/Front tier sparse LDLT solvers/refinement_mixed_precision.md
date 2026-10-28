# Iterative refinement and mixed precision for sparse LDL^T (IPM / IPOPT context), state of the art to Oct 2026

Scope: which refinement and precision techniques are proven to restore accuracy after perturbed (static-pivoted) or low-precision factorizations; how many steps they take; what they cost; implementation guidance for smf (C++20 multifrontal LDL^T inside IPOPT; gate: normwise backward error <= 1e-10, componentwise <= 1e-8; dev GPU RTX 5080 with FP64 at about 1/64 of FP32; late-IPM diagonals down to about 1e-21).

Unit roundoffs used in my arithmetic below: bf16 3.9e-3, fp16 4.9e-4, fp32 6.0e-8 (2^-24), fp64 1.1e-16 (2^-53).

Method note: several PDFs (GMRES-IR5, the MUMPS sparse-IR TOMS paper, Arioli–Duff–Gratton–Pralet, Zounon et al.) could not be parsed by the fetch tool, so I extracted their text with a small zlib script. Quotes and numbers from those papers are therefore primary-source, but table cells that were figures or garbled glyphs were not recoverable and are listed as gaps.

---

## 1. Classical IR vs GMRES-IR vs FGMRES-IR vs MINRES-IR: convergence conditions in terms of kappa and precisions

### Takeaway
Classical (LU-based) IR with factorization precision u_f converges only if kappa(A)·u_f << 1 (inflated by the growth factor and any factorization approximation). GMRES-IR, which uses the factors as a preconditioner, moves the limit to roughly kappa^3·u_p·u_f^2 << 1 (forward error). With an fp32 factorization and fp64 GMRES/preconditioner that is about 1e10 instead of about 2e7. FGMRES is the documented fix when classical IR stalls after static pivoting. I found no peer-reviewed mixed-precision MINRES-IR analysis for symmetric indefinite systems.

### Cited Findings
- **LU-IR (classical/three-precision IR) condition.** For both forward and backward error the condition is kappa(A)·u_f << 1. The limiting backward error is about u. The limiting forward error is about u_r·cond(A,x) + u. — [Amestoy et al., TOMS 2023, eq. 4.1](https://www.lip6.fr/Theo.Mary/doc/SparseIR.pdf); [Carson & Khan summary, arXiv 2202.10204](https://ar5iv.labs.arxiv.org/html/2202.10204)
- **Numeric LU-IR forward limits (u = fp64).** kappa << 3e2 for bf16, 2e3 for fp16 and 2e7 for fp32 factorization. — [Amestoy, Buttari, Higham, L'Excellent, Mary, Vieublé, "Five-precision GMRES-based iterative refinement" (SIMAX 2024), Table 3.2 caption](https://www.lip6.fr/Theo.Mary/doc/GMRESIR5.pdf)
- **GMRES-IR3 (Carson & Higham), with GMRES in u and preconditioned matvec in u^2.**
  - Forward-error condition: kappa << u^(-1/2)·u_f^(-1).
  - Backward-error condition: kappa << u^(-1/2)·u_f^(-1/2).
  - The commonly implemented variant with u_g = u_p = u has a forward condition of kappa <= u^(-1/3)·u_f^(-2/3); the source calls these bounds "overly strict in practice".
  - — [Carson & Khan, arXiv 2202.10204](https://ar5iv.labs.arxiv.org/html/2202.10204)
- **GMRES-IR5 general conditions (approximate factorization included).**
  - Forward error converges if (u_g + u_p·rho·kappa)·((u_f·rho + eps)·kappa)^2 << 1.
  - Backward error converges if (u_g + u_p·rho·kappa)·(1 + (u_f·rho + eps)·kappa)·kappa << 1.
  - LU-IR needs (u_f·rho + eps)·kappa << 1.
  - Here rho is the growth factor and eps the factorization approximation (BLR threshold plus static-pivot threshold).
  - — [Amestoy et al., TOMS 2023, Theorem 4.1, eqs. 4.20–4.23](https://www.lip6.fr/Theo.Mary/doc/SparseIR.pdf)
- **GMRES-IR5 worked examples with u = fp64.**
  - fp16 factorization:
    - u_p = fp128 gives kappa << 2e11 (the GMRES-IR3 result).
    - u_p = fp64 gives kappa << 3e7. The paper notes this variant "has been successfully used in practice".
    - u_p = fp32 gives kappa << 4e4.
    - u_g = fp32 with u_p = fp64 gives kappa << 8e6.
  - fp32 factorization with u = u_p = fp64 and u_g = fp16 gives kappa << 7e8.
  - — [GMRES-IR5 paper, Sec. 3.4](https://www.lip6.fr/Theo.Mary/doc/GMRESIR5.pdf)
- **Meaningful combinations.** These satisfy u^2 <= u_r <= u <= u_f. u_p can usefully be below, equal to or above u, and u_g can usefully be below, equal to or above u_f. Of more than 3000 combinations, 112 (3.5%) are meaningful. — [GMRES-IR5 paper](https://www.lip6.fr/Theo.Mary/doc/GMRESIR5.pdf)
- **Experiments vs. bounds.** On random dense matrices with u_g = d, the success rate was 100% up to kappa about 1e7 for u_p = s and about 1e15 for u_p = d or q. The u_p = s and u_p = d variants "perform much better than expected" because kappa(Ã) is in practice about kappa·u_f rather than the worst-case (kappa·u_f)^2. — [GMRES-IR5 paper, Sec. 4](https://www.lip6.fr/Theo.Mary/doc/GMRESIR5.pdf)
- **Cost of GMRES-IR5 vs LU-IR.** GMRES-IR needs an extra LU solve at the start of each outer IR step. On the SuiteSparse set with bf16 factors, the "bdd" variant never needed more LU solves than "bdq". — [GMRES-IR5 paper](https://www.lip6.fr/Theo.Mary/doc/GMRESIR5.pdf)
- **FGMRES after static pivoting.**
  - IR (Richardson) after a statically pivoted LDL^T "fails to converge" on tough problems.
  - GMRES converges when IR does not, and restarted GMRES performs better, but GMRES "even with restarting… is not robust".
  - "FGMRES succeeds when these other methods fail" and is "much less sensitive" to the static-pivot value.
  - Tests used perturbation tau·max|a_ij| with tau in {1e-6, 1e-8, 1e-10} on the CONT-201 and CONT-300 KKT-type problems.
  - — [Arioli, Duff, Gratton, Pralet, RAL-TR-2006-007 / SISC 29(5) 2007](https://numerical.rl.ac.uk/media/reports/adgpRAL2006007.pdf)
- **Variant summary.** The scheme used in [12] runs GMRES in u and uses u^2 preconditioned products. The effective solver precision for GMRES-IR is u_s ≈ u_g + u_p·kappa(A). — [arXiv 2202.10204](https://ar5iv.labs.arxiv.org/html/2202.10204)
- **Weighted least squares with IPM-like weights (FGMRES-IR on the augmented system [D^-1 A; A^T 0]).**
  - Low-precision preconditioners are reliable only when kappa(D) is at most about 1e6 for the left block preconditioner.
  - For the split block-diagonal preconditioner, kappa(M_b^(1/2)) is small enough only when kappa(D) < 1e9.
  - The paper explicitly flags interior-point methods as producing far worse kappa(D), and concludes that "using lower precision can be tricky" because the bounds depend on kappa(D).
  - — [arXiv 2401.03755, "Mixed precision FGMRES-based iterative refinement for weighted least squares"](https://arxiv.org/pdf/2401.03755)

### Inferences
- **My arithmetic for fp32 factorization with u_g = u_p = fp64 (Theorem 4.1 with rho = 1, eps = 0).**
  - The forward condition u·kappa^3·u_f^2 << 1 gives kappa << about 1.4e10.
  - The backward condition u·kappa^2·(1 + kappa·u_f) << 1 gives kappa << about 5e7.
  - So the guaranteed backward-error regime for "fp32 LDL^T + fp64 GMRES" is only about 2–3 orders of magnitude beyond fp32 LU-IR (about 2e7). The forward regime is about 3 orders beyond that. The paper's experiments say practice is substantially better than the bounds.
- **Effect of a looser accuracy gate.** smf's gate (normwise BE 1e-10) is about 6 orders looser than u_fp64. Convergence rate matters more than the limiting accuracy: LU-IR contracts by roughly phi ≈ c·kappa_eff·(u_f·rho + eps) per step. With fp32 factors and kappa_eff (after scaling) about 1e5, phi is about 6e-3, so going from about 1e-7 to below 1e-10 takes about 2 steps. With kappa_eff about 1e7, phi is about 0.6 and the step count explodes or diverges.
- **MINRES-IR.** I found no 2017–2026 mixed-precision MINRES-IR paper for symmetric indefinite LDL^T. The natural construction is MINRES preconditioned by the SPD matrix L|D|L^T, with |D| taken eigenvalue-wise on the 2x2 blocks. That preconditioner is classical in optimization (Gill–Murray–Ponceleón–Saunders 1992; not fetched this session). The preconditioned spectrum clusters near ±1, so MINRES would need short recurrences and no Krylov basis storage. There is no rounding-error analysis comparable to GMRES-IR5 for it, so treat it as engineering, not as proven.
- **FGMRES vs GMRES.** For smf, FGMRES (right preconditioning, flexible) is the variant with published evidence for static-pivoted LDL^T (Arioli et al.). GMRES-IR with left preconditioning is the variant with the mixed-precision rounding analysis (Carson–Higham, Amestoy et al.).

### Gaps
- Table 3.2/3.3 cell values for u_f = fp32 (e.g., the exact "sdd" bound) were not recoverable from the extracted PDF text. The about 1.4e10 and about 5e7 above are my arithmetic from the stated formulas.
- Carson & Higham SISC 2017/2018 were not fetched directly this session. Their conditions are quoted via the two secondary summaries above, which agree with each other and with Amestoy et al.
- No source found giving a rounding-error analysis of MINRES-IR or of split-preconditioned MINRES with an indefinite LDL^T preconditioner.

---

## 2. Mixed-precision factorization in sparse direct solvers (MUMPS, BLR, cuDSS, PARDISO, STRUMPACK): measured speedups and failure cases

### Takeaway
An fp32 sparse factorization plus fp64 refinement typically halves factor memory. It speeds up the factorization by 1.5–2x only on large problems where numerical factorization dominates; medium-size problems are dominated by analysis and ordering. It can be slower than plain fp64 when many refinement solves are needed or kappa is large. The best published MUMPS results combine fp32 with BLR and static pivoting: up to 5.6x time and 4.4x memory reduction. Ill-conditioned matrices are the documented failure mode.

### Cited Findings
- **MUMPS mixed precision (Amestoy et al., ACM TOMS 2023).**
  - The study uses MUMPS (multifrontal) with LU-IR and GMRES-IR, fp32 factorization, fp64 working precision, and fp64 or fp128 residual.
  - The GMRES tolerance was fixed at 1e-6. GMRES ran in fixed precision u_g = u_p.
  - Half precision was not tested because "MUMPS … does not currently support" it.
  - — [Amestoy et al. TOMS 2023](https://www.lip6.fr/Theo.Mary/doc/SparseIR.pdf)
- **Results without approximation (full-rank, partial pivoting).**
  - "LU-IR variants with single precision factorization generally achieve the lowest execution times."
  - fp32 LU-IR "consumes half as much memory as DMUMPS" in the common case.
  - GMRES-IR (fp32 factors cast to fp64 for u_p = d) "converge for all matrices in our set", unlike fp32 LU-IR. Its memory peak is lower than the fp64-factorization peak by "almost up to 50%".
  - — [Amestoy et al. TOMS 2023, Sec. 5.3](https://www.lip6.fr/Theo.Mary/doc/SparseIR.pdf)
- **Failure and slowdown cases.**
  - Matrix 6, a small 2D problem: the solve and matvec cost is high relative to factorization, so DMUMPS is fastest.
  - Matrix 17: the fp32 factorization is only 1.6x faster and fill is small, so the fp128 residual is relatively expensive.
  - Matrix 24, "badly conditioned": fp32-factor variants "either do not converge or require so many iterations" that they are slower than DMUMPS, but with reduced memory.
  - With fp32 factorization, matrix 24 still used about 55% of the fp64 memory because symbolic structures are precision-independent.
  - — [Amestoy et al. TOMS 2023, Secs. 5.3.1–5.3.2](https://www.lip6.fr/Theo.Mary/doc/SparseIR.pdf)
- **fp128 residual costs little for sparse matrices.** Because factors are much denser than A, computing the residual in fp128 "does not necessarily have a significant impact on the performance". The variants with u_r = q reach a forward error of about 1e-16 "with only a small additional overhead". — [Amestoy et al. TOMS 2023, Secs. 3 and 5.3](https://www.lip6.fr/Theo.Mary/doc/SparseIR.pdf)
- **Best results with approximate factorizations (BLR and/or static pivoting).**
  - The best time reduction is 5.6x (industrial matrix 22). The best memory reduction is 4.4x (matrix 25).
  - A balanced setting on matrix 25 gave a 3.6x memory reduction at a modest slowdown.
  - Static pivoting inside BLR made the factorization 1.26x faster (88 s to 70 s) because panel reduction can use low-rank operations.
  - A larger BLR threshold reduces factorization cost but raises iteration count. With many iterations, the Krylov basis memory can exceed the BLR savings.
  - — [Amestoy et al. TOMS 2023, Secs. 5.4–5.5](https://www.lip6.fr/Theo.Mary/doc/SparseIR.pdf)
- **MUMPS now exposes this mode directly.** PETSc's MUMPS interface lists ICNTL(47), "single precision factorization in a double precision instance", alongside ICNTL(10) (max refinements), CNTL(2) (refinement stopping criterion), CNTL(4) (static pivoting value), ICNTL(35)/CNTL(7) (BLR). — [PETSc MATSOLVERMUMPS manual page](https://petsc.org/main/manualpages/Mat/MATSOLVERMUMPS/)
- **Single vs double precision across solvers (Zounon, Higham, Lucas, Tisseur).**
  - The "anticipated speedup of 2 over a double precision LU factorization is obtained only for the very largest" test problems.
  - Serial PARDISO: half of 36 matrices exceed 1.5x. On 10 cores this drops to 30%, all of them large matrices.
  - MUMPS on 10 cores: only five matrices exceed 1.5x.
  - SuperLU_MT: 33% (serial) and 24% (parallel) of 21 matrices exceed 1.5x.
  - Causes: (1) subnormals. Without flush-to-zero, 15 of 36 matrices were slower in fp32, and more than half generated subnormals through "cascading fill-ins". (2) Reordering and analysis are precision-independent and poorly parallel; many medium matrices spend more than 25% of time there.
  - — [Zounon et al., "Performance impact of precision reduction in sparse linear systems solvers" (MIMS eprint 2811 / PeerJ CS 2022)](https://eprints.maths.manchester.ac.uk/2811/1/paper.pdf)
- **PARDISO (oneMKL).**
  - iparm(28) = 1 selects single precision: "Input arrays (a, x and b) must be presented in single precision". This is a pure-fp32 solver mode, not fp32-factor/fp64-refine.
  - iparm(8) = 0 (default): "automatically performs two steps of iterative refinement when perturbed pivots are obtained"; a positive value sets the maximum.
  - iparm(10): pivots are perturbed with 10^-iparm(10). The default is 8 (1e-8) for symmetric indefinite and 13 for unsymmetric.
  - iparm(14) reports the number of perturbed pivots. iparm(21) = 1 (default) uses 1x1 and 2x2 Bunch–Kaufman pivoting.
  - — [oneMKL PARDISO parameter reference (archived manual)](https://www.irya.unam.mx/computo/sites/manuales/fce12/mkl/mkl_manual/ssr/ssr_pardiso_parameters.htm)
  - A user reported wrong single-precision results until setting iparm(1) = 1 (anecdotal). — [Intel community forum](https://community.intel.com/t5/Intel-oneAPI-Math-Kernel-Library/Single-precision-PARDISO-gives-wrong-results/m-p/1361530)
- **cuDSS (current docs).**
  - Refinement: CUDSS_CONFIG_IR_N_STEPS (default 0, "Maximum number of steps during the iterative refinement") and CUDSS_CONFIG_IR_TOL (default 0.0, early exit when the residual norm is below tolerance).
  - Pivoting: CUDSS_CONFIG_PIVOT_EPSILON, "the absolute value to test and replace small diagonal elements", default 1e-13 (double) and 1e-5 (single). Pivot type includes Bunch–Kaufman for symmetric indefinite, with default alpha about 0.64. Matching is available.
  - CUDSS_CONFIG_DOWNCAST_FACTORS stores fp64 factors in fp32 after factorization (R_64F to R_32F).
  - CUDSS_CONFIG_EMULATION_MANTISSA_BIT_COUNT enables floating-point emulation with a chosen mantissa width, for factorization only.
  - An experimental double-double type CUDSS_R_64F_MP2 exists.
  - — [NVIDIA cuDSS types/config docs](https://docs.nvidia.com/cuda/cudss/latest/types.html)
- **Conflict on cuDSS pivoting.** The 2024 MadNLP condensed-IPM paper states cuDSS "2x2 pivoting strategy is not supported". The current cuDSS docs list Bunch–Kaufman (1x1/2x2) pivoting. This is likely a version difference. — [Pacaud, Shin et al., arXiv 2405.14236](https://arxiv.org/html/2405.14236v3) vs [cuDSS docs](https://docs.nvidia.com/cuda/cudss/latest/types.html)
- **Dense precedent.** LAPACK DSGESV/DSPOSV factor in fp32, refine to fp64 backward error, and fall back to fp64 factorization if refinement fails. — [gesv_mixed man page](https://www.mankier.com/3/gesv_mixed); [posv_mixed man page](https://www.mankier.com/3/posv_mixed)

### Inferences
- **For smf on CPU.**
  - Expect at best about 1.5–2x on the numeric factorization and about 2x on factor memory from fp32 fronts.
  - smf's memory notes show analysis-bound gaps on several cases, so the end-to-end gain per IPOPT iteration will be smaller. This matches Zounon et al.: analysis and ordering are precision-independent.
  - Flush-to-zero/denormals-are-zero in the fp32 kernels is mandatory. Late-IPM KKT entries near 1e-21 and their products will hit fp32 subnormals (min normal about 1.2e-38) after scaling and Schur updates.
- **For a GPU path on RTX 5080 (FP64 about 1/64 of FP32).**
  - An fp32 dense-front factorization is the only throughput-viable option.
  - The SpMV residual in fp64 is memory-bound, so the 1/64 FP64 rate hurts it far less than it would hurt dense fronts.
  - GMRES-IR with u_p = fp64 requires fp64 triangular solves with the fronts, which is also memory-bound. Storing factors in fp32 and upcasting on the fly would avoid the 2x memory of casting (the TOMS paper casts and pays the memory).
  - cuDSS's EMULATION_MANTISSA option (FP64-class accuracy emulated on lower-precision units) is the alternative for consumer GPUs. I found no published accuracy or speed data for it.
- **MKL PARDISO fp32 mode is not a usable baseline for mixed-precision IR.** It wants fp32 arrays end to end. The refinement would have to be wrapped externally.

### Gaps
- MUMPS 5.7/5.8 release notes: could not retrieve the official changelog (404 on the Debian copy). ICNTL(47) is confirmed only via the PETSc docs, not the MUMPS user guide; its exact semantics (where refinement runs, fallbacks) are unverified.
- STRUMPACK mixed precision: no source retrieved this session.
- cuDSS: no published speedup or accuracy figures for DOWNCAST_FACTORS or EMULATION modes, and no statement of whether IR in cuDSS computes the residual in higher precision.
- Table 2 per-matrix numbers (seconds/GB) of the TOMS paper were not recoverable as text.

---

## 3. Refinement after static pivoting: componentwise backward error and how many perturbed pivots IR or GMRES-IR can tolerate

### Takeaway
Static pivoting replaces tiny pivots with tau·||A|| (SuperLU_DIST: sqrt(eps)·||A||; PARDISO symmetric indefinite: 1e-8; cuDSS: 1e-13 in fp64). It acts as a factorization perturbation eps ≈ tau that enters the convergence condition exactly like u_f: LU-IR needs (u_f·rho + tau)·kappa << 1. The theory bounds the size of the perturbation, not the count of perturbed pivots. Classical IR is documented to fail on tough KKT-like problems; FGMRES restores robustness. There is an optimal, problem-dependent tau that balances perturbation size against growth.

### Cited Findings
- **Static pivoting in MUMPS (Li & Demmel origin).**
  - "when a pivot is found to be too small with respect to a prescribed threshold tau_s·||A||, it is replaced with tau_s·||A||".
  - "A small value … makes the factorization more accurate but might lead to large element growth, while a large value controls element growth but reduces the accuracy".
  - Static pivoting adds no extra fill and improves BLAS3 use and parallelism.
  - — [Amestoy et al. TOMS 2023, Sec. 2](https://www.lip6.fr/Theo.Mary/doc/SparseIR.pdf)
- **Convergence condition with static pivoting.** Theorem 4.1 applies with eps = eps_BLR + tau_s. The conditions are those of Section 1 above, with the growth factor rho. "There is an optimal value of tau_s, which is clearly problem dependent, that leads to the fastest convergence by balancing the u_f·rho and tau_s terms". The factorization time does not depend on tau_s. — [Amestoy et al. TOMS 2023, Sec. 4.2.4 and 5.4.2](https://www.lip6.fr/Theo.Mary/doc/SparseIR.pdf)
- **SuperLU_DIST.** It performs LU with static pivoting. An interface option replaces tiny pivots by sqrt(epsilon)·norm(A) (default on). It offers working- and extra-precision refinement and backward-error and error-bound estimation. — [GEOSX SuperLU_DIST parameter docs](https://geosx-geosx.readthedocs-hosted.com/en/develop/doxygen_output/html/structgeos_1_1_linear_solver_parameters_1_1_direct.html); [SuperLU_DIST docs](https://portal.nersc.gov/project/sparse/superlu/superlu_dist_code_html/index.html); [OSTI record of Li & Demmel TOMS 2003](https://www.osti.gov/biblio/836786)
- **Riedy's thesis (Demmel's group).** Restrictive and static pivoting solutions "still have small error so long as … the growth factor does not grow too large". Refinement can flag failures when growth is large. — [Riedy, UCB EECS-2010-172](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2010/EECS-2010-172.pdf)
- **Arioli–Duff–Gratton–Pralet (LDL^T, MA57/MUMPS-style two-phase pivoting).**
  - The first phase pivots numerically in the fully summed block.
  - The remaining fully summed variables are eliminated with 1x1 perturbations: a_ii is replaced by s(a_ii)·tau·max_ij|a_ij|.
  - The analysis requires the perturbation to exceed the rounding term c·eps·|| |L||D||L^T| || "in order not to dominate the global error".
  - With tau in {1e-6, 1e-8, 1e-10}, classical IR "is unable to obtain a scaled residual close to rounding". FGMRES succeeds.
  - — [RAL-TR-2006-007](https://numerical.rl.ac.uk/media/reports/adgpRAL2006007.pdf)
- **Componentwise backward error and stopping in production solvers.** MUMPS exposes CNTL(2) (refinement stopping criterion), ICNTL(10) (max refinements) and ICNTL(11) (error-analysis statistics). — [PETSc MUMPS page](https://petsc.org/main/manualpages/Mat/MATSOLVERMUMPS/)
- **PARDISO.** Two automatic refinement steps when perturbed pivots occur (default). The count of perturbed pivots is reported in iparm(14). — [oneMKL PARDISO parameters](https://www.irya.unam.mx/computo/sites/manuales/fce12/mkl/mkl_manual/ssr/ssr_pardiso_parameters.htm)
- **Extra-precise IR error bounds.** A single application of extra-precise IR can yield an error bound in any norm at small cost. The bound stays reliable for condition numbers below about 1/(max(10, sqrt(n))·eps_w). This was tested on more than 6.2 million matrices. — [Demmel, Hida, Kahan, Li, Mukherjee, Riedy, UCB/CSD-04-1344](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2004/5919.html)

### Inferences
- **"How many perturbed pivots can IR tolerate?"** has no count-based answer in the literature. What matters is ||Delta A||/||A|| ≈ tau times the effective kappa of the scaled system, as seen through the condition (u_f·rho + tau)·kappa << 1.
  - k perturbations of size tau·||A|| form a rank-k perturbation. Preconditioned GMRES or FGMRES then has about k outlying eigenvalues and converges in about k + O(1) iterations in exact arithmetic. This is the standard rank-k argument, not stated with numbers in the fetched sources.
  - So GMRES/FGMRES-IR tolerates "a few" to "tens" of perturbed pivots at modest cost. Classical IR may stall even for k = 1 if tau·kappa is not small.
  - This is consistent with Arioli et al., where FGMRES succeeds and IR fails.
- **For smf (fp64 factor, static pivot after the 2x2 attempt).**
  - Use tau ≈ 1e-8 relative to the scaled max entry (the PARDISO default for symmetric indefinite). sqrt(eps_64) ≈ 1.5e-8 matches the SuperLU_DIST rule.
  - Run classical IR first, with componentwise BE (Arioli–Demmel–Duff omega_1/omega_2) as the stopping test.
  - If 2 steps do not drop below the 1e-8 componentwise / 1e-10 normwise gate, switch to FGMRES(m ≈ 10–20) preconditioned by the same factors, with fp64 everywhere.
  - Use the perturbed-pivot count only as a trigger, e.g. 0 means IR is likely to converge in one step and k > 0 means expect FGMRES. Do not use it as a hard limit.
- **Static pivots and inertia.** A perturbed 1x1 pivot in an IPM KKT system changes inertia counting. IPOPT's inertia correction assumes exact inertia. A static-pivot path should either count perturbed pivots as "wrong inertia" or report them so IPOPT can raise delta_w/delta_c. This is an inference, not sourced.

### Gaps
- The Li & Demmel 2003 TOMS paper and the Arioli–Demmel–Duff 1989 SIMAX paper were not fetched. The componentwise backward error definitions (omega_1, omega_2 with thresholded "sparse" denominators) and SuperLU's default refinement count are from background knowledge, not verified this session.
- No source gives an empirical distribution of "number of perturbed pivots vs. IR steps" for IPM KKT matrices.

---

## 4. Refinement inside IPMs: IPOPT's loop, late-iteration ill-conditioning, and whether FP32 factorizations work for IPM KKT systems

### Takeaway
IPOPT runs 1–10 Richardson IR steps on the full KKT system until a residual ratio of 1e-10. If refinement fails, it tightens the pivot tolerance (to at most 1e-4) and refactors. FP32 factorization inside IPMs has been shipped only as an opt-in that is known to degrade or fail on ill-conditioned problems. The clearest data (CuClarabel) shows about 1.8–2x on favorable problems and slowdowns or extra IPM iterations on others, with heavier regularization (sqrt(eps)) required. No published 2022–2026 result shows FP32 LDL^T handling late-iteration NLP KKT systems (diagonals of about 1e-21) to a 1e-10 backward-error gate. Theory and WLS experiments both predict failure without a Krylov outer solver and/or condensation.

### Cited Findings
- **IPOPT refinement options and defaults.**

  | Option | Default | Meaning |
  |---|---|---|
  | min_refinement_steps | 1 | minimum number of refinement steps |
  | max_refinement_steps | 10 | maximum number of refinement steps |
  | residual_ratio_max | 1e-10 | "Iterative refinement is performed until the residual test ratio is less than this tolerance" |
  | residual_ratio_singular | 1e-5 | if exceeded after failed IR, the system is treated as singular |
  | residual_improvement_factor | 1 | stop if a step does not improve the ratio by this factor |
  | ma27_pivtol / ma57_pivtol | 1e-8 | pivot tolerance |
  | ma27_pivtolmax | 1e-4 | "Ipopt may increase pivtol as high as … to get a more accurate solution" |
  | jacobian_regularization_value | 1e-8 | Jacobian regularization |

  — [Ipopt options](https://www.coin-or.org/Ipopt/OPTIONS.html)
- **CuClarabel (GPU conic IPM, cuDSS LDL^T).**
  - Mixed precision factorizes an fp32 copy of the KKT matrix and solves in fp32. Residual and update are in fp64. fp64 is the default.
  - Static regularization is delta_s ≈ 3.45e-4 (sqrt of fp32 eps) in mixed mode vs 1e-8 in full precision.
  - IR runs at most 10 steps with t_abs = t_rel = 1e-12.
  - Huber QPs: mixed precision is 1.8–2x faster with identical IPM iteration counts (e.g. 598 s to 334 s).
  - Portfolio QPs: mixed results. n = 5000 is slower (0.18 s to 0.29 s), and n = 20000/25000 need more IPM iterations (19 to 23 and 19 to 21).
  - The authors warn mixed precision "may take longer to converge or fail" on ill-conditioned KKT systems, and advise caution on, e.g., exponential cones.
  - — [CuClarabel, arXiv 2412.19027](https://arxiv.org/html/2412.19027)
- **Condensed-space NLP IPMs on GPU (MadNLP + cuDSS).**
  - HyKKT and LiftedKKT condense to an SPD matrix so that Cholesky works "without numerical pivoting".
  - kappa_2(K_gamma) = Theta(max(1/s_min, gamma)). Structured ill-conditioning damps errors in the large-eigenvalue subspace, so the solution is accurate despite high kappa_2, but "an accurate iterative refinement algorithm is necessary".
  - The analysis is in fp64 (u = 1.1e-16); no fp32 is used.
  - Most IPMs use Richardson iterations on the unreduced K3 system for refinement.
  - — [Pacaud, Shin, Anitescu et al., arXiv 2405.14236](https://arxiv.org/html/2405.14236v3)
- **MadNCL.** Sets cuDSS pivot epsilon to 1e-10 (pivot regularization). It recovers the original-system solution by iterative refinement (Richardson by default) when inertia is correct. — [MadNCL, arXiv 2510.05885](https://arxiv.org/abs/2510.05885)
- **cuOpt barrier (LP).** The barrier method uses cuDSS for sparse Cholesky/LDL^T with Mehrotra predictor-corrector. NVIDIA reports 8.81x geomean over an open-source CPU solver on 61 Mittelmann LPs, but publishes no precision or accuracy details. — [NVIDIA cuOpt barrier blog](https://developer.nvidia.com/blog/solve-linear-programs-using-the-gpu-accelerated-barrier-method-in-nvidia-cuopt/); [cuOpt 25.10 release notes](https://docs.nvidia.com/cuopt/user-guide/25.10.00/release-notes.html)
- **Earlier mixed-precision LP IPM.** A regularized inexact IPM for LP uses single-precision Cholesky with CG. It switches to full double precision once CG time grows too large relative to factorization time, because the fp32 speedup is then lost. — [arXiv 2105.01333](https://arxiv.org/pdf/2105.01333)
- **WLS / augmented-system evidence.** Low-precision preconditioners for [D^-1 A; A^T 0] are guaranteed only for kappa(D) up to about 1e6–1e9, and IPMs are named as producing worse D. — [arXiv 2401.03755](https://arxiv.org/pdf/2401.03755)
- **Conditioning beyond IPMs.** A 2026 Newton/Riccati mixed-precision IR paper finds fp32 speedups shrink as conditioning worsens and recommends avoiding low-precision inner solves for highly ill-conditioned problems without preconditioning. — [arXiv 2607.14742](https://arxiv.org/pdf/2607.14742)

### Inferences
- **Late-IPM KKT systems.**
  - In late iterations, Sigma = X^-1 Z has entries ranging from about 1e-21 to about 1e+21 (complementarity mu down to about 1e-11). Equilibration (MC64/MC77) cannot remove this two-sided spread because it is intrinsic to the active/inactive partition.
  - kappa_eff is therefore far above fp32's 1/u_f ≈ 1.7e7. fp32 LU-IR is outside its convergence regime.
  - fp32 GMRES-IR with fp64 u_p is guaranteed (backward error) only to about 5e7 and observed to work to about 1e10–1e15 on SuiteSparse. It is still unlikely to be reliable at kappa_eff ≥ 1e20 without exploiting structure.
  - The structured ill-conditioning argument (Wright; Pacaud et al.) relies on backward-stable fp64 elimination producing errors aligned with the large-eigenvalue subspace. A u_f = 6e-8 factorization perturbs the small pivots by about 6e-8 × max entry, which wipes out 1e-21 pivots entirely. In effect it is static pivoting with tau about 6e-8. That is roughly the CuClarabel experience: they had to raise regularization to sqrt(eps_32) ≈ 3.45e-4 to stay stable.
- **Practical regimes for an FP32 path in smf.**
  - **(a) Early and middle IPM iterations.** mu is still moderate and the KKT spread is bounded, so fp32 LDL^T plus fp64 residual IR may hit the 1e-10 gate in about 2–4 steps.
  - **(b) Late iterations.** Detect via IR stagnation (ratio > 0.5 per step or more than about 5 steps) and fall back to the fp64 factorization, as LAPACK DSGESV does. Alternatively use fp32 factors as the preconditioner for FGMRES with fp64 matvec and fp64 preconditioner application, and accept tens of iterations.
  - **(c) Condensed SPD formulations** (HyKKT/LiftedKKT) are the GPU path with published success, all in fp64 to date.
- **IPOPT's loop composes poorly with mixed precision.** Its outer loop (min 1 / max 10 steps, ratio 1e-10, raise pivtol to 1e-4, refactor) would treat an fp32-factor IR failure as a pivtol problem and refactor in the same precision. smf should own the precision escalation internally (fp32 to fp64 refactor) and only report failure to IPOPT after fp64 also fails.

### Gaps
- I found no 2022–2026 peer-reviewed paper titled or framed as a "mixed-precision interior-point method" for NLP with fp32 sparse LDL^T, and no MadNLP fp32 results.
- HyKKT (Regev et al. 2022) itself was not fetched; its fp64 CG-on-Schur-complement details are from the Pacaud et al. summary only.
- No IPM-specific data on the number of IR steps vs mu for fp32 factors.

---

## 5. Cost-aware stopping criteria, multi-RHS refinement, and practical defaults (implementation guidance for smf)

### Takeaway
Stop on the backward error you are gated on (componentwise omega, normwise eta), not on step count. Stop early on stagnation (contraction ratio > about 0.5) and escalate rather than iterate. Budget refinement by its cost relative to the factorization. Sparse IR is cheap per step (solve flops are about 1e-3 of factorization flops in the MUMPS study), but GPU solves and many RHS change that. Defaults from production codes cluster at 2–10 steps with tolerances of 1e-10 to 1e-12.

### Cited Findings
- **Solve vs factorization cost.**
  - In well-behaved MUMPS cases "the complexity of the solve is much lower than that of the factorization (three orders of magnitude in all cases)".
  - The gain "is considerably reduced when many solves are necessary". This is worse for GMRES-IR because its solves run in fp64 instead of fp32.
  - — [Amestoy et al. TOMS 2023, Sec. 5.3.1](https://www.lip6.fr/Theo.Mary/doc/SparseIR.pdf)
- **GMRES stopping tolerance.** It was fixed at 1e-6 in the MUMPS experiments and "could be tuned on a case by case basis". The iteration count is "very sensitive to the choice of the GMRES stopping criterion". — [Amestoy et al. TOMS 2023](https://www.lip6.fr/Theo.Mary/doc/SparseIR.pdf); [GMRES-IR5](https://www.lip6.fr/Theo.Mary/doc/GMRESIR5.pdf)
- **Krylov memory vs restarts.** Krylov basis memory can exceed BLR savings when iterations are many. A tighter tau_g or restarted GMRES is the suggested mitigation. — [Amestoy et al. TOMS 2023, Sec. 5.4.1](https://www.lip6.fr/Theo.Mary/doc/SparseIR.pdf)
- **Production defaults.**

  | Solver | Default refinement behaviour |
  |---|---|
  | IPOPT | 1–10 steps, residual ratio 1e-10, improvement factor 1 |
  | PARDISO | 2 steps automatically when perturbed pivots appear |
  | cuDSS | 0 steps and tolerance 0 (refinement off) |
  | CuClarabel | ≤ 10 steps, 1e-12 absolute and relative |

  — [Ipopt](https://www.coin-or.org/Ipopt/OPTIONS.html); [MKL PARDISO](https://www.irya.unam.mx/computo/sites/manuales/fce12/mkl/mkl_manual/ssr/ssr_pardiso_parameters.htm); [cuDSS](https://docs.nvidia.com/cuda/cudss/latest/types.html); [CuClarabel](https://arxiv.org/html/2412.19027)
- **Error bounds at small cost.** Extra-precise IR provides error bounds at small cost after one application. — [Demmel et al. UCB/CSD-04-1344](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2004/5919.html)
- **Mixed-precision SpMV payoff is modest.** fp32 SpMV averages a 1.5x speedup over fp64, and "some form of refinement is needed" to recover fp64 accuracy. — [Zounon et al.](https://eprints.maths.manchester.ac.uk/2811/1/paper.pdf)

### Inferences (implementation guidance for smf)
1. **Stopping test.**
   - Compute both eta = ||r||_inf / (||A||_inf·||x||_inf + ||b||_inf) and the componentwise omega = max_i |r_i| / (|A||x| + |b|)_i. Use the Arioli–Demmel–Duff split into omega_1/omega_2 for rows where the denominator is tiny; the threshold is about n·u·(||A||·||x||+||b||).
   - Stop when eta ≤ 1e-10 and omega ≤ 1e-8, which is smf's gate.
   - Also stop when ratio_k = eta_k/eta_(k-1) > 0.5 (stagnation) or k = k_max. Use k_max = 3 for fp64 factors, which almost always need 0–2 steps. Use k_max of about 5–6 for fp32 factors.
   - Compute |A||x| in the same SpMV pass as r, so the extra cost is about 1 SpMV.
2. **Escalation ladder** (each rung only if the previous one stagnates):
   - (i) classical IR with the existing factors;
   - (ii) FGMRES(m = 10–20) preconditioned by the same factors, with fp64 matvec and fp64 preconditioner application, stopping on the same eta/omega;
   - (iii) refactor with a tighter threshold (2x2 / larger pivtol) or, on a mixed path, in fp64;
   - (iv) report failure to IPOPT (residual_ratio_singular).
   - This mirrors DSGESV's fall-back and Arioli et al.'s FGMRES remedy.
3. **Cost-aware budget.**
   - Track t_solve/t_factor per matrix. Allow refinement work up to about 0.3–0.5·t_factor before escalating to a refactorization.
   - On CPU with fp64 factors, t_solve/t_factor is about 1e-2 to 1e-3, so a handful of steps is free. On a GPU with fp32 factors, triangular solves are latency-bound and can be 5–20% of factorization (unsourced estimate), so stop IR earlier and escalate sooner.
4. **Multi-RHS.**
   - IPOPT's second-order correction and inertia-free steps solve several RHS against one factorization. Refine all of them together as a block: blocked SpMV, and triangular solves with nrhs > 1 that use BLAS3 TRSM on the fronts.
   - Use a per-column convergence mask so converged columns drop out.
   - The block solve amortizes front traversal, which dominates sparse solve time. This is a design inference; no source with numbers was found this session.
5. **Precision choices for a future fp32 path.**
   - Factor in fp32 with FTZ/DAZ on and fronts scaled to avoid underflow (mandatory per Zounon et al.).
   - Compute the residual in fp64. fp128 or double-double is cheap for sparse residuals (Amestoy et al.) but unnecessary for a 1e-10 gate.
   - Apply the preconditioner in fp64 inside GMRES/FGMRES only when LU-IR stagnates (the u_p = d "sdd" variant). Keep factors in fp32 and upcast per front during the solve to avoid the 2x memory cast seen in the MUMPS implementation.
   - Expect benefit only on large KKT systems where numeric factorization is at least 70% of the time.
6. **Static pivoting defaults.**
   - tau = 1e-8·max|a_ij| of the scaled matrix (PARDISO default; matches sqrt(eps_64) in SuperLU_DIST).
   - Always refine at least once when any pivot was perturbed (PARDISO behaviour).
   - Expose the perturbed-pivot count to the IPOPT interface, because it affects inertia.

### Gaps
- Exact LAPACK xGERFSX stagnation constants (the 0.5 contraction threshold and iteration caps) were not verified from source this session. The 0.5 rule above is a common practice, stated as an inference.
- No published measurements of GPU sparse triangular-solve cost relative to factorization for KKT matrices were found. The 5–20% figure is an unsourced estimate.
- No source found on block/multi-RHS iterative refinement performance for sparse LDL^T.
