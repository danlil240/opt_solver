# Factorization kernels, parallelism, memory and GPU for sparse symmetric (Cholesky / LDL^T) direct solvers — state of the art, Oct 2026

Scope note: findings come from roughly 20 web searches and fetches done on 2026-10-27. Many primary papers (TOMS/SISC full texts) could not be fetched; where only an abstract or catalogue record was seen, the note says so. Items in "Inferences" that come from the researcher's background knowledge, not from a source fetched this session, carry the tag **[bg, not re-verified]**. The report writer should treat those as leads to check, not as cited facts.

Target context: smf is a C++20 MA97-like multifrontal LDL^T solver. It has a lean serial engine, a std::thread tree-parallel engine and a tree-parallel solve. It is strong at 1 thread, weak on chains of thin top fronts, and nondeterministic in parallel. The development machine is Windows with 16 threads and an RTX 5080.

## 1. Supernodal vs multifrontal for KKT matrices; hybrids

### Takeaway
No source fetched this session gives a modern, controlled supernodal-vs-multifrontal benchmark on KKT matrices. The evidence is indirect. The robust indefinite solvers (MA97, SSIDS, MUMPS) are multifrontal with threshold or a-posteriori pivoting. PARDISO is supernodal and gets its speed partly from supernode-restricted Bunch-Kaufman pivoting with static perturbation, at some cost in robustness. On IPM KKT systems the deciding factor is pivoting strategy, not which looking pattern is used.

### Cited Findings
- Duff, Hogg and Lopez (SISC 42(2), 2020, C23–C42) note that pivoting usually limits parallelism. They cite PARDISO's supernode Bunch-Kaufman as an example of trading stability for performance: "the factors obtained might not be used to accurately compute the solution." — [SIAM epubs 10.1137/18M1225963](https://epubs.siam.org/doi/abs/10.1137/18M1225963)
- The same paper's APTP multifrontal solver (SSIDS v2 CPU path) "performs favorably" against HSL_MA86, HSL_MA97 and PARDISO in both multicore performance and numerical robustness. Only the abstract was seen; the numeric tables were not retrieved. — [SIAM epubs](https://epubs.siam.org/doi/10.1137/18M1225963); [STFC record](https://epubs.stfc.ac.uk/work/30692911)
- COMSOL documents that cuDSS, like PARDISO, relies on pivot perturbation plus iterative refinement. This is the static-pivoting family. — [COMSOL 6.4 solver docs](https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/comsol_ref_solver.36.144.html)
- An independent CPU comparison on severely ill-conditioned geophysical systems found oneMKL PARDISO had the best time to solution and median time per iteration, closely followed by UMFPACK, KLU and MUMPS. These are not KKT systems. — [arXiv 2409.11515](https://arxiv.org/html/2409.11515v2)
- HSL_MA87 (supernodal-ish block DAG Cholesky, SPD only) "generally outperforms" existing modern solvers, particularly on very large problems, on 8-core machines. Its efficiency depends on the user-supplied elimination order. — [Hogg, Reid, Scott SISC 2010 PDF](https://www.numerical.rl.ac.uk/media/people/jennifer-scott/HoggReidScot2010_sisc.pdf); [HSL_MA87 catalogue v2.6.6](https://www.hsl.rl.ac.uk/catalogue/hsl_ma87.html)
- SSIDS documentation states it is designed primarily for indefinite systems and "may be slower than a dedicated Cholesky solver when the matrix is positive definite." — [SPRAL SSIDS docs 2023.08.02](https://www.numerical.rl.ac.uk/spral/doc/v2023.08.02/C/ssids.html)

### Inferences
- For IPM KKT (quasi-definite after regularization, with pivoting needed near degeneracy), multifrontal with threshold or a-posteriori pivoting is the robust standard. A "supernodal" layout matters mainly for the solve phase and for avoiding contribution-block copies.
- smf's in-place fronts plus packed contribution stack already sit between the two styles. The useful hybrid is **multifrontal factorization with supernodal (column-panel) factor storage**, which smf likely already has, plus **left-looking or relaxed-amalgamation treatment of thin top-chain fronts** (Section 2).
- **[bg, not re-verified]** Generally reported, e.g. in Davis's 2016 Acta Numerica survey: multifrontal pays extra memory traffic for extend-add, while supernodal left-looking has better locality for thin supernodes. Right-looking supernodal and multifrontal expose more BLAS-3 on fat fronts. Chains of thin fronts are where left-looking or merged-chain treatment wins.

### Gaps
- No 2015–2026 head-to-head of PARDISO / CHOLMOD / MUMPS / SSIDS / MA97 specifically on IPOPT-style KKT matrices with per-phase times was retrieved. The Duff–Hogg–Lopez tables and the Tasseff et al. IPOPT linear-solver study ([arXiv 1909.08104](https://arxiv.org/pdf/1909.08104)) are the places to look.

## 2. Dense kernels for frontal LDL^T (pivoting, recursion, small-front batching, SIMD, BLAS-3 fraction)

### Takeaway
The leading CPU technique for indefinite fronts is **a posteriori threshold pivoting (APTP)**, used by SSIDS. It factors a block column optimistically with BLAS-3 and no pivoting, checks the threshold afterwards, and backtracks only the failed columns. This keeps threshold-partial-pivoting robustness while recovering a task-parallel, BLAS-3-dominated kernel. GPUs instead use static pivoting or Bunch-Kaufman within supernodes, with perturbation and refinement. cuDSS added Bunch-Kaufman only in 1.0.0.

### Cited Findings
- APTP (Duff, Hogg, Lopez 2020) is "much more suitable for modern multicore architectures and has the same numerical robustness as the TPP strategy." It is implemented as OpenMP 4.0 tasks in a multifrontal code. — [SIAM epubs](https://epubs.siam.org/doi/abs/10.1137/18M1225963); [SSIDS docs](https://www.numerical.rl.ac.uk/spral/doc/v2023.08.02/C/ssids.html)
- SSIDS v2.0.0 (2016) added CPU and hybrid execution. The CPU path uses OpenMP 4.0 tasks with a posteriori threshold pivoting. v1.0.0 (2014) was GPU-only. — [SSIDS docs v2016](https://www.numerical.rl.ac.uk/spral/doc/v2016.09.23/C/ssids.html)
- The SSIDS GPU paper (Hogg, Ovtchinnikov, Scott, ACM TOMS 2016) lists the GPU challenges for indefinite sparse factorization: "thousands of small dense subproblems, pivoting on large dense blocks, and substantial sparse scatter/gather." Early speedups over a host CPU were about 2.2–6.8x on C2050/K20 hardware. — [STFC sparse_direct_gpu slides](https://epubs.stfc.ac.uk/manifestation/12167414/sparse_direct_gpu.pdf); [STFC record](https://epubs.stfc.ac.uk/work/12189719)
- cuDSS 1.0.0 release notes add "Bunch-Kaufman pivoting for symmetric/Hermitian indefinite matrices." Earlier versions offered LDL^T without 2x2 pivoting. The MadNLP paper states cuDSS's "2x2 pivoting strategy is not supported," which limited augmented-system IPMs, and the authors switched to SPD condensed systems. — [cuDSS release notes](https://docs.nvidia.com/cuda/cudss/latest/release_notes.html); [arXiv 2405.14236](https://arxiv.org/html/2405.14236v3)
- cuDSS 0.7.0 reintroduced `CUDSS_ALG_1` "improved for very sparse factors." This is a separate algorithm path for tiny supernodes, which is the GPU analogue of an unblocked small-front kernel. — [cuDSS release notes](https://docs.nvidia.com/cuda/cudss/latest/release_notes.html)
- Tacho and a TBB multifrontal study use a two-level scheme. Many small tasks run in parallel at the bottom; parallel BLAS handles large fronts at the top. Synchronization overhead and memory management are the main obstacles. — [TBB multifrontal (Springer)](https://link.springer.com/chapter/10.1007/978-3-030-36592-9_7); [Tacho (Sandia)](https://www.sandia.gov/research/publications/details/tacho-memory-scalable-task-parallel-sparse-cholesky-factorization-2018-08-03/)
- On A64FX, CHOLMOD with selective OpenMP nesting avoids creating many small inner tasks and reports average speedups over the non-nested baseline. — [arXiv 2202.09288](https://arxiv.org/pdf/2202.09288)

### Inferences
- **For smf:** the biggest kernel-level upgrade for indefinite fronts is an APTP-style blocked kernel. Factor a block of nb columns with LDL^T (1x1 only, no row swaps), run TRSM and the Schur update, check |l_ij| ≤ 1/u, then accept the passed prefix and delay or retry the failed columns with 2x2 Bunch-Kaufman/TPP. This turns the pivoting kernel into GEMM-dominated work and parallelizes within a front. MA97 itself uses a classic threshold kernel.
- **[bg, not re-verified]** SSIDS's CPU code uses a custom small-block LDL^T kernel with block size around 32 for the APTP inner kernel and calls BLAS-3 for the outer updates. MA86 uses blocks of nb=256 with tasks per block. PARDISO uses supernodal Bunch-Kaufman with eps·||A|| pivot perturbation, scaling plus weighted matching, and iterative refinement.
- The thin-top-chain weakness is a classic symptom of BLAS-3 with small inner dimension (k = pivots per front) on a few large-m fronts. Remedies to test:
  1. relaxed amalgamation of chain fronts, so fewer, fatter supernodes;
  2. for a front with npiv ≪ nrow, a fused update kernel (rank-k SYRK/GEMMT with small k is bandwidth-bound — consider a single pass that applies the update directly into the parent's slots, the "multifrontal with direct assembly" trick);
  3. intra-front parallelism (2D block tasks) on top fronts instead of relying on threaded BLAS.
- An AVX-512/AVX2 microkernel matters mainly below the BLAS cutoff (fronts < 96). Batching many tiny leaf fronts in one call, or a register-blocked LDL^T for n ≤ 32, is the CPU analogue of cuDSS ALG_1 and SSIDS GPU small-front batching.

### Gaps
- No source gave a measured BLAS-3 flop fraction or per-kernel GFLOP/s for SSIDS, MA86/87 or PARDISO on modern AVX-512 CPUs. The cuDSS internal kernel design (supernodal vs multifrontal, left vs right looking) is not documented publicly.
- No numeric results from the Duff–Hogg–Lopez APTP paper were retrieved.

## 3. Tree and node parallelism, task runtimes, determinism, scaling

### Takeaway
The proven design is **subtree-to-thread mapping at the bottom (Geist–Ng / proportional mapping) plus a fine-grained task DAG over 2D blocks of the top fronts**. This is the MA87/SSIDS/PaStiX/Tacho design. Generic runtimes (StarPU, PaRSEC) match hand-tuned schedulers on CPU-only nodes and pay off mainly on heterogeneous CPU+GPU nodes. Determinism needs a fixed reduction order for extend-add/assembly, independent of thread timing. cuDSS ships an explicit deterministic mode.

### Cited Findings
- HSL_MA87: work at each assembly-tree node is split into block tasks whose dependencies form a DAG (Fortran 95 + OpenMP). It got good serial and parallel times on 8-core machines. — [Hogg, Reid, Scott SISC 2010](https://www.numerical.rl.ac.uk/media/people/jennifer-scott/HoggReidScot2010_sisc.pdf)
- PaStiX with PaRSEC and StarPU in place of its internal scheduler gave "comparable results to the application-optimized embedded scheduler on homogeneous platforms." The runtimes "significantly speed up the solver on heterogeneous environments" by using GPUs. The paper claims the first supernodal sparse direct solver with hybrid multicore + multi-GPU execution. — [arXiv 1405.2636](https://arxiv.com/abs/1405.2636); [ICL page](https://icl.utk.edu/node/709)
- A sequential-task-flow sparse Cholesky on StarPU is "as efficient as a state-of-the-art solver on a multicore system." — [AIMS NACO 2018](https://www.aimsciences.org/article/doi/10.3934/naco.2018014)
- Tacho builds tasks while walking the assembly tree, uses a custom memory pool for frontal workspace, and re-queues a task that cannot get memory so the thread runs other work. Tree parallelism shrinks near the root, so nested parallelism inside supernodal blocks is needed. — [Tacho Sandia record](https://www.sandia.gov/research/publications/details/tacho-memory-scalable-task-parallel-sparse-cholesky-factorization-2018-08-03/); [arXiv 1601.05871](https://arxiv.org/pdf/1601.05871)
- cuDSS 0.7.0 added `CUDSS_CONFIG_DETERMINISTIC_MODE`. Even the GPU vendor treats bitwise reproducibility as an opt-in mode with a performance cost. — [cuDSS release notes](https://docs.nvidia.com/cuda/cudss/latest/release_notes.html)
- The search found no 2023–2024 paper that directly addresses bitwise-reproducible parallel multifrontal factorization. Dynamic task scheduling can change floating-point summation order from run to run.

### Inferences
- **Root cause of smf nondeterminism (likely):** contribution blocks from children are extend-added into the parent in completion order. Atomic or locked accumulation into shared parent storage, or threaded BLAS with dynamic partitioning, would also cause it. Fixes in order of cost:
  1. Assemble children into the parent in a fixed (child-index) order. A child that finishes early parks its CB on the stack; the parent assembles them all in canonical order when the last child completes. This is the standard MA97 behaviour. **[bg, not re-verified]** MA97 advertises bit-compatible results independent of thread count.
  2. Make the thread budget passed to MKL deterministic: fixed nthreads per task plus `MKL_CBWR` (conditional numerical reproducibility) set to a fixed code path. MKL's CNR mode guarantees reproducibility only with a fixed thread count unless `MKL_CBWR=...,STRICT` is used **[bg, not re-verified]**.
  3. For 2D-block DAG updates on top fronts, enforce a fixed order for updates into the same target block. Use either per-target-block update chains, like MA87's "update tasks applied in a fixed order", or private accumulation followed by an ordered sum.
- **[bg, not re-verified]** Geist–Ng (1989) maps subtrees to processors greedily until the remaining subtrees balance. Proportional mapping (Pothen–Sun) assigns processor sets proportional to subtree flops. With 16 threads on Windows a simple scheme works well: the Geist–Ng layer for leaves, then a work-stealing DAG for the top with 2D tiling of fronts larger than about 1–2k. MUMPS on shared memory and SSIDS both do roughly this.
- Thin top chains need *intra-front* tasks, not tree parallelism. Split each top front into 2D tiles (nb ≈ 128–256) so that TRSM and update tiles of successive chain fronts overlap. This gives "lookahead across the chain", which a per-front threaded BLAS call cannot provide.

### Gaps
- No 8–64-core scaling numbers for SSIDS, MA87, Tacho or PaStiX were retrieved with concrete values. The abstracts claim good scaling only qualitatively.
- No paper found that quantifies the cost of deterministic reductions in sparse factorization.

## 4. Memory: contribution stack, in-place assembly, memory-aware scheduling, out-of-core, NUMA

### Takeaway
The sources seen this session confirm the practical pattern of **preallocated memory pools for fronts and CBs, with tasks that defer when the pool is exhausted** (Tacho). Out-of-core is mostly a CPU-cluster or GPU-hybrid concern. In cuDSS it appears as "hybrid memory mode", with factors in host RAM. A 16-thread desktop needs a per-thread arena and a bounded tree traversal, not out-of-core.

### Cited Findings
- Tacho uses a custom memory pool for frontal workspace. A task that cannot get memory is re-queued so its thread can run other work. This gives memory scalability under task parallelism. — [Tacho (Sandia)](https://www.sandia.gov/research/publications/details/tacho-memory-scalable-task-parallel-sparse-cholesky-factorization-2018-08-03/)
- The TBB multifrontal work names synchronization overhead and memory management as the main obstacles on shared-memory machines with dozens of cores. — [Springer](https://link.springer.com/chapter/10.1007/978-3-030-36592-9_7)
- cuDSS hybrid memory mode is off by default and keeps factors in host memory. It uses device memory first and touches host memory only when needed. A per-device memory limit can be set between `CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN` and the memory available after analysis. Performance depends on CPU–GPU bandwidth. — [NVIDIA blog, large-scale cuDSS](https://developer.nvidia.com/blog/solving-large-scale-linear-sparse-problems-with-nvidia-cudss)
- cuDSS 0.4.0 added memory-estimate queries after analysis. 0.5.0 improved hybrid-memory performance and memory use. 1.0.0 added pinned-host-memory handlers to speed hybrid mode. — [cuDSS release notes](https://docs.nvidia.com/cuda/cudss/latest/release_notes.html)

### Inferences
- **[bg, not re-verified]** Classic results (Liu 1986; Guermouche–L'Excellent; MUMPS) say the order of children determines the peak of the CB stack, and post-ordering children by decreasing peak minimises it. In parallel, memory-aware mapping (MUMPS) caps peak memory by restricting which subtrees run concurrently. smf's packed CB stack is the right design. In parallel mode each Geist–Ng subtree should own its own stack segment, sized from the analyse-phase per-subtree peak, so nothing is allocated during factorization.
- Allocation avoidance: compute all front, CB and stack sizes symbolically. Allocate one arena per thread at analyse or first factorization and reuse it across IPM refactorizations. Delayed pivots make sizes data-dependent, so reserve slack (e.g. +x%) and fall back to a growable overflow region.
- NUMA is irrelevant on a single-socket 16-thread desktop. First-touch by the owning thread is enough.

### Gaps
- No 2015–2026 quantitative memory-peak data, e.g. parallel vs serial peak ratios for MUMPS or SSIDS, was retrieved.

## 5. Triangular solve performance (parallel, multi-RHS, selective inversion, layouts)

### Takeaway
Solve is memory-bandwidth-bound. The established gains come from tree-parallel solve over supernodal panels, multi-RHS blocking (turning TRSV/GEMV into TRSM/GEMM), and synchronization-free or level-scheduled GPU solves. GPU solves are dramatically faster in absolute terms (e.g. 7 ms vs 52–89 ms on a 675k KKT system).

### Cited Findings
- On a 674,562-unknown condensed OPF KKT matrix, backsolve times were CHOLMOD 0.052 s, PARDISO 0.089 s, MA86 0.055 s (8 threads, AMD EPYC 7443), against cuDSS 0.0064–0.0072 s on an A100. That is about 7–14x faster on the GPU. — [arXiv 2405.14236v3](https://arxiv.org/html/2405.14236v3)
- cuDSS hybrid memory mode gives a 2.3x solve speedup on GB200 vs B200 + x86 for 1–18M-row matrices, which tracks CPU–GPU link bandwidth. MG mode cut solve time from about 0.25 s (1 H200) to about 0.09 s (4 H200) on a 31M-row, 1B-nnz matrix. — [NVIDIA blog](https://developer.nvidia.com/blog/solving-large-scale-linear-sparse-problems-with-nvidia-cudss)
- cuDSS 0.8.0 added "faster multi-right-hand-side solves". 1.0.0 added faster solves for large uniform batches with up to 16 RHS. — [cuDSS release notes](https://docs.nvidia.com/cuda/cudss/latest/release_notes.html)
- Liu, Li, Hogg, Duff and Vinter's synchronization-free SpTRSV/SpTRSM report an average speedup of over 2x for SpTRSV and up to an order of magnitude for SpTRSM (multi-RHS) against a vendor library on 20 matrices. — [NTNU Open](https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2494579)
- A Kokkos supernode-based sparse triangular solver (Yamazaki, Rajamanickam, Ellingwood 2020) was 12.4x / 19.5x faster than cuSPARSE on V100 / P100. — [Sandia record](https://www.sandia.gov/research/publications/details/performance-portable-supernode-based-sparse-triangular-solver-for-manycore-2020-08-17/)
- The MUMPS BLR triangular solve with many RHS (SIMAX 2024) got up to 20% time reduction on real applications. — [HAL hal-04082415](https://hal-emse.ccsd.cnrs.fr/IRIT-APO/hal-04082415v1)

### Inferences
- **[bg, not re-verified]** "Partitioned inverse" (Alvarado–Pothen–Schreiber) and selective inversion (Raghavan; used in PSPIKE) replace sequential TRSVs on supernode diagonal blocks with GEMVs on explicitly inverted diagonal blocks. This gives roughly 2x on solve-heavy workloads, at the cost of an O(n_s^3) inversion at factor time, and reduces precision slightly. The Yamazaki et al. Kokkos solver uses exactly this "invert the diagonal block" trick on GPU.
- For IPM, solves per factorization are typically 1 plus iterative-refinement steps plus second-order corrections, so about 2–6 solves. Storing L panels in a solve-friendly contiguous column-major layout with the diagonal block's inverse (D^{-1}·L_11^{-1}) cached can cut solve time materially. smf already closed the IPOPT backsolve gap; the next step is caching inverted diagonal blocks for fronts with npiv ≤ 64 and batching multi-RHS in refinement.

### Gaps
- No 2020s CPU-side study was found giving solve/factor time ratios for PARDISO, MUMPS or SSIDS on KKT matrices.

## 6. Block low-rank (BLR) and hierarchical compression

### Takeaway
BLR (MUMPS) and HSS/HODLR (STRUMPACK) pay off on large 3D-PDE-like fronts (front sizes in the thousands, smooth kernels). There the gains are 2.5–4x in factorization time at 900 cores and up to 7x in STRUMPACK's suite. For IPM KKT matrices from optimisation (OPF, CUTEst), fronts are small and rank structure is weak or destroyed by pivoting. **BLR is not a priority for smf** unless PDE-constrained problems with 3D meshes become a target.

### Cited Findings
- BLR reduces the multifrontal factorization complexity for 3D problems from O(n^2) to about O(n^1.33) (theory, 2016–2017). — [T. Mary thesis 2017](https://www-pequan.lip6.fr/~tmary/doc/thesis.pdf); [Mary slides](https://www-pequan.lip6.fr/~tmary/doc/JLC16.pdf)
- Industrial cases (structural mechanics, seismic imaging, electromagnetism) on 900 cores showed BLR factorization roughly 2.5–4x faster than the full-rank reference, e.g. 386 s vs 131 s for structural mechanics. The extracted table was partly garbled. — [Mary slides PHDD16](https://www-pequan.lip6.fr/~tmary/doc/PHDD16.pdf)
- The multicore BLR performance and scalability paper is Amestoy, Buttari, L'Excellent, Mary, ACM TOMS 45(2), 2019. Its headline numbers were not retrieved. — [Mary thesis](https://www-pequan.lip6.fr/~tmary/doc/thesis.pdf)
- BLR combined with mixed-precision storage (Gerest, Boiteau, Buttari, Jézéquel, L'Excellent, Mary) reduces time and memory "without compromising the accuracy." — [M. Gerest page](https://lip6.fr/Matthieu.Gerest)
- STRUMPACK uses randomized HSS compression with ULV factorization, giving "speedups up to sevenfold" in its test suite. Its sparse side is multifrontal LU with rank-structured fronts (HSS, BLR, HODLR, HODBF). — [Ghysels et al. (ULB record)](https://difusion.ulb.ac.be/vufind/Record/ULB-DIPOT:oai:dipot.ulb.ac.be:2013/200878/TOC); [Ghysels LBNL slides](https://simber.simula.no/wp-content/uploads/2022/12/ghysels_lbnl_sparse_rank_structured.pdf)
- MUMPS interleaves compression with panel factorization, while STRUMPACK compresses the whole front first and then does ULV. BLR has worse asymptotic complexity than H/HSS but "works well in practice." — [Amestoy et al. CSE'17 slides](https://www-pequan.lip6.fr/~tmary/doc/CSE17.pdf); [MUD17](https://www-pequan.lip6.fr/~tmary/doc/MUD17.pdf)

### Inferences
- Indefinite pivoting complicates BLR: delayed pivots move rows between fronts. MUMPS handles this, but the gains reported are mostly on SPD or mildly indefinite PDE operators. For optimisation KKT matrices with max front sizes in the low thousands, BLR overhead would likely exceed savings.

### Gaps
- No study was found that applies BLR or HSS specifically to IPM KKT matrices (PDE-constrained optimisation) with timing data.

## 7. GPU sparse direct: cuDSS, SSIDS GPU, CHOLMOD GPU, PaStiX GPU, HyKKT; consumer GPUs with weak FP64

### Takeaway
cuDSS is now the de facto GPU sparse direct solver. It reached 1.0.0 with Bunch-Kaufman LDL^T, FP64 emulation, mixed-precision factors, deterministic mode, hybrid memory and execution, MG/MGMN and uniform batching. On a datacenter A100 its numeric factorization is about 3.3x faster than MA86 and 7–18x faster than CHOLMOD/PARDISO on a 675k OPF KKT matrix. Analysis (ordering on CPU) is about 2.6x slower than CHOLMOD's. For the RTX 5080, native FP64 is a tiny fraction of INT8/FP32 throughput, so practical routes are cuDSS's opt-in FP64 emulation (Ozaki scheme on INT8 tensor cores) or FP32 factors with FP64 refinement.

### Cited Findings
- **cuDSS per-phase on a KKT-type matrix** (condensed OPF K_γ, n = 674,562; AMD EPYC 7443 with 8 threads vs NVIDIA A100):
  - Analysis: CHOLMOD 1.37 s, PARDISO 4.52 s, MA86 3.88 s, cuDSS 3.55–3.63 s.
  - Factorization: CHOLMOD 0.744 s, PARDISO 0.294 s, MA86 0.138 s, cuDSS-Cholesky 0.0405 s, cuDSS-LDL^T 0.0421 s, cuDSS-LU 0.0628 s.
  - Backsolve: 0.052 / 0.089 / 0.055 s on CPU vs about 0.007 s on GPU.

  Ordering runs on the CPU, and analysis is reusable while the pattern is fixed. — [arXiv 2405.14236v3](https://arxiv.org/html/2405.14236v3)
- **End-to-end IPM** (MadNLP + ExaModels):
  - 78484_epigrids: MA27 CPU 191.1 s total (1.87 s/iteration, 86% in the linear solver); LiftedKKT + cuDSS 26.0 s; HyKKT + cuDSS 16.2 s (about 11.8x).
  - Per-iteration, augmented KKT with MA27: 1.23 s factor + 0.348 s solve. Lifted KKT with cuDSS: 0.049 s + 0.011 s.
  - Small cases lose: 1354_pegase MA27 0.46 s vs 1.05 s with cuDSS.
  - A cheap A1000 GPU already beats MA27. CUTEst robustness is weaker, with failures caused by dense rows and large symbolic costs.

  — [arXiv 2405.14236v3](https://arxiv.org/html/2405.14236v3)
- HyKKT runs CG on the Schur complement with an inner Cholesky (Regev et al., OMS 38, 2023). On an A100 vs MA57 on CPU, speedups were 0.62x (Illinois grid), 1.04x (Texas), 2.32x (Western US) and 10.5x (Eastern US: 3.48 s vs 0.331 s). The gain grows with size. — [Regev ICCOPT'22 talk](https://web.stanford.edu/group/SOL/talks/22ICCOPT-shaked-regev.pdf); [ORNL record](https://impact.ornl.gov/en/publications/hykkt-a-hybrid-direct-iterative-method-for-solving-kkt-linear-sys/)
- **cuDSS feature timeline:**
  - 0.1: Windows x86-64 supported from the start.
  - 0.3: MGMN and hybrid memory.
  - 0.4: non-uniform batch and memory estimates.
  - 0.5: MT mode (VCOMP on Windows), hybrid execute mode, Blackwell sm_120.
  - 0.6: matching and scaling, uniform batch.
  - 0.7: MG mode, deterministic mode, Schur complement mode, int64, ALG_1 for very sparse factors.
  - 0.7.1: reuse of a user-supplied etree and permutation.
  - 0.8: double-double type, hybrid execute in analysis, `CUDSS_PIVOT_AUTO`, faster multi-RHS and ubatch factorization, CUDA 13.
  - 1.0: Bunch-Kaufman LDL^T, opt-in FP64 emulation for factorization, factor downcast to FP32 with FP64 solves, in-place iterative refinement, pinned-host handlers.

  — [cuDSS release notes](https://docs.nvidia.com/cuda/cudss/latest/release_notes.html)
- cuDSS hybrid memory: factorization 1.4x to nearly 4x and solve 2.3x faster on GB200 vs B200 + x86 (1–18M rows). MG on an H200: factorization about 20 s (1 GPU) to under 10 s (4 GPUs) on a 31M-row matrix. No CPU-solver comparison in that blog. — [NVIDIA blog](https://developer.nvidia.com/blog/solving-large-scale-linear-sparse-problems-with-nvidia-cudss)
- An NVIDIA SparseDays 2025 deck (pre-0.6) reports hybrid-memory speedups averaging 2.1x for factorization and 2.5x for solve. This is vendor material; the full deck could not be fetched because of a TLS certificate error. — [SparseDays 2025 Rommes deck](https://sparsedays.cerfacs.fr/wp-content/uploads/sites/72/2025/07/20250602_SparseDays_Rommes_cuDSS_emulation_Dx.pdf)
- Vendor case study: cuDSS was 2–78x faster than Honeywell's in-house solver (A100 vs AMD EPYC), counting analysis + factorization + solve for cold starts and refactorization for hot starts. — [NVIDIA blog, Honeywell](https://developer.nvidia.com/blog/spotlight-honeywell-accelerates-industrial-process-simulation-with-nvidia-cudss/)
- An NVIDIA forum reply says cuDSS's advantage over CPU appears only above about 2–3k unknowns, and that host execution mode helps small matrices. In a thermal-simulation study, cuDSS setup cost lost to a CPU iterative solver for a single RHS. — [arXiv 2606.17850](https://arxiv.org/pdf/2606.17850); [NVIDIA forum](https://forums.developer.nvidia.com/t/why-i-cannot-get-correct-results-from-cudss-solver-for-a-simple-matrix/335506)
- CHOLMOD GPU (Rennich, Stosic, Davis 2014) streams entire elimination-tree branches through the GPU to avoid PCIe traffic. It gave a 2.4x average and 3.5x maximum speedup vs CPU on the 12 largest SuiteSparse matrices (K40m). Larger, higher-fill matrices gain more. — [Rennich/Stosic/Davis preprint](https://people.engr.tamu.edu/davis/publications_files/IA3_2014_Workshop_Rennich_Stosic_Davis_preprint.pdf); [NVIDIA CHOLMOD page](https://developer.nvidia.com/cholmod)
- PaStiX on PaRSEC/StarPU gave the first hybrid multicore + multi-GPU supernodal solver, with GPU kernels for the heaviest tasks. — [arXiv 1405.2636](https://arxiv.com/abs/1405.2636)
- **FP64 on consumer GPUs:**
  - CUDA 13.0 Update 2 cuBLAS added FP64 matmul emulation on Blackwell INT8 tensor cores (Ozaki scheme). It picks emulation or native FP64 per call with accuracy "equal to or better than native." NVIDIA notes that the mantissa bits needed depend on the data.
  - An Ozaki-II DGEMM emulation paper reports 2.3x over cuBLAS DGEMM on GB200 and 13.2x on an RTX Pro 6000 Blackwell in favourable cases.
  - A B200 paper's spec table shows the GeForce RTX 5080 with native FP64 far below its INT8 throughput.
  - No GeForce-specific benchmark was found.

  — [NVIDIA blog: FP emulation in cuBLAS](https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/); [arXiv 2512.08321](https://arxiv.org/html/2512.08321v3); [arXiv 2511.13778](https://arxiv.org/html/2511.13778v1)
- A review of GPU sparse solvers says GPU solvers now often beat CPU on positive-definite or quasi-definite systems that do not need sophisticated pivoting. SuperLU_DIST with GPU was slightly slower than CPU-only in one study. — [COMSOL / arXiv summaries via search](https://arxiv.org/pdf/2409.11515)

### Inferences
- **RTX 5080 reality check:** consumer Blackwell FP64 runs at 1/64 of the FP32 rate. **[bg, not re-verified]** That puts it near 0.8–1 TFLOP/s FP64, comparable to or below a modern 16-thread AVX2/AVX-512 desktop CPU running DGEMM (roughly 0.5–1.5 TFLOP/s). Native FP64 cuDSS on the 5080 is therefore unlikely to beat a good CPU factorization by much. Viable GPU paths:
  1. cuDSS 1.0 FP64 emulation (INT8 tensor cores), whose efficiency depends on the data's exponent range. KKT matrices with IPM barrier terms spanning 1e-8..1e8 are a worst case for Ozaki slicing.
  2. FP32 factorization ("factor downcast") with FP64 iterative refinement or GMRES-IR. This is risky near IPM convergence, where KKT condition numbers exceed 1e10, beyond FP32's about 1e7 reach.
  3. Use the GPU for the solve phase and Schur-complement GEMMs only.
- cuDSS is usable on Windows (since 0.1, MT mode with VCOMP since 0.5, sm_120 since 0.5). That makes it a cheap **benchmark baseline** for smf on the dev machine, not necessarily a component.
- The GPU wins in the literature come from datacenter A100/H100/B200 FP64 (about 10–60 TFLOP/s) on large OPF-like problems. Small problems (< about 10^4–10^5) lose to CPU because of launch and analysis overheads. This matches the 1354_pegase result.

### Gaps
- No benchmarks of cuDSS (native or emulated FP64) on a GeForce RTX 50-series card for sparse factorization were found.
- No numbers were retrieved for SSIDS-GPU vs cuDSS, or for current CHOLMOD 5.x GPU vs cuDSS.
- The SparseDays 2025 cuDSS-emulation deck could not be read in full.

## 8. Repeated factorization with fixed pattern (IPM): symbolic reuse, value-only refactor, GPU residency

### Takeaway
Every modern solver separates analysis from factorization and amortises ordering and symbolic work across IPM iterations. cuDSS goes further with uniform batches, reusable user etree and permutation, and fully GPU-resident pipelines (MadNLP). On the CPU, the main lever beyond symbolic reuse is **value-only refactor with a precomputed scatter map** (A-values to front positions). Allocation, index work and pivot-sequence setup are then all hoisted out of the loop. The pivot sequence is kept when still acceptable; delayed pivots force re-planning.

### Cited Findings
- cuDSS analysis (ordering + symbolic) is about 3.6 s vs a 0.04 s numeric factorization on a 675k KKT matrix. Analysis can be reused when the sparsity pattern is unchanged, so amortising it is essential. — [arXiv 2405.14236v3](https://arxiv.org/html/2405.14236v3)
- In MadNLP's GPU IPM, condensation took about 8 s on the CPU vs 0.17 s on the GPU. Keeping the whole IPM (function evaluation via ExaModels, KKT assembly, factorization) GPU-resident is what makes the 10x+ end-to-end gains possible. — [arXiv 2405.14236v3](https://arxiv.org/html/2405.14236v3)
- cuDSS 0.7.1 allows reusing a user-supplied elimination tree with a user permutation to cut reordering time. 0.6.0 added uniform-batch solves (same pattern, different values). 0.8.0 added selective processing inside uniform batches (`UBATCH_MASK`). 1.0.0 sped up large uniform batches with up to 16 RHS. — [cuDSS release notes](https://docs.nvidia.com/cuda/cudss/latest/release_notes.html)
- The Honeywell case study measured cold start (analysis + factorization + solve) separately from hot-start refactorization, showing refactorization as the steady-state cost. — [NVIDIA blog, Honeywell](https://developer.nvidia.com/blog/spotlight-honeywell-accelerates-industrial-process-simulation-with-nvidia-cudss/)

### Inferences
- For smf (CPU-first), a fixed-pattern refactor should do zero allocations and zero index searches. Precompute:
  1. a per-front scatter list for original entries;
  2. extend-add relative-index maps per child;
  3. the arena layout.

  Delayed pivots break the static maps. Keep a fast path while the previous iteration's pivot structure holds, and fall back on change.
- Parallel IPM refactor determinism matters for reproducible optimiser trajectories. A nondeterministic factorization can make IPOPT take different iteration paths, which hurts debugging and benchmarking. This is another reason to fix Section 3's reduction order.

### Gaps
- No public numbers were found on the share of factorization time spent in assembly or index work vs dense kernels for MA97, SSIDS or PARDISO refactorization.

---

## Cross-cutting recommendations for smf (synthesis, not sourced claims)

Ranked by expected payoff for a CPU-first, Windows, 16-thread target:

1. **Deterministic parallel assembly.** Use a canonical child order for extend-add, a fixed per-task BLAS thread count, and MKL CNR (`MKL_CBWR`). This is cheap and removes the run-to-run nondeterminism. cuDSS's opt-in deterministic mode shows the industry treats it as a configurable feature.
2. **Intra-front 2D block tasks for the top of the tree (MA87/SSIDS/PaStiX style)** in place of one threaded-BLAS call per top front. Successive thin chain fronts can then overlap, because a child's tile updates feed the parent tiles without a front-level barrier. This addresses the thin-top-chain weakness better than BLAS threading.
3. **APTP-style blocked indefinite kernel (Duff–Hogg–Lopez 2020).** It is BLAS-3 dominated, task-parallel within a front, and as robust as TPP. Reported to beat MA86/MA97/PARDISO on multicore.
4. **Chain amalgamation and fused small-k updates** for chains of thin fronts. Treat a chain as one supernode when fill cost is low; use a left-looking-style update for npiv ≪ nrow.
5. **Small-front batching and microkernels** below 96 (register-blocked LDL^T for n ≤ 32; batch sibling leaves).
6. **Solve-phase:** cache inverted diagonal blocks for small fronts (partitioned inverse) and use multi-RHS blocking in iterative refinement.
7. **GPU:** use cuDSS 1.0 on the RTX 5080 as a reference baseline, with FP64 emulation and FP32-factor + FP64-refinement variants. Do not build a native GPU factorization for weak-FP64 consumer hardware. Revisit if a datacenter GPU target appears.
8. **BLR/HSS:** skip unless large 3D PDE-constrained problems become a target.
