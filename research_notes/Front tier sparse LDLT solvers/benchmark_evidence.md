# Benchmark evidence: sparse symmetric indefinite direct solvers (especially as IPM linear solvers), as of Oct 2026

Scope note: the target is "smf" (C++20 multifrontal LDL^T). Its current status: 1.4x geomean over MA27 on a 20-matrix suite; beats CHOLMOD, MKL PARDISO and MUMPS on 7 IPOPT KKT models at 1 thread; loses to MKL PARDISO on some hard GHS_indef matrices. The notes below collect published evidence on who is front-tier, on which test sets, and how the literature benchmarks. Older classics (pre-2015) are flagged **[classic]**.

Sources actually read in full text: Tasseff et al. 2019 (arXiv 1909.08104), Gould/Hu/Scott 2007 TOMS **[classic]**, Pacaud et al. (arXiv 2405.14236 v3, Jul 2026), Duff/Hogg/Lopez RAL-TR-2018-012 (= SISC 2020 SSIDS/APTP paper), Hogg & Scott 2013 TOMS "Pivoting strategies for tough sparse indefinite systems" **[classic]**, Montoison et al. 2025 MadNCL (arXiv 2510.05885), the Ipopt install docs, and the NVIDIA cuOpt barrier blog. Other items come from search-result summaries and are flagged as such.

---

## 1. Head-to-head comparisons: who wins, by how much, on which problem classes

### Takeaway
No single solver wins everywhere, and the ranking depends on the problem class. (1) On CUTEst and PGLib-OPF (many small/medium, super-sparse KKT systems, tens to hundreds of IPM iterations), the **serial HSL codes MA57 and MA27** are at or near the virtual best, and parallel solvers lose to their overhead. (2) On large PDE-constrained KKT systems, **PARDISO** (Schenk) and **SPRAL SSIDS** are fastest. (3) On tough GHS_indef-style matrices, PARDISO's static-pivoting factorization is the fastest, but at an accuracy cost (larger residuals, iterative refinement, occasional failures). SSIDS (APTP) roughly matches PARDISO's speed with threshold-pivoting robustness. (4) GPU (cuDSS) wins on very large OPF only via condensed or regularized reformulations at relaxed tolerance (1e-6), and is less robust on CUTEst.

### Cited Findings

**Tasseff, Coffrin, Wächter & Laird (2019), "Exploring Benefits of Linear Solver Parallelism on Modern Nonlinear Optimization Applications"**: the reference study for Ipopt linear solvers. [arXiv 1909.08104](https://arxiv.org/pdf/1909.08104)
- 10 solver configurations inside Ipopt: MA27, MA57, HSL_MA77, HSL_MA86, HSL_MA97, MUMPS 4.10.0, PARDISO 6.0.0 (pardiso-project), WSMP 18.06.02, SPRAL CPU, and SPRAL GPU (2016-09-23 source). HSL was the 2014-01-10 Coin-HSL Full release. CPU: 2x Xeon E5-2695 v4 (36 cores). GPU: Tesla K40M. 4-hour wall limit, 9999 iterations, tol 1e-8. The full set took more than 82 days of wall-clock compute. [arXiv 1909.08104](https://arxiv.org/pdf/1909.08104)
- Thread calibration on a 3D boundary-control PDE (N=48: 124,416 variables, 110,592 equalities). Best thread counts: MA86 24, MA97 22, MUMPS 16, PARDISO 22, WSMP 48. The minimum solve times were MA86 352.1 s, MA97 749.2 s, MUMPS 2164.7 s, **PARDISO 65.8 s**, WSMP 110.5 s, so PARDISO was about 5x faster than MA86 and about 11x faster than MA97 on this 3D PDE KKT. MA97 behaved "erratically" at low thread counts, and times rose at higher counts. [arXiv 1909.08104](https://arxiv.org/pdf/1909.08104)
- **CUTEst (all 1369 instances as of May 2019):** "MA57 and MUMPS appear to be the best performers". MA86 and PARDISO were similar but solved fewer instances. MA27 was slower than MA57/MUMPS but eventually solved a similar number. "HSL MA97 and WSMP are generally slower and appear less robust", and SPRAL GPU was the worst. [arXiv 1909.08104](https://arxiv.org/pdf/1909.08104)
- **PGLib-OPF v19.05 (3 to 13,659 buses; AC, QC and SOC formulations; 396 runs per solver):** "MA27 and MA57 closely track the virtual best profile". MA97 was "only somewhat slower", while PARDISO and both SPRAL builds were "typically much slower". [arXiv 1909.08104](https://arxiv.org/pdf/1909.08104)
- **2D PDE (Maurer–Mittelmann scalable problems, N=128..1024, 176 runs; largest has 2.1M variables and 1.05M equalities):** "PARDISO vastly outperforms all other solvers, but for a small number of problems, fails to converge". MA86 solved all 176 instances. MA27 "generally cannot contend" on large problems, and the gap between MA57 and MA27 grows with size (attributed to BLAS). [arXiv 1909.08104](https://arxiv.org/pdf/1909.08104)
- **3D PDE (N=4..64, 64 runs; about 290k variables, denser fill):** PARDISO and both SPRAL builds were fastest and most robust. MA97 performed about like serial MA57 and was "less robust on more challenging problems". MUMPS 4.10 was "very poor", and MA27 was the worst. [arXiv 1909.08104](https://arxiv.org/pdf/1909.08104)
- Recommendation table: Easy problems: best licensed **MA57**, best free **MA27**. Difficult problems: best licensed **PARDISO**, best free **SPRAL**. [arXiv 1909.08104](https://arxiv.org/pdf/1909.08104)
- On parallelism: "the benefits of parallelism are sometimes suspect" for CUTEst/OPF because of the many IPM iterations and per-factorization reinitialisation overhead, but they are clear on PDE problems. [arXiv 1909.08104](https://arxiv.org/pdf/1909.08104)

**Gould, Hu & Scott (2007), TOMS 33(2) [classic]**: the canonical serial comparison. [TOMS PDF](https://numerical.rl.ac.uk/media/people/nick-gould/GoulHuScot07_toms.pdf)
- Test set: 88 positive-definite and 61 indefinite matrices of order >10,000 (5 indefinite ones structurally singular). The "large" subset (n ≥ 50,000) holds 43 definite and 30 indefinite matrices. Everything is from the UF (SuiteSparse) collection, Matrix Market, Harwell-Boeing and contributors. Many indefinite ones became the GHS_indef group. Single 500 MHz Alpha EV6 core, 30-minute CPU limit. [TOMS PDF](https://numerical.rl.ac.uk/media/people/nick-gould/GoulHuScot07_toms.pdf)
- Indefinite results: "Only MA57 and PARDISO had just one failure, while some of the solvers failed on 20% or more of the problems". All solvers failed on SPARSINE. [TOMS PDF](https://numerical.rl.ac.uk/media/people/nick-gould/GoulHuScot07_toms.pdf)
- Factorization: "a significant gap between PARDISO and the remaining solvers" (static pivoting). However, PARDISO's scaled residuals reached O(1e-9), O(1e-7) and O(1e-6) on CONT-300, crystk02 and crystk03; everywhere else they were ≤ O(1e-10). [TOMS PDF](https://numerical.rl.ac.uk/media/people/nick-gould/GoulHuScot07_toms.pdf)
- Solve phase: "MA57 is a clear winner". A perturbed-pivot PARDISO solve is up to 3x slower than MA57 because of up to two extra refinement steps. [TOMS PDF](https://numerical.rl.ac.uk/media/people/nick-gould/GoulHuScot07_toms.pdf)
- Full analyse+factorize+solve: "a clear preference for MA57 and PARDISO... little to choose between the two", followed by BCSLIB-EXT, MUMPS (sym and unsym) and WSMP. For SPD matrices "a slight edge overall for CHOLMOD". [TOMS PDF](https://numerical.rl.ac.uk/media/people/nick-gould/GoulHuScot07_toms.pdf)
- With a small threshold u = 1e-10 (the optimization setting), MA57 failed 3 more problems without refinement and UMFPACK failed 12 more. [TOMS PDF](https://numerical.rl.ac.uk/media/people/nick-gould/GoulHuScot07_toms.pdf)

**Duff, Hogg & Lopez (2018/2020), the SSIDS v2 APTP solver** (RAL-TR-2018-012; SIAM J. Sci. Comput. 42(2):C23–C42): [RAL-TR PDF](https://epubs.stfc.ac.uk/manifestation/40018271/RAL-TR-2018-012.pdf); [SISC](https://epubs.siam.org/doi/abs/10.1137/18M1225963)
- Setup: 28-core Broadwell (2x E5-2690 v4). The "easy indefinite" set has 21 SuiteSparse matrices (no scaling), from t2dal at n=4k up to PARSEC Ga19As19H42 with 9.1 Tflop. The "hard indefinite" set has 24 matrices (TSOPF_FS power-flow, GHS_indef cont-201/300, ncvxqp1/3/5/7, cvxqp3, bratu3d, stokes128, darcy003, d_pretok, turon_m, c-big), using a matching-based ordering and scaling. MKL PARDISO used its default iterative refinement. [RAL-TR PDF](https://epubs.stfc.ac.uk/manifestation/40018271/RAL-TR-2018-012.pdf)
- Easy-set factorization times (s), SSIDS (locality) / MA86 / MA97 / MKL PARDISO: Ga19As19H42 23.89 / 29.30 / 175.36 / 61.63; Ge99H100 21.24 / 24.48 / 141.31 / 38.60; sparsine 7.88 / 9.00 / 43.90 / 12.29; H2O 4.55 / 6.23 / 28.50 / 7.39; pct20stif 0.16 / 1.02 / **18.88** / 0.12. PARDISO is "marginally better for the smallest problems". SSIDS's advantage "grows with the problem size". [RAL-TR PDF](https://epubs.stfc.ac.uk/manifestation/40018271/RAL-TR-2018-012.pdf)
- PARDISO "could not reach a solution" for t2dal and qa8fk (iterative refinement failed to converge) unless the matrix was first scaled and reordered. [RAL-TR PDF](https://epubs.stfc.ac.uk/manifestation/40018271/RAL-TR-2018-012.pdf)
- Hard set: SSIDS "performs favourably compared to HSL MA97", and the gap increases with size. "SSIDS and PARDISO achieve similar performance although PARDISO seems marginally better on smaller problems". PARDISO residuals were worse on some matrices, e.g. one TSOPF case at about 3.9e-3 and about 8.9e-7 on another, against SSIDS at ≤ about 1e-9 (the table text is column-misaligned in extraction, so only the pattern is cited). [RAL-TR PDF](https://epubs.stfc.ac.uk/manifestation/40018271/RAL-TR-2018-012.pdf)
- Abstract claim: favourable vs MA86, MA97 and PARDISO "both in terms of performance on a multicore machine and in terms of numerical robustness". [SISC](https://epubs.siam.org/doi/abs/10.1137/18M1225963)

**Hogg & Scott (2013), "Pivoting strategies for tough sparse indefinite systems", TOMS 40(1) [classic]**: [TOMS](http://dx.doi.org/10.1145/2513109.2513113) (full text read from local copy)
- Defines a "tough" problem as one where threshold partial pivoting (TPP) with METIS and MC64 scaling causes delayed pivots ≥ max(5000, 0.2n). Candidates were all SuiteSparse symmetric indefinite matrices with n ≥ 10,000. This is a useful, citable definition of a "hard GHS_indef" subset.
- Recommendation: "a matching-based ordering should be used" for tough problems; it is built into HSL_MA97 and available via HSL_MC80. Static pivoting with matching ordering removes all delays, but inertia knowledge "may be lost". Matching orderings are *not* recommended when TPP gives no delays, because they cost more and give denser factors.
- Ipopt context: Ipopt uses u = 1e-8 by default for MA27/MA57 and raises u up to 1e-4 when it detects wrong inertia or large backward error.

**Pacaud, Shin, Montoison, Schanen & Anitescu, "Condensed Interior-Point Methods for Scalable NLP on GPUs" (arXiv 2405.14236, v3 Jul 2026)**: [arXiv 2405.14236](https://arxiv.org/pdf/2405.14236)
- Hardware: AMD EPYC 7443 (24 cores) vs NVIDIA A100. Parallel CPU solvers peaked at 8 threads.
- On the condensed SPD matrix of 78484_epigrids (n = 674,562, nnz = 7.34M), factorization took: CHOLMOD 0.744 s (1 thread), Panua PARDISO 0.294 s, MA86 0.138 s, cuDSS 0.040–0.063 s. cuDSS analysis took about 3.6 s, roughly 4x slower than CHOLMOD's 1.37 s. Numerical factorization in cuDSS was "approximately twice as fast as in HSL MA86" (the exact row mapping in the extracted table is ambiguous; ratios as stated in the text). [arXiv 2405.14236](https://arxiv.org/pdf/2405.14236)
- One IPM iteration on the full augmented K2 with **MA27: 1.23 s to factorize**, against **MA57 at 1.9 s** ("increasing the number of threads in the BLAS backend used by HSL MA57 yielded no improvement"). MA86 factorizes the *condensed* system about 10x faster than MA27 factorizes K2, and cuDSS adds about 4x on top. [arXiv 2405.14236](https://arxiv.org/pdf/2405.14236)
- PGLib-OPF (29 cases, tol 1e-6): 78484_epigrids took 191.1 s with MadNLP+MA27 (CPU), 26.0 s with LiftedKKT+cuDSS and 16.2 s with HyKKT+cuDSS. Ipopt+MA27 took 198.5 s. HyKKT reaches "an 8x speedup over HSL MA27" on large cases. On small cases (1354_pegase, 2000_goc) the CPU and GPU totals are within about 0.5–1 s, so there is no GPU win there. [arXiv 2405.14236](https://arxiv.org/pdf/2405.14236)
- "The super-sparse structure explains why HSL MA27 yields better results than HSL MA57 [on OPF]... OPF results may not represent general nonlinear programs." On CUTEst, by contrast, "HSL MA57 has better performance than HSL MA27". [arXiv 2405.14236](https://arxiv.org/pdf/2405.14236)
- CUTEst (413 instances with >1,000 variables and ≥1 constraint): the GPU condensed methods are "less competitive than a standard method running on the CPU" and "less robust". The failure modes are cuDSS symbolic-analysis time, dense Jacobian columns, and degenerate problems (ROCKET needs δ = 1e12). [arXiv 2405.14236](https://arxiv.org/pdf/2405.14236)

**Montoison, Pacaud, Saunders, Shin & Orban (Oct 2025), MadNCL**: cuDSS LDL^T on *indefinite* KKT. [arXiv 2510.05885](https://arxiv.org/html/2510.05885v1)
- Uses cuDSS sparse LDL^T **with static pivoting** (pivot epsilon 1e-10) plus iterative refinement on a stabilized K2r system. cuDSS does not let the user set the sign of pivot perturbations. Hardware: EPYC 7443 vs H100.
- 78484_epigrids: 54.4 s on GPU, about 18x faster than MadNCL on CPU with MA27 and about 6x faster than MadNLP+MA27 (353.9 s). cuDSS factorizes K2r about 17x faster per iteration than MA27. K2r fails on 1 of 11 OPF cases; the condensed K1s variant fails on all 11.
- The authors say GPU reformulations "cannot achieve the same degree of robustness as the classical CPU implementations of IPM solvers", and GPU MadNLP's default tolerance is relaxed to 1e-6.

**Community and vendor claims (lower confidence)**
- MadNLP team (Julia Discourse): MadNLP+ExaModels+cuDSS solves case70k about 18x faster than Ipopt+ExaModels+MA27. Community post, not peer-reviewed. [Discourse](https://discourse.julialang.org/t/examodels-jl-and-madnlp-jl-on-gpus/111377)
- NVIDIA cuOpt barrier (uses cuDSS for the per-iteration sparse solves): tested on 61 large Mittelmann LPs on a GH200 + H200, barrier without crossover, 1-hour limit with failures counted as 1 h. Geomean results: 8.81x faster than an open-source CPU solver; 1.5x *slower* than "commercial solver A" (faster on 31/61; attributed to A's presolve); 2x faster than "commercial B". Solved: cuOpt 55/61, open-source 48/61, A 60/61, B 58/61. The internal benchmark does not say whether it uses normal equations or an augmented LDL^T. cuDSS 0.7 made symbolic factorization about 2.5x faster on optimization matrices. [NVIDIA blog](https://developer.nvidia.com/blog/solve-linear-programs-using-the-gpu-accelerated-barrier-method-in-nvidia-cuopt/)
- Panua PARDISO 8.2 (Feb 2024): the vendor page benchmarks against "Intel MKL Pardiso 2020" on circuit matrices (NXP, up to 16 cores; Silvaco, up to 32 cores), with speedups shown only in plots. The page says "Intel MKL version of Pardiso is based on our version from 2006". [Panua](https://panua.ch/pardiso/)
- A CalculiX forum user claims PARDISO 8 is 4–100x better than MKL, while an Intel forum reply says MKL PARDISO is faster "for some problems". Both are anecdotal and conflicting. [search summary: CalculiX discourse](https://calculix.discourse.group/t/power-comparison-calculix-2-21-2-19-2-2-18-to-code-aster-17-0/3932); [Intel community](https://community.intel.com/t5/Intel-oneAPI-Math-Kernel-Library/PARDISO-New-Version-4/m-p/851639)
- Apple Accelerate (Eigen merge request, M1, about 1.95M unknowns, not a KKT matrix): Accelerate LDL^T took about 10.8 s total, CHOLMOD simplicial LDL^T about 22 s, and Eigen LDLT about 47 s. Accelerate's default LDLT is a threshold-partial-pivoting variant. (From search summary.) [Eigen MR 856](https://gitlab.com/libeigen/eigen/-/merge_requests/856)

### Inferences
- **Front tier is class-dependent.** For IPOPT-style KKT on CUTEst/OPF, the bar is serial MA57 (general) and MA27 (super-sparse OPF). For large or 3D-PDE KKT and dense-fill problems, the bar is PARDISO (Panua or MKL) and SSIDS. For tough GHS_indef, the speed bar is PARDISO's static-pivoting factorization and the speed-plus-robustness bar is SSIDS APTP.
- smf already beats MA27 (1.4x geomean) and CHOLMOD/MKL/MUMPS at 1 thread on 7 KKT models. That is encouraging, but **MA57 is the missing key comparator**: it is the IPOPT community's de facto licensed default and wins CUTEst in Tasseff. Note that on OPF, MA27 beat MA57 (1.23 s vs 1.9 s per factorization on 78k-bus K2), so a 1.4x win over MA27 on OPF-like KKT is meaningful.
- Losing to MKL PARDISO on hard GHS_indef in *factorization time* is the expected outcome for any solver doing threshold pivoting with delays; every published study shows this. A fair comparison should report (a) the backward error after PARDISO's default refinement, (b) solve time *including* refinement, and (c) failures. PARDISO lost on these in Gould et al. 2007 (residuals up to 1e-6, solves up to 3x slower) and in Duff et al. 2018 (2 failures without scaling, residual about 4e-3).
- The literature shows no peer-reviewed head-to-head of MA97 vs MA57 *inside Ipopt* newer than Tasseff 2019. That study used 2014 Coin-HSL, MUMPS 4.10 and SPRAL 2016, so the solver versions are old. A modern 2026 rerun (MA57/MA86/MA97 from HSL 2023+, SPRAL current, Panua 8.x, MKL 2025, MUMPS 5.7) would itself be novel evidence.

### Gaps
- No peer-reviewed 2023–2026 benchmark found that compares **Panua PARDISO 7/8 vs MKL PARDISO** on symmetric indefinite or KKT matrices; only the vendor plots (circuit matrices) and anecdotes exist. The Bollhöfer/Schenk et al. chapter "State-of-the-Art Sparse Direct Solvers" ([arXiv 1907.05309](https://arxiv.org/abs/1907.05309)) was located but not read.
- No published comparison found of **STRUMPACK, QDLDL or Apple Accelerate** against HSL/PARDISO on KKT matrices. QDLDL (OSQP/Clarabel) is a non-pivoting LDL^T, used with regularisation and quasi-definite structure.
- No paper found isolating **cuDSS LDL^T on unregularized augmented KKT** vs CPU HSL with dynamic pivoting. Published GPU wins rely on condensation or static-pivot regularisation at tol 1e-6.
- The Hogg & Scott 2013 *Algorithms* paper "New Parallel Sparse Direct Solvers for Multicore Architectures" (MA86/MA97 vs others) was located ([doi:10.3390/a6040702](https://doi.org/10.3390/a6040702)) but not read, so no numbers are given.
- The "SPRAL GPU worst" result in Tasseff used a 2013-era K40M, so modern GPUs could change it.

---

## 2. Which HSL solver is IPOPT's de facto best, and the known failure modes

### Takeaway
In practice, **MA57** is the de facto best licensed choice for general Ipopt use (CUTEst-like problems), with **MA27** as the free fallback and often the fastest on super-sparse OPF. HSL officially recommends **HSL_MA97 for general use**, but independent Ipopt benchmarks found MA97 slower and less robust than MA57 on CUTEst and 3D PDEs. MA86 and MA97 only pay off on large factorizations. MA57's BLAS-level parallelism does not help on super-sparse KKT systems.

### Cited Findings
- Ipopt docs: "MA57 can be considerably faster than MA27 on some problems". They say the linear solver choice "impacts Ipopt's speed and robustness" and recommend trying several. The Coin-HSL Archive (MA27, MA28, MC19) is free for personal commercial or non-commercial use. Coin-HSL Full (MA57, MA77, MA86, MA97) is "freely available for academic use only". The docs report MUMPS 5.2.x performing worse than MUMPS 4.10.0 on average on CUTEst. [Ipopt INSTALL](https://www.coin-or.org/Ipopt/INSTALL.html)
- HSL's own guidance (per STFC licence listing, via search summary; the page returned 403 when fetched directly): HSL_MA97 for general use, MA57 for small or highly sparse problems, HSL_MA86 for huge problems whose factors fit in memory, otherwise HSL_MA77. [search summary of STFC Coin-HSL page](https://licences.stfc.ac.uk/product/coin-hsl)
- An HSL developer on the 2012 Ipopt mailing list (via search summary; the original archive host was unreachable) said MA97 was almost comparable to MA57 except on the smallest problems and faster on larger ones. MA86 was considerably slower than MA57 on small problems, and parallel runs may not be bit-reproducible, so they can reach different local optima. [Ipopt list (2012)](https://list.coin-or.org/pipermail/ipopt/2012-February/002731.html)
- Tasseff et al. 2019: MA57 was best licensed on easy problems (CUTEst, OPF). MA97 was "generally slower and... less robust" on CUTEst and "less robust on more challenging problems" in 3D PDEs. MA97 behaved erratically with thread count. MA86 offers only one ordering (MC68, no matching), which "may result in more delays" on hard indefinite systems. If many pivots are delayed, MA86's DAG parallelism degrades. [arXiv 1909.08104](https://arxiv.org/pdf/1909.08104)
- MA27 failure mode: no BLAS-3, so it "generally cannot contend" on large 2D/3D PDE KKT systems, while the gap to MA57 grows with problem size. [arXiv 1909.08104](https://arxiv.org/pdf/1909.08104)
- MA57 failure mode on super-sparse KKT: 1.9 s vs MA27's 1.23 s per factorization on the 78k-bus OPF K2 system, with no gain from more BLAS threads. [arXiv 2405.14236](https://arxiv.org/pdf/2405.14236). Likewise, a 2023 GPU paper reports MA57 is not faster than MA27 despite multithreaded BLAS (search summary). [arXiv 2301.04869](https://arxiv.org/pdf/2301.04869)
- MA97 failure mode on easy indefinite matrices without scaling: pct20stif took 18.88 s against 0.16 s for SSIDS, and MA97 was 4–7x slower than SSIDS on large PARSEC matrices. [RAL-TR-2018-012](https://epubs.stfc.ac.uk/manifestation/40018271/RAL-TR-2018-012.pdf)
- MA97's design goal is bit-compatibility: the results are identical regardless of thread count. [search summary citing Ipopt/GAMS docs and arXiv 1909.08104](https://gams.com/53/docs/S_IPOPT.html)
- Within Ipopt, MA27/MA57 default to pivot tolerance u = 1e-8 and increase u up to 1e-4 on wrong inertia or large backward error. [Hogg & Scott 2013 TOMS](http://dx.doi.org/10.1145/2513109.2513113)
- GAMS Ipopt docs: MA57 and HSL_MA97 often outperform MA27 on larger instances, and PARDISO often beats MUMPS and is similar to the HSL solvers (search summary). [GAMS IPOPT docs](https://gams.com/53/docs/S_IPOPT.html)

### Inferences
- For smf's credibility with the IPOPT community, the headline comparisons should be **MA57 (default u=1e-8 Ipopt-style settings) and MA27 at 1 thread**, then **MA97/SSIDS/PARDISO at N threads** on larger KKT systems.
- The documented HSL failure modes make good test targets for smf: MA27's lack of BLAS-3 on large fill; MA57 BLAS overhead on super-sparse fronts; MA97 delayed-pivot blowup without scaling; and MA86's lack of a matching ordering.

### Gaps
- No quantitative failure counts per HSL solver were extracted from Tasseff (the results are only in profile plots).
- No primary 2020+ source was found ranking MA57 vs MA97 inside Ipopt on modern hardware.

---

## 3. Standard benchmark sets and typical sizes

### Takeaway
The literature uses two families: (a) **raw matrix sets** (SuiteSparse incl. GHS_indef, TSOPF, PARSEC, Boeing, Oberwolfach; the Gould–Hu–Scott n>10k set; the Hogg–Scott "tough" set), and (b) **optimization-driven KKT streams** (CUTEst, PGLib-OPF, COPS, the Maurer–Mittelmann scalable PDE problems shipped with Ipopt, Maros–Meszaros QP, Mittelmann LP). Typical IPM KKT sizes run from about 1e4 to 1e6 rows. The OPF and CUTEst matrices are easy to factor, while PDE (especially 3D) and dense-fill problems separate the solvers.

### Cited Findings
- **SuiteSparse GHS_indef and others (raw-matrix set):** Gould et al. used all real symmetric matrices with n > 10,000 available in June 2003: 61 indefinite and 88 SPD, now in SuiteSparse. [TOMS 2007](https://numerical.rl.ac.uk/media/people/nick-gould/GoulHuScot07_toms.pdf)
- **Easy and hard indefinite sets:** 21 and 24 matrices (listed in §1) that span n = 4k–1M, nnz(L) = 0.28M–823M and 0.02–9100 GFlop. [RAL-TR-2018-012](https://epubs.stfc.ac.uk/manifestation/40018271/RAL-TR-2018-012.pdf)
- **The "tough" criterion:** delayed pivots ≥ max(5000, 0.2n) under METIS+MC64 TPP. [Hogg & Scott 2013](http://dx.doi.org/10.1145/2513109.2513113)
- **CUTEst:** 1369 instances (May 2019), but "skewed toward easier instances" and many have too few degrees of freedom. Pacaud et al. filter to >1,000 variables and ≥1 constraint, leaving 413 instances. [arXiv 1909.08104](https://arxiv.org/pdf/1909.08104); [arXiv 2405.14236](https://arxiv.org/pdf/2405.14236)
- **PGLib-OPF:** v19.05 covers 3–13,659 buses. Recent GPU work uses up to 78484_epigrids, which has 674,562 variables and 1,039,062 constraints. The 30000_goc case (208k variables) is notoriously iteration-heavy (150–230 IPM iterations). [arXiv 1909.08104](https://arxiv.org/pdf/1909.08104); [arXiv 2405.14236](https://arxiv.org/pdf/2405.14236)
- **Ipopt ScalableProblems (Maurer–Mittelmann):** these are 2D boundary/distributed control problems. The largest, MDistCntrl6a at N=1024, has 2.1M variables, 1.05M equalities, 6.3M Jacobian nonzeros and 3.1M Hessian nonzeros. The 3D boundary control problem at N=64 has about 290k variables and up to 7.1M Jacobian nonzeros with heavy fill. [arXiv 1909.08104](https://arxiv.org/pdf/1909.08104)
- **COPS** (rocket, robot, pinene, dirichlet, ...) is used for GPU NLP benchmarks. Rocket is a known degenerate case (Goddard singular arc). [arXiv 2405.14236](https://arxiv.org/pdf/2405.14236)
- **Maros–Meszaros:** 138 convex QPs (qpbenchmark), used for QP solvers such as Clarabel and OSQP. [qpbenchmark](https://github.com/qpsolvers/maros_meszaros_qpbenchmark)
- **Mittelmann LP set:** 61 large LPs, about a dozen with more than 1M rows and columns. [NVIDIA cuOpt blog](https://developer.nvidia.com/blog/solve-linear-programs-using-the-gpu-accelerated-barrier-method-in-nvidia-cuopt/)
- Tasseff et al. conclude that "conventional nonlinear programming benchmarking suites... do not result in matrices that are particularly large or difficult to factorize" and call for larger benchmarks. [arXiv 1909.08104](https://arxiv.org/pdf/1909.08104)

### Inferences
- A credible smf suite has three tiers: (1) **KKT streams** dumped from Ipopt on CUTEst (filtered), PGLib-OPF (up to 78484_epigrids), COPS and the Ipopt ScalableProblems (2D and 3D); (2) **SuiteSparse GHS_indef plus the Duff–Hogg–Lopez easy and hard lists** for raw factorization; (3) a **tough subset** defined by the Hogg–Scott delayed-pivot criterion. Reusing published lists lets smf be placed directly next to published MA86/MA97/PARDISO/SSIDS timings.
- smf's current 20-matrix and 7-model suites are small relative to the literature, where 21+24 matrices is the minimum and 149 or 413 instances are common.

### Gaps
- Mittelmann's NLP/QP benchmark pages were not fetched (TLS certificate rejected for plato.asu.edu), so their current instance counts are unverified.
- No canonical public "KKT matrix dump" collection from Ipopt runs was found. Studies regenerate matrices from the models.

---

## 4. Methodology: profiles vs shifted geomeans, accuracy gates, cold vs warm factorization, threads, over-tuning

### Takeaway
Best practice combines (a) Dolan–Moré performance profiles **or** cumulative "solved vs time" profiles with virtual best and worst lines, together with (b) a shifted geometric mean, (c) an explicit backward-error gate with a fixed refinement policy, (d) default parameters for every solver except documented, justified changes, (e) separate reporting of analyse, factorize and solve phases plus full IPM wall time, and (f) calibrated or fixed thread counts with 1-thread results reported separately.

### Cited Findings
- **Performance profiles:** p_i(τ) is the fraction of problems on which solver i is within τ of the best. Gould et al. apply them to the analyse, factorize and solve times, nnz(L) and memory. [TOMS 2007](https://numerical.rl.ac.uk/media/people/nick-gould/GoulHuScot07_toms.pdf); original [Dolan & Moré](https://arxiv.org/pdf/cs/0102001)
- **Caveat:** with more than two solvers, a profile does not rank the non-best solvers against each other, and removing the best solver can reorder the rest. Gould & Scott (TOMS 43(2), 2016) recommend caution and a series of profiles (search summary). [Gould & Scott 2016](https://centaur.reading.ac.uk/74694); [Nested performance profiles](https://arxiv.org/pdf/1809.06270)
- Tasseff et al. avoid Dolan–Moré profiles because they "can be misleading when comparing more than two profiles". They instead plot the cumulative number solved vs absolute time, with "virtual best" and "virtual worst" curves. [arXiv 1909.08104](https://arxiv.org/pdf/1909.08104)
- **Shifted geometric mean:** Mittelmann's slides (via search summary; PDF not fetched due to a TLS error) say commercial developers prefer the SGM to multi-solver profiles, with a shift of about 10 s for MIP/LP-scale times. The CuClarabel paper uses shift k = 1. [Mittelmann 2026 talk](https://plato.asu.edu/talks/hongkong26.pdf); [CuClarabel](https://arxiv.org/pdf/2412.19027)
- **Accuracy gate (Gould et al.):** set b = A·1 (exact solution of all ones). Allow at most one step of iterative refinement unless the solver refines by default (PARDISO does up to 2 when pivots are perturbed). The scaled residual ||b−Ax||/(||A||·||x||+||b||) must be ≤ 1e-4 to count as solved; this is deliberately loose, to flag gross failures. The forward error was checked but only warned on. [TOMS 2007](https://numerical.rl.ac.uk/media/people/nick-gould/GoulHuScot07_toms.pdf)
- Duff–Hogg–Lopez report the scaled residual ||Ax−b||₂/(||A||₁||x||₂+||b||₂) alongside the times and let PARDISO run its default refinement. [RAL-TR-2018-012](https://epubs.stfc.ac.uk/manifestation/40018271/RAL-TR-2018-012.pdf)
- **Defaults and over-tuning:** Gould et al. "decided early on... not to try and fine tune the input parameters for each solver on each problem". The only exception was the pivot threshold u, run at both the default and 1e-10. Tasseff likewise left solvers "mostly at their defaults, intended to capture the behavior of typical users", except for documented SPRAL and PARDISO settings. [TOMS 2007](https://numerical.rl.ac.uk/media/people/nick-gould/GoulHuScot07_toms.pdf); [arXiv 1909.08104](https://arxiv.org/pdf/1909.08104)
- **Threads:** Tasseff calibrated OMP_NUM_THREADS on one representative problem (2–72 threads, 5 repeats, mean), then fixed it. The best was 16–24 threads on a 36-core, 2-socket node, not the maximum, because of inter-socket overhead. Pacaud et al. found 8 threads best for CPU solvers on their 24-core EPYC. [arXiv 1909.08104](https://arxiv.org/pdf/1909.08104); [arXiv 2405.14236](https://arxiv.org/pdf/2405.14236)
- **Cold vs warm:** symbolic analysis is done once and reused across IPM iterations with the same sparsity. cuDSS analysis is about 4x slower than CHOLMOD's but amortised. Pacaud et al. report "init" (including the initial symbolic factorization) separately from per-iteration linear-solver time and exclude JIT compile time. Gould et al. note that for many refactorizations of a fixed pattern, WSMP, PARDISO and CHOLMOD were strongest, while for many solves per factor, BCSLIB-EXT, MA57 and PARDISO were. [arXiv 2405.14236](https://arxiv.org/pdf/2405.14236); [TOMS 2007](https://numerical.rl.ac.uk/media/people/nick-gould/GoulHuScot07_toms.pdf)
- **IPM-level metrics:** report the iteration count alongside time, because a solver that returns a slightly different direction (static pivoting, regularisation) changes IPM iteration counts (e.g. 30000_goc took 150 iterations with Ipopt+MA27 vs 223 with HyKKT+Pardiso). Failures should be counted at the time limit, as cuOpt counts them at 1 h. [arXiv 2405.14236](https://arxiv.org/pdf/2405.14236); [NVIDIA cuOpt blog](https://developer.nvidia.com/blog/solve-linear-programs-using-the-gpu-accelerated-barrier-method-in-nvidia-cuopt/)
- **Variance:** use repeated runs (Tasseff used 5) and timeouts (30 min in Gould et al.; 4 h in Tasseff). [arXiv 1909.08104](https://arxiv.org/pdf/1909.08104)

### Inferences
- For smf: report the **SGM (shift about 1 ms–10 ms for per-factorization times, or about 1 s for full IPM solves)** plus profiles or cumulative-solved plots with virtual best. Gate on scaled backward error ≤ 1e-10, or ≤ 1e-4 as the Gould "gross failure" gate, after a fixed refinement budget applied identically to all solvers. Report correct inertia as a separate pass/fail for IPM use. Report analyse, first factorize, refactorize (warm) and solve separately. Give 1-thread and calibrated N-thread tables separately.
- Avoid over-tuning: freeze smf parameters before running the final suite and use solver defaults (Ipopt-style u for HSL). Publish any non-default settings, and hold out a test subset that smf was not tuned on, given smf's current tuning on a 20-matrix suite.

### Gaps
- No primary source found specifying a standard SGM shift for *linear-solver* (sub-second) timings. Choosing the shift is a judgment call that should be disclosed.

---

## 5. Licensing and availability of competitors

### Takeaway
Everything needed for a credible comparison is obtainable at no cost for academic use. MA27 is free outright (Coin-HSL Archive); MA57/MA77/MA86/MA97 are free for academics (Coin-HSL Full / HSL academic licence); SPRAL SSIDS is open source; MKL PARDISO and cuDSS are free downloads; Panua PARDISO needs a licence from Panua (an academic/evaluation licence historically; current terms unverified); MUMPS is open source.

### Cited Findings
- Coin-HSL Archive (MA27, MA28, MC19): "freely available for personal commercial or non-commercial usage", but no redistribution without a licence. Coin-HSL Full (adds MA57, HSL_MA77, HSL_MA86, HSL_MA97): "freely available for academic use only". Prebuilt libHSL and Coin-HSL binaries are available for macOS and Windows, and they are runtime-loadable by Ipopt. [Ipopt INSTALL](https://www.coin-or.org/Ipopt/INSTALL.html)
- HSL academic licence: perpetual and free for academic staff and students for teaching and research. The commercial named-user licence is about £5,800–£6,200 ex-VAT (search summary of STFC listings). [HSL 2023](https://www.hsl.rl.ac.uk/hsl2023/); [STFC libHSL](https://licences.stfc.ac.uk/product/libhsl)
- Pardiso (pardiso-project / Panua): per the Ipopt docs, "a limited-time academic or evaluation license" is available, and non-profit and commercial licences can be bought. From Ipopt 3.14, `pardiso` means the Panua version and `pardisomkl` means MKL's. Licences cover Linux, Windows and Mac (Intel and Arm) via a Panua account. [Ipopt INSTALL](https://www.coin-or.org/Ipopt/INSTALL.html); [Panua](https://panua.ch/pardiso/)
- MKL PARDISO: ships with oneMKL. Ipopt selects it with `pardisomkl`, and ILP64 MKL is needed for 64-bit integers. [Ipopt INSTALL](https://www.coin-or.org/Ipopt/INSTALL.html)
- SPRAL (SSIDS): download and build separately. Ipopt recommends "SPRAL together with linear algebra routines from the Intel MKL" for best performance, and GPU builds are supported. Tasseff et al. list SPRAL as the best *free* solver for difficult problems. The BSD licence is from general knowledge; the licence text was not fetched, and Tasseff's licence table was garbled in extraction. [Ipopt INSTALL](https://www.coin-or.org/Ipopt/INSTALL.html); [arXiv 1909.08104](https://arxiv.org/pdf/1909.08104)
- MUMPS: described as public domain/open source in the Ipopt docs. [Ipopt INSTALL](https://www.coin-or.org/Ipopt/INSTALL.html)
- WSMP: IBM trial licence (90-day) for non-profit academic benchmarking. [Ipopt INSTALL](https://www.coin-or.org/Ipopt/INSTALL.html)
- cuDSS: the docs list general, symmetric and SPD real/complex matrices, "Numerical pivoting controls", hybrid host/device memory, MG and MGMN execution, and a multithreaded layer. The latest release-notes entry is v1.0.0. The licence is an NVIDIA Software License Agreement with some open-source-licensed portions, and the download is at developer.nvidia.com/cudss-downloads. The terms were not read. [cuDSS docs](https://docs.nvidia.com/cuda/cudss/latest/index.html)
- In practice, cuDSS LDL^T for indefinite KKT uses static pivoting with a pivot epsilon. The user cannot control the sign of perturbations, so inertia is not directly controlled. [arXiv 2510.05885](https://arxiv.org/html/2510.05885v1)

### Inferences
- smf (academic, MSc context) can legally benchmark against MA27/MA57/MA86/MA97 (Coin-HSL Full academic), SSIDS (open source), MKL PARDISO (free), MUMPS (open source), CHOLMOD (open source) and cuDSS (free download). Panua PARDISO needs a licence request from Panua. Publishing results is normally fine, but check each licence's benchmarking-publication clause, especially NVIDIA's SLA and Panua's.

### Gaps
- Panua's current academic licence terms (free or not, duration) could not be verified; the panua.ch page shows no terms.
- cuDSS licence terms (redistribution, benchmark publication) were not read.
- STRUMPACK (BSD), QDLDL (Apache-2.0) and Apple Accelerate (bundled with macOS) licences are from general knowledge only; no source was fetched, so treat them as unverified.
