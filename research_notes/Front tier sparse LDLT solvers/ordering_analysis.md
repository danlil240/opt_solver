# Fast symbolic analysis and fill-reducing ordering for sparse symmetric direct solvers (state of the art, Oct 2026)

Scope: what a C++20 multifrontal LDL^T code ("smf", MA97-like, inside IPOPT) can adopt to make its analyse phase (clean -> AMD / METIS_NodeND / matching-compressed AMD -> etree -> GNP column counts -> relaxed amalgamation -> assembly tree) front-tier fast without worse fill. Measured problems: (a) n <= 4000, analyse slower than MA27's, AMD is 40-50% of analyse; (b) large SPD (bmwcra_1, pwtk) where serial METIS dominates and MKL PARDISO orders in parallel.

Labelling: items marked **[classic]** are pre-2015. Items marked **[background, not re-verified this session]** come from the researcher's prior knowledge and have no fetched source; treat them as leads to check.

## Q1. AMD variants, speedups, dense rows, and how MA57 / MA97 / PARDISO / CHOLMOD / MUMPS choose AMD vs nested dissection

### Takeaway
Serial AMD itself has not changed meaningfully in a decade (SuiteSparse AMD 3.3.4, July 2025, still the 1996 algorithm with aggressive absorption and a 10*sqrt(n) dense threshold). The only new algorithmic result is ParAMD (2025): 3.2-7.3x on 64 threads for n >= 72K, at about 1.1x fill and slower than serial AMD on 1 thread, so it does not help the n <= 4000 case. Production "auto" rules (MA57, MA97, CHOLMOD, MUMPS) all run AMD first or pick on n/density, and only call METIS when the AMD-predicted fill is large.

### Cited Findings
**SuiteSparse AMD (current)**
- The current header is AMD 3.3.4, dated July 25, 2025. Defaults: `AMD_DENSE = 10`, so rows/columns with more than 10*sqrt(n) entries are ignored (removed and ordered last); rows with 16 or fewer off-diagonal entries are never treated as dense. `AMD_AGGRESSIVE` defaults to on (aggressive absorption). Workspace is "1.2 * nnz(A+A^T) + 9n", at most 2.4 nz + 9n — [SuiteSparse amd.h](https://raw.githubusercontent.com/DrTimothyAldenDavis/SuiteSparse/dev/AMD/Include/amd.h)
- The AMD 3.3.x change notes in the SuiteSparse 7.5-7.12 releases list no AMD performance changes. 3.3.0 was "minor change for CHOLMOD 5.1.0 tests", and later bumps carry no notes. No AMD speedups are documented after 2015 — [SUSE package notes](https://packagehub.suse.com/packages/suitesparse/7_10_3-bp160_1_2/), [newreleases SuiteSparse 7.12.3](https://newreleases.io/project/github/DrTimothyAldenDavis/SuiteSparse/release/v7.12.3.paru), [MacPorts](https://ports.macports.org/port/SuiteSparse_AMD/)
- **[classic, 1996-97]** The original Netlib AMD distribution: AMDBAR is "nearly always superior to, or at least comparable with" the other five variants (exact-degree AMDEXA, AMDHAT) in quality and run time. AMDPRE removes dense nodes first and "can be a lot faster than AMDBAR, for matrices with dense rows and columns". Aggressive absorption was then in Harwell MC47B/BD, not in the Netlib AMD routines, and "can improve ordering quality and reduce ordering time" — [Netlib AMD readme](https://www.netlib.org/linalg/amd/readme)

**ParAMD (parallel AMD; Chang, Buluc, Demmel, arXiv 2504.17097, 2025)**
- **Method:** eliminate several pivots per step from *distance-2 independent sets* (Luby-style, single iteration), so pivot neighbourhoods do not overlap. Candidates are relaxed to approximate degree <= mult x min-degree (default mult = 1.1). Each thread keeps its own degree lists with an affinity array for lazy invalidation. The CSC structure is pre-allocated with 1.5x elbow room, which removes garbage collection; extra memory "never exceeded 1.5x" E^0. Element absorption, mass elimination, indistinguishable nodes and external degrees are kept — [arXiv 2504.17097v2](https://arxiv.org/html/2504.17097v2)
- **64-thread speedup vs SuiteSparse AMD** (fill ratio vs serial AMD in brackets): nd24k (72K rows, 28.7M nnz) 3.18x [1.02]; ldoor (952K) 4.39x [1.04]; Flan_1565 (1.56M) 5.31x [1.06]; Serena 5.06x [1.19]; dielFilterV3real 5.59x [1.16]; Bump_2911 4.76x [1.15]; Cube5317k 4.99x [1.04]; Queen_4147 5.61x [1.10]; nlpkkt240 (28M rows, 761M nnz) 7.29x [1.14], with time going from 393 s to 56 s. Fill ratios range from 1.01 to 1.19 — [arXiv 2504.17097v2](https://arxiv.org/html/2504.17097v2)
- **Limits:** a 1-thread run is *slower* than SuiteSparse because of the extra independent-set selection. The |A|+|A^T| preprocessing scales poorly (contention, irregular access) and can be skipped for symmetric input. Average independent-set size at mult = 1.0 vs 1.1: nd24k 2.2 vs 9.0; Flan_1565 42 vs 448; nlpkkt240 57.5 vs 4084. The smallest test matrix is 72K rows — [arXiv 2504.17097v2](https://arxiv.org/html/2504.17097v2)
- **End-to-end on cuDSS/A100 (order + solve):** Flan_1565 4.85 + 18.92 s (SuiteSparse AMD) vs 0.95 + 19.62 s (ParAMD) vs 2.41 + 9.21 s (cuDSS multithreaded ND); Cube5317k 13.94 + 43.90 vs 2.90 + 44.33 vs 7.43 + 33.45 s. ND gives 4.02e9 vs 6.54e9 fill on Cube5317k, so ND still wins end-to-end for large 3D problems — [arXiv 2504.17097v2](https://arxiv.org/html/2504.17097v2)
- Code: [github.com/PASSIONLab/ParAMD](https://github.com/PASSIONLab/ParAMD) (shared-memory OpenMP, no MPI) — [arXiv 2504.17097v2](https://arxiv.org/html/2504.17097v2)

**Automatic AMD-vs-ND choice in production codes**
- **[classic, RAL-TR-2006-001, Duff & Scott]** MA57 3.0.1, positive-definite case: choose METIS if n >= 50,000. Otherwise choose QAMD if (n <= 30,000 and avnum <= 100) or (30,000 < n < 50,000 and avnum <= 46), where avnum = average entries per row; else METIS — [RAL-TR-2006-001](https://numerical.rl.ac.uk/media/reports/dsRAL2006001.pdf)
- **[classic]** MA57 indefinite rule (Fig. 3.7). Parameters: N = 1e5, c1 = 3, c2 = 1.8, c3 = 10.
  - n > N: QAMD if nzl < 3n, else METIS.
  - "oxo" matrices (zero (1,1) and (2,2) blocks): QAMD if m1 > 1.8 m2, else METIS.
  - Otherwise run QAMD first. If its predicted fill p1 > 10 nzl, run METIS and keep it only if p2 < p1.
  - Results: METIS was called for 36 of 63 problems and rejected in 3 — [RAL-TR-2006-001](https://numerical.rl.ac.uk/media/reports/dsRAL2006001.pdf)
- **[classic]** CHOLMOD rule as described there: run AMD first. If lnz_AMD > 5 nzl and flops > 500 lnz_AMD, also run METIS and keep the sparser one. Inside MA57 this ran METIS on 57/94 problems and rejected it in 7. The authors found "little to choose" between the MA57 and CHOLMOD rules. "Always both, pick lower predicted fill" gave the fastest factorize/solve but costs analyse time — [RAL-TR-2006-001](https://numerical.rl.ac.uk/media/reports/dsRAL2006001.pdf)
- **[classic]** Analyse-cost evidence from the same report, MA57 analyse time QAMD vs METIS (s): bmwcra_1 0.25 vs 0.90; bodyy4 0.78 vs 4.04; OBSTCLAE 0.62 vs 4.18. For bodyy4 and OBSTCLAE, METIS analyse can exceed the factorization time. Some large problems order better with QAMD: Fcondp2 (n = 201,822), M_T1, Srb1 and X104 have METIS factors at least 5% denser. (The PDF table layout is garbled; the factor-size columns could not be paired reliably.) — [RAL-TR-2006-001](https://numerical.rl.ac.uk/media/reports/dsRAL2006001.pdf)
- **[classic]** MUMPS default (circa 2005), on a single processor: an AMD variant (QAMD) for n < 10^4 and for larger, very sparse matrices with quasi-dense rows; otherwise METIS_NodeND — [RAL-TR-2006-001](https://numerical.rl.ac.uk/media/reports/dsRAL2006001.pdf)
- **HSL_MA97 v2.8.1 (Nov 2023).** `ordering` values: 1 = AMD; 2 = MD; 3 = METIS; 4 = MA47; 5 (default) = heuristic AMD/METIS choice "assuming the factorization is to be run in parallel"; 6 = same, assuming serial; 7 = HSL_MC80 matching-based ordering with AMD on the compressed matrix; 8 = same with METIS. MA97 uses HSL_MC68 for orderings and HSL_MC78 for the assembly tree. It supports METIS 4 only. It recommends METIS or an ND order for parallel runs — [HSL_MA97 spec](https://www.hsl.rl.ac.uk/specs/hsl_ma97.pdf)
- HSL_MA57 records the ordering actually used (`ainfo%ordering`, info(36)). The changelog notes METIS 4 / METIS 5 incompatibility issues — [HSL_MA57 ChangeLog](https://www.hsl.rl.ac.uk/changes/ChangeLog.hsl_ma57)
- In circuit matrices, PARDISO's METIS ND "generates more fill-ins than minimum degree for most benchmarks", so ND is not universally better for very sparse, irregular graphs — [CKTSO, arXiv 2411.14082](https://arxiv.org/pdf/2411.14082)
- Ceres Solver's ND patch notes that AMD remains the better default on most of its (bundle-adjustment) problems — [Ceres commit](https://ceres-solver.googlesource.com/ceres-solver/+/39ec5e8f99c595226ab3cd760d4945e61c31fc57%5E%21/)

### Inferences
- For n <= 4000, ParAMD is the wrong tool: it is slower on one thread and its benefit only appears at much larger n. The small-n AMD gap against MA27 is more likely constant-factor overhead than algorithmic:
  - building |A|+|A^T|, `amd_valid`/`amd_preprocess` checks, and allocating 1.2 nnz + 9n workspace;
  - the dense-row scan;
  - cold-cache first touch.
  
  Cheaper options: pass an already symmetric, sorted, deduplicated pattern straight into `amd_2`; reuse persistent workspace across analyse calls; skip `amd_aat`; turn off dense-row detection when max degree <= 16. (Inference from the amd.h workspace formula and the ParAMD note that the A+A^T step is costly.)
- **[background, not re-verified this session]** MA27's ordering is an MD on a quotient graph with supervariables and no approximate degrees. On small, very sparse KKT matrices its cost is comparable to AMD; MA27 also skips a separate etree/colcount pass because its analyse produces the assembly tree directly from the elimination.
  - Mimicking that would mean emitting the elimination tree and column counts directly from the AMD elimination: AMD's element absorption already defines the assembly tree, and AMD's `Len`/`Nv` arrays give supernode sizes. That would remove the separate etree + GNP pass.
  - This is what MA27/MA57 do, and it is a plausible win for small n.
- A cheap rule for smf modelled on MA57/CHOLMOD: always run AMD; run METIS only if lnz_AMD > 5 nnz(tril A) and flops > 500 lnz_AMD (CHOLMOD) or p1 > 10 nzl (MA57 indefinite). Large SPD matrices like bmwcra_1/pwtk will always take METIS under these rules, so the large-n fix must be parallel ND (Q2).

### Gaps
- No source found for AMF (approximate minimum fill, Rothberg-Eisenstat / Ng-Raghavan, 1990s classics) with modern 2015+ timing comparisons against AMD; there is no evidence it is used in any current production code.
- No numeric MMD-vs-AMD timing from a 2015+ source; ParAMD does not benchmark MMD.
- The exact heuristic behind MA97 `ordering=5/6` (thresholds) is not given in the spec sheet. MUMPS's current (5.x/6.x) automatic rule was not retrieved.
- PARDISO's choice between ND and MD (iparm(2)=0 is MMD) is user-driven; no automatic switch is documented.

## Q2. Parallel nested dissection (ParMETIS, PT-Scotch, mt-METIS, KaHIP/KaMinPar, METIS 5.2 options, MKL PARDISO, MUMPS)

### Takeaway
In shared memory, the practical options are:
- MKL PARDISO's internal parallel ND, the default since oneMKL 2026.0: about 2x faster than serial METIS on a full node, geomean 1.7x vs 2024.0;
- mt-METIS / multithreaded METIS (as used in cuDSS);
- ParMETIS / PT-Scotch, which need MPI.

Domain-specific ND with coarse separators (patch-based, 2026) gets 4.6x geomean over METIS at a few percent more fill. In METIS_NodeND itself, the cheap levers are `nseps`, `compress` and `pfactor`.

### Cited Findings
**MKL PARDISO**
- `iparm(2)=2` is serial METIS ND; `iparm(2)=3` is the "parallel (OpenMP) version of the nested dissection algorithm", which can cut run time "especially when Phase 1 (reordering) takes significant time". The older docs say `iparm(2)=3` disables CNR mode (iparm(34) > 0). With two-level factorization (iparm(24) = 1) only ND options 2/3 are allowed — [oneMKL 2025.2 iparm reference](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-fortran/2025-2/pardiso-iparm-parameter.html)
- **oneMKL 2026.0:**
  - parallel ND (`iparm[1]=3`) is now the *default*;
  - "on average 2x faster than the serial counterpart" on the test set;
  - reordering has a geomean 1.7x (max 2.7x) speedup vs oneMKL 2024.0;
  - may use slightly more memory;
  - CNR mode no longer falls back to serial METIS;
  - better ordering reduced nnz(L+U) on circuit5M.
  
  Hardware: 2x Xeon 6980P, 256 cores, SuiteSparse matrices; per-matrix numbers are only shown in a figure. Parallel ND was "significantly improved in oneMKL 2025.0" — [Intel PARDISO 2026 article](https://www.intel.com/content/www/us/en/developer/articles/technical/pardiso-2026-performance-breakthrough.html)
- Intel's 2026 article does not cover changes to symbolic factorization or supernode detection — [Intel PARDISO 2026 article](https://www.intel.com/content/www/us/en/developer/articles/technical/pardiso-2026-performance-breakthrough.html)

**MUMPS**
- `ICNTL(28)`: 1 = sequential analysis with the `ICNTL(7)` ordering; 2 = parallel analysis with the `ICNTL(29)` ordering. `ICNTL(29)`: 1 = PT-Scotch, 2 = ParMETIS. Both are MPI-distributed — [PETSc MATSOLVERMUMPS](https://web.cels.anl.gov/projects/petsc/vault/petsc-3.13/docs/manualpages/Mat/MATSOLVERMUMPS.html)

**mt-METIS / multithreaded METIS**
- mt-Metis algorithm improvements (SC'15 short paper) "decrease the runtime by 1.5-11.7X and improve strong scaling by 82%" on a 36-core machine and 20 graphs. These are partitioning results, not ND fill — [SC'15 mt-Metis via unpaywall](https://unpaywall.org/10.1145%2F2833179.2833188)
- In NVIDIA cuDSS, fill-reducing ordering is still about 83.6% of single-solve time on average even with MT (multithreaded METIS) mode enabled, on mesh matrices — [Fast Sparse Matrix Permutation, arXiv 2602.00898 (search summary)](https://arxiv.org/pdf/2602.00898)

**Patch-based ND for meshes (Zarebavani, Mahmoud, ..., Dehnavi, Solomon; SIGGRAPH 2026, arXiv 2602.00898)**
- **Method:** partition the mesh into patches, compute separators on the much smaller patch quotient graph, run ND to `nd_level`, then order each etree node locally with AMD. The etree is built during the permutation, so the solver's etree step is skipped. Strict balance and separator optimality are deliberately relaxed — [arXiv 2602.00898](https://arxiv.org/abs/2602.00898)
- **Results:**
  - ordering time up to 10.27x faster (4.58x geomean) than METIS; up to 2.27x vs ParMETIS (Intel oneAPI);
  - end-to-end up to 6.62x (3.51x geomean) on cuDSS and 1.76x geomean on MKL (Xeon Gold 6248, 20 cores);
  - over 1.5M vertices: up to 6.62x (cuDSS) and 2.55x (MKL);
  - break-even above about 50K vertices (cuDSS) and 100K (MKL);
  - "less competitive on small problems (n < 10^5)";
  - permutation is on average 86% (up to 96%) of total cuDSS runtime on large meshes — [arXiv 2602.00898 PDF](https://arxiv.org/pdf/2602.00898)
- **Ablation:**
  - quotient-graph separators vs METIS_ComputeVertexSeparator: about 3x faster with +15.44% / +6.07% nnz(L) (small/large mesh);
  - local AMD instead of local METIS inside etree nodes: 12.42x / 12.74x faster with +6.67% / +10.78% nnz(L).
  
  Code: [github.com/BehroozZare/fast-permute](https://github.com/BehroozZare/fast-permute) — [arXiv 2602.00898 PDF](https://arxiv.org/pdf/2602.00898)

**METIS_NodeND internals and options (METIS 5.x source)**
- With `pfactor > 0`, dense vertices are pruned first; compression is disabled if pruning happened.
- With `compress`, `CompressGraph` merges vertices with identical adjacency, and `cfactor = nvtxs/nnvtxs`. If `cfactor > 1.5` and `nseps == 1`, then `nseps` is raised to 2.
- `ccorder` orders connected components separately (`MlevelNestedDissectionCC`).
- Recursion stops at `MMDSWITCH` vertices and finishes with MMD.
- `MlevelNodeBisectionMultiple` does a single bisection when `nseps == 1` or the graph is below 1000 (compressed) / 2000 vertices. `MlevelNodeBisectionL2` hard-codes 5 coarse bisection runs, with a 5000-vertex multilevel threshold.

  — [METIS ometis.c](https://git.acem.ece.illinois.edu/lib/METIS/src/branch/main/libmetis/ometis.c)
- METIS 5.2.x unbundled GKlib into a separate dependency (packagers build 5.2.1 against a separate GKlib) — [CRUX commit metis 5.1.0 -> 5.2.1](https://git.crux.nu/ports/contrib/commit/c472807cc386770b05ee8eeab8b2fda9a4ca05c3), [EasyBuild GKlib-METIS](https://docs.easybuild.io/version-specific/supported-software/g/GKlib-METIS/)

**KaHIP / data-reduction ND**
- Applying graph reduction rules before ND (Ost, Schulz, Strash, arXiv 2004.11315) gives better orderings and runs faster. On road networks it is "on average six times faster than Metis while computing orderings with less fill-in". This is sequential — [arXiv 2004.11315](https://export.arxiv.org/abs/2004.11315)
- **[classic, 1999]** Bornstein-Maggs-Miller: less parallel ND can give less fill than METIS, AMD and BEND — [Springer chapter](https://link.springer.com/chapter/10.1007/BFb0097983)

### Inferences
- **For smf on Windows/Linux without MPI**, the realistic large-SPD options are:
  1. run several independent METIS_NodeND calls concurrently on the two top-level subgraphs. That is DIY parallel ND: compute the top separator(s) with `METIS_ComputeVertexSeparator`, then recurse on the halves with OpenMP tasks, falling back to serial `METIS_NodeND` or AMD below a size threshold. This is essentially what PARDISO's parallel ND and the fast-permute paper do;
  2. use mt-METIS (OpenMP; licensing and maintenance need checking);
  3. tune `METIS_NodeND` options: `nseps=1`, `niter` low, and `compress=1` (helps multi-dof FEM such as bmwcra_1/pwtk, which are 6-dof structural problems);
  4. replace leaf-level METIS/MMD with AMD (the fast-permute ablation shows about 12x faster leaves at +7-11% nnz(L); the fill cost may be lower if leaves are small).
  
  Option 1 needs only serial METIS and OpenMP, so it is fully portable. The speedup bound is roughly 2x for the top bisection plus near-linear for subtrees; PARDISO's reported "about 2x average on full node" is a realistic target.
- Running ParAMD and parallel ND concurrently and keeping the better result (the paper's suggested future work) is a plausible strategy for large matrices.
- Because IPOPT reuses the symbolic analysis across iterations while the pattern is fixed, large-n ordering cost is amortized over many factorizations. The MKL comparison mainly matters for single-solve benchmarks.

### Gaps
- No primary source found with fill/flop numbers for MKL parallel ND vs serial METIS. Intel says only "almost no drawbacks" and slightly more memory.
- No 2024-2026 source found for KaMinPar or Mt-KaHyPar ND-ordering benchmarks (fill vs METIS, shared-memory speedups). Searches returned nothing specific, so it is unknown whether they offer a node-ordering mode competitive with METIS.
- PT-Scotch's threaded (non-MPI) ordering performance and quality numbers were not retrieved.
- No quantitative study found on METIS 5.2 option sweeps (`nseps`, `niter`, `ufactor`, `ccorder`) for ordering time vs fill; this would need in-house benchmarking.

## Q3. Supervariable / indistinguishable-node compression before ordering (multi-dof FEM)

### Takeaway
Compression is standard and cheap. METIS does it internally (`compress`) and raises `nseps` when the compression ratio exceeds 1.5. AMD detects indistinguishable nodes during elimination (mass elimination) but does not pre-compress. HSL's analyse codes use supervariables. For 3- and 6-dof FEM matrices, compressing up front cuts ordering, etree and column-count work by roughly the dof factor.

### Cited Findings
- METIS_NodeND `compress` option: `CompressGraph` merges vertices with identical adjacency lists before ND; `cfactor = nvtxs/nnvtxs`; if `cfactor > 1.5` then `nseps` is raised to 2. Compression is skipped if dense-row pruning (`pfactor`) was done — [METIS ometis.c](https://git.acem.ece.illinois.edu/lib/METIS/src/branch/main/libmetis/ometis.c)
- ParAMD keeps AMD's indistinguishable-node detection and mass elimination — [arXiv 2504.17097v2](https://arxiv.org/html/2504.17097v2)
- **[2012-13, borderline classic]** Hogg & Scott, "An efficient analyse phase for element problems" (NLAA 20, 2013): adapts Gilbert-Ng-Peyton to elemental input without assembling A, and exploits supervariables. The abstract reports "significant advantages" on practical problems — [Hogg & Scott NLAA PDF](https://www.numerical.rl.ac.uk/media/people/jennifer-scott/HoggScot2012_nlaa.pdf), [eurekamag record](https://eurekamag.com/research/081/708/081708327.php)
- HSL's definition: a supervariable is a set of variables that belong to the same set of finite elements — [HSL MC44](https://www.hsl.rl.ac.uk/catalogue/mc44.html)
- The patch-based ordering also handles block (multi-dof) cases by mapping each mesh vertex to a small number of unknowns ("block case") — [arXiv 2602.00898 PDF](https://arxiv.org/pdf/2602.00898)

### Inferences
- For smf, do an explicit supervariable pass before ordering: hash each column's sorted pattern including the diagonal, then confirm by comparison; this is O(nnz).
  - Run AMD or METIS on the quotient graph with vertex weights. METIS supports `vwgt`; SuiteSparse AMD has no weights, but compressing still reduces its work, and AMD's own mass elimination recovers the same supernodes.
  - Then expand the permutation and run etree and column counts on the compressed graph, scaling counts by supervariable size.
  
  For 6-dof structural matrices (bmwcra_1, pwtk) this should cut METIS time several-fold. This is implementable in portable C++.
- IPOPT KKT matrices rarely have large identical-pattern groups except in discretized optimal-control or PDE problems. The matching-based compression (Q5) is the KKT analogue.

### Gaps
- No primary documentation found on whether MKL PARDISO or MUMPS pre-compress supervariables before ordering. MUMPS has a known compressed-graph option for KKT matrices (`ICNTL(12)`), but it was not retrieved this session.
- No quantitative before/after numbers for METIS `compress` on FEM matrices were found.

## Q4. Fast symbolic factorization: etree, column counts, supernodes, relaxed amalgamation, parallel/GPU symbolic, reuse

### Takeaway
The symbolic pipeline after ordering (Liu etree, Gilbert-Ng-Peyton column counts, fundamental supernodes, relaxed amalgamation) is near-linear and well settled. Recent work speeds up the *combined* ordering + symbolic step in three ways:
- building the etree during ND (fast-permute);
- reusing orderings across pattern changes (Parth);
- reusing the whole symbolic analysis across IPM iterations (SDSL-Solver).

Amalgamation defaults differ widely: MA97 uses nemin = 8; CHOLMOD uses a tiered nrelax/zrelax rule.

### Cited Findings
- **CHOLMOD relaxed amalgamation:** merge adjacent supernodes if (ns <= nrelax[0]) || (no new zeros) || (ns <= nrelax[1] && z < zrelax[0]) || (ns <= nrelax[2] && z < zrelax[1]) || (z < zrelax[2]). Defaults give (ns <= 4) || no new zeros || (ns <= 16 && z < 0.8) || (ns <= 48 && z < 0.1) || (z < 0.05), where z is the fraction of zero entries including prior amalgamations. `supernodal_switch` defaults to 40 (flops/lnz ratio for choosing supernodal over simplicial) — [cholmod.h](https://rweb.cla.umn.edu/R/library/Matrix/include/cholmod.h)
- **HSL_MA97 nemin:** "Two neighbours in the elimination tree are merged if they both involve fewer than nemin eliminations. The default is nemin=8" — [HSL_MA97 spec](https://www.hsl.rl.ac.uk/specs/hsl_ma97.pdf)
- **Symbolic share of solver time (Parth, arXiv 2501.04011):**
  - symbolic analysis is up to 78% of Cholesky solver time in the IPC benchmark;
  - fill-reducing ordering is 62.3% (MKL), 81.1% (Apple Accelerate) and 86.1% (CHOLMOD) of symbolic time on average;
  - in remeshing, ordering is 57% of symbolic time for small meshes and 82% for large (MKL).
  
  — [Parth arXiv 2501.04011v2](https://arxiv.org/html/2501.04011v2)
- **Parth (adaptive reuse of ND orderings when the pattern changes):**
  - it re-dissects only "dirty" subtrees;
  - symbolic-phase speedups (max / median): CHOLMOD 8.05x / 4.9x; Accelerate 5.69x / 3.5x; MKL 3.14x / 2.2x;
  - end-to-end per solve: up to 2.95x (median 1.5x);
  - fill within about ±5% of the best of MKL, Accelerate and CHOLMOD;
  - Hessians n = 23K-201K, nnz(L) 11M-344M.
  
  Code: [github.com/BehroozZare/Parth](https://github.com/BehroozZare/Parth) — [Parth arXiv 2501.04011v2](https://arxiv.org/html/2501.04011v2)
- **Etree built during ordering:** fast-permute produces the etree as a by-product of ND, "eliminating a costly step in the symbolic Cholesky phase". cuDSS accepts a user permutation plus etree. Post-order vs level-order schedules give identical fill — [arXiv 2602.00898 PDF](https://arxiv.org/pdf/2602.00898)
- **Symbolic reuse in IPMs (SDSL-Solver, arXiv 2604.23979):** reuses the symbolic factorization from iteration k and refactors numerically only; for well-conditioned problems, symbolic + numeric can be redone periodically — [SDSL-Solver](https://arxiv.org/pdf/2604.23979)
- **Sympiler (SC'17):** fully decouples symbolic analysis from numeric computation and generates code specialized to the sparsity pattern, which is useful when the pattern is fixed across many factorizations — [Sympiler NSF PAR](https://par.nsf.gov/biblio/10049149)
- **Elemental-input analyse (Hogg & Scott 2013):** a Gilbert-Ng-Peyton variant that works without assembling A and uses supervariables — [Hogg & Scott NLAA](https://www.numerical.rl.ac.uk/media/people/jennifer-scott/HoggScot2012_nlaa.pdf)

### Inferences
- In IPOPT the pattern is fixed, so analyse runs once per problem (plus any re-analyse after delayed-pivot blowup). The analyse-vs-MA27 gap therefore matters only for short solves and small problems. Shaving constant overheads (allocation reuse, fused passes) matters more than asymptotics.
- **Fusion opportunities in smf [background, not re-verified this session]:**
  1. Compute the etree and the postorder in one pass with the GNP skeleton.
  2. Detect fundamental supernodes from column counts plus the postorder (child count = 1 and cc[j] = cc[parent] + 1) without forming L's pattern.
  3. Amalgamate in a single bottom-up pass over the supernodal tree.
  4. Take the assembly tree straight from AMD's absorption tree (as MA27/MA57 do) for small n, skipping the etree/colcount pass entirely.
  
  Each pass is O(nnz(A)) with small constants. Allocating one arena per analyse (or reusing it across calls) avoids several `std::vector` allocations per stage.
- For indefinite LDL^T, nemin-style amalgamation (MA97 = 8; SSIDS uses a larger value) interacts with delayed pivots. Bigger fronts give more room to place delayed pivots, so the amalgamation rule should be tuned with delays in mind, not just BLAS-3 efficiency.

### Gaps
- No source found on "SympilerX". Searches returned only the original Sympiler.
- No 2015+ paper found specifically on *parallel* or GPU column-count / etree computation. Shared-memory parallel symbolic factorization exists in SuperLU_DIST/PARDISO, but no fetched source gives numbers.
- SPRAL SSIDS documentation (nemin default, ordering defaults) was not retrievable (404). The SSIDS nemin default could not be confirmed.
- No source found for PARDISO's supernode / amalgamation rule.
- No source on "solve-aware" amalgamation (choosing supernodes to optimize triangular solves) was found.

## Q5. Orderings for KKT / saddle-point matrices (constrained and matching-compressed orderings, METIS vs AMD on the compressed graph, delayed pivots)

### Takeaway
The established technique, used in MA97 ordering = 7/8 via HSL_MC80, PARDISO/IPOPT (Schenk-Wächter-Hagemann) and MUMPS, is:
1. compute a (near-)maximum-weight matching (MC64-type);
2. pair matched off-diagonal entries into 2x2 candidate pivots;
3. compress the pairs into single nodes;
4. order the compressed graph with AMD or METIS;
5. expand.

It cuts delayed pivots on tough indefinite problems but can noticeably increase fill when no pivoting would have been needed. The matching must be close to optimal for ordering purposes, though cheap suboptimal matchings are enough for scaling.

### Cited Findings
- **HSL_MA97:** `ordering=7` computes a matching-based ordering with HSL_MC80 and AMD on the compressed matrix; `ordering=8` uses METIS on the compressed matrix. Both are "only ... for indefinite systems", need numerical values at analyse time, and come with a MC64 scaling (`scaling=3`). If values change, "it may be necessary to recall the analyse phase". A matching-based ordering "may be a good choice for tough indefinite systems" — [HSL_MA97 spec](https://www.hsl.rl.ac.uk/specs/hsl_ma97.pdf)
- IPOPT's MA97 interface exposes the same orderings, including "5 - Automatic choice between 1 and 3" — [IPOPT hsl_ma97d.h](https://www.coin-or.org/Doxygen/Ipopt/hsl__ma97d_8h_source.html)
- **Hagemann & Schenk (SISC 2006):** symmetrized maximum-weight matchings arrange matched entries into 1x1 or 2x2 diagonal blocks for LDL^T preconditioning — [SIAM SISC 10.1137/040615614](https://epubs.siam.org/doi/10.1137/040615614)
- **Hogg & Scott (NLAA 22(4), 2015):** high-cardinality *suboptimal* matchings (for example from auction algorithms, which are faster than the Hungarian algorithm in MC64) are as effective as optimal ones for *scaling*, but "a higher degree of optimality is required to effectively use matching-based ordering techniques" — [DOI 10.1002/nla.1978](https://doi.org/10.1002/nla.1978), via [STFC ePubs record](https://epubs.stfc.ac.uk/work/12387102)
- **Hogg, Scott & Thorne (ACM TOMS 44(2), 2017), "Numerically-aware orderings":** existing matching-based methods are numerically effective, "but fill-in may be significantly increased over a standard ordering if no pivoting is required". The proposed method combines matching with a numerically aware ND that depends on scaled values as well as the pattern — [CentAUR record](https://centaur.reading.ac.uk/71797)
- **[classic]** MA57 auto-ordering detected "oxo" KKT-like structure (zero (1,1) and (2,2) blocks) and chose QAMD if m1 > 1.8 m2, else METIS. The authors concluded that "attempts to use the structure of some indefinite problems do not appear to yield any additional benefits", and that the CHOLMOD-style "AMD first, METIS if fill large" rule works equally well for indefinite problems — [RAL-TR-2006-001](https://numerical.rl.ac.uk/media/reports/dsRAL2006001.pdf)
- ParAMD's best speedup (7.29x) and a fill ratio of 1.14 were on nlpkkt240 (28M rows, a KKT matrix), so parallel AMD works on large KKT graphs — [arXiv 2504.17097v2](https://arxiv.org/html/2504.17097v2)

### Inferences
- smf's current "matching-compressed AMD for KKT" matches MA97 `ordering=7`.
  - Adding the `ordering=8` analogue (METIS on the compressed graph) would let the AMD-vs-METIS auto rule from Q1 apply to the compressed graph.
  - The compressed graph has about n - |matched pairs| vertices, so METIS cost drops accordingly.
- Given Hogg-Scott-Thorne's warning, an auto policy is reasonable:
  1. run plain AMD;
  2. switch to matching-compressed ordering only when needed (the first factorization shows many delays, or the matrix has a large zero diagonal block).
  
  This matches smf's existing "auto KKT matching policy" direction (see project memory) and the RAL finding that structure-based a-priori choices did not help.
- Because a matching-based analyse depends on values, IPOPT's changing values may in principle require re-analyse. In practice the matching from the first iteration is reused, and the cost shows up as delayed pivots later.

### Gaps
- Quantitative fill / flop / delayed-pivot comparisons for MC80-compressed AMD vs METIS vs plain AMD could not be retrieved (the Reading PDF is blocked by Anubis; Duff-Pralet 2005 was not fetched).
- No source retrieved for "constrained orderings" (for example Bridson's constraint-aware ordering or Tůma/Rozložník saddle-point orderings) with numbers.
- MUMPS `ICNTL(12)` (compressed-graph ordering for symmetric indefinite) and PARDISO `iparm(13)`/`iparm(11)` matching interactions with ordering were not documented in retrieved sources.

## Q6. Memory / allocation strategy in the analysis phase of the fastest codes

### Takeaway
Fast analyse codes pre-size one workspace with elbow room and avoid both garbage collection and per-step allocation:
- SuiteSparse AMD: 1.2 nnz(A+A^T) + 9n integers;
- ParAMD: 1.5x elbow room, never exceeded in tests.

MA97 keeps the cleaned integer data in `akeep` for reuse across factorizations.

### Cited Findings
- SuiteSparse AMD memory: "1.2 * nnz(A+A^T) + 9*n", at most 2.4 nz + 9n integers. `amd_2` takes the caller's `iwlen`, so the caller can pre-size the workspace and garbage collection runs only when elbow room is exhausted — [amd.h](https://raw.githubusercontent.com/DrTimothyAldenDavis/SuiteSparse/dev/AMD/Include/amd.h)
- ParAMD pre-augments CSC storage by 1.5x. Extra memory "never exceeded 1.5x" E^0 in tests, which removes garbage collection; the factor is user-adjustable — [arXiv 2504.17097v2](https://arxiv.org/html/2504.17097v2)
- HSL_MA97 analyse optionally checks the data (HSL_MC69: sums duplicates, discards out-of-range indices) and stores the "cleaned integer matrix data" in `akeep`. Checking "incurs both time and memory overheads" and is optional — [HSL_MA97 spec](https://www.hsl.rl.ac.uk/specs/hsl_ma97.pdf)
- oneMKL PARDISO 2026 parallel ND "may use slightly more memory" than serial — [Intel PARDISO 2026 article](https://www.intel.com/content/www/us/en/developer/articles/technical/pardiso-2026-performance-breakthrough.html)

### Inferences
- For smf, small-n analyse speed should improve from:
  1. one arena allocation per analyse sized from nnz and n (AMD's 2.4 nz + 9n bound plus etree/colcount arrays of about 6n), reused across repeated analyse calls on the same handle;
  2. making the "clean matrix" step optional or fused with building the A+A^T pattern for AMD (MA97 makes checking optional);
  3. avoiding `std::vector` value-initialization (zero-fill) of large integer buffers that are immediately overwritten, for example via `std::unique_ptr<int[]>(new int[n])` or C++20 `make_unique_for_overwrite`.
- For OpenMP parallel ND, per-thread arenas avoid allocator contention. On Windows, the default CRT heap is a known bottleneck under contention **[background, not re-verified this session]**.

### Gaps
- No source found describing PARDISO's or MUMPS's internal analyse memory layout.
- No 2015+ study found that quantifies allocation overhead in symbolic analysis specifically.
