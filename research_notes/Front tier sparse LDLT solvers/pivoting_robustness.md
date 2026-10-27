# Pivoting strategies for robust sparse symmetric indefinite LDL^T on KKT / saddle-point / IPM systems (state of the art, Oct 2026)

Scope: how front-tier solvers (SPRAL SSIDS, PARDISO (MKL/Panua), MUMPS, HSL MA57/MA86/MA97) pivot, and how IPM codes (IPOPT, MadNLP, HyKKT) get around dynamic pivoting. The target is "smf", a C++20 MA97-like multifrontal LDL^T inside IPOPT. smf currently uses MA27-style threshold 1x1/2x2 pivoting with delays, MC64-style matching scaling, Duff–Pralet matching-compressed ordering for KKT matrices, and opt-in static pivoting. Its weak cases are GHS_indef cvxqp3, ncvxqp3/5, bratu3d and cont-201. All five are in the "tough/hard indefinite" sets of Hogg–Scott 2013 and Duff–Hogg–Lopez 2018, so those papers' numbers transfer directly.

Extraction caveat: several PDF tables (Hogg–Scott 2013 Tables II–VI, Duff–Hogg–Lopez 2018 Tables 7.4–7.5) came out of pdftotext with misaligned columns. Numbers below are quoted only where the paper's prose confirms them, or where the alignment could be checked; anything else is flagged.

---

## Q1. Threshold partial pivoting variants (Bunch–Kaufman, bounded Bunch–Kaufman/rook, Duff–Reid/MA57 threshold 1x1/2x2): stability and performance in blocked dense kernels

### Takeaway
Threshold partial pivoting (TPP) with Duff–Reid 1x1/2x2 tests and u in [0.01, 0.1] is the robustness reference: it bounds |l_ij| <= 1/u and is backward stable. Plain Bunch–Kaufman (BK) does not bound L, so it is unsafe as a front kernel unless something else guarantees accuracy (iterative refinement, or a posteriori checks). The modern answer for blocked kernels is not rook or BK but *a posteriori* TPP (Q2): factor a block optimistically using complete pivoting inside a cache-resident tile, then check the L bound and roll back. Restricted pivoting (search only inside the diagonal block) is fast but measurably less robust.

### Cited Findings
- TPP test: a 1x1 pivot is accepted if |a_qq| >= u * max_i |a_iq|. For 2x2 pivots MA97 uses the Duff et al. (1991) test, which replaced the original Duff–Reid 2x2 test because that one "proved unnecessarily severe". u lies in (0, 0.5], and u > 0.5 can make it impossible to complete pivoting. Commonly used u is 0.1 or 0.01. — [Hogg & Scott 2013, TOMS 40(1) art. 4](https://www.numerical.rl.ac.uk/media/people/jennifer-scott/HoggScot2013_TOMS.pdf); [Duff, Hogg & Lopez, RAL-TR-2018-012](https://epubs.stfc.ac.uk/manifestation/40018271/RAL-TR-2018-012.pdf)
- Stability: bounding L entries by 1/u, together with a backward-stable 2x2 solve, is enough for backward stability. Ashcraft–Grimes–Lewis showed that the widely used Bunch–Kaufman strategy does *not* have this property; TPP does. (AGL 1998, SIMAX 20(2), is the classic source for bounded BK/rook.) — [Hogg & Scott 2013](https://www.numerical.rl.ac.uk/media/people/jennifer-scott/HoggScot2013_TOMS.pdf); [Hogg & Scott, Compressed threshold pivoting, arXiv:1305.2353](https://arxiv.org/abs/1305.2353)
- Bounded Bunch–Kaufman ("rook") is in LAPACK as `?sytf2_rook` / `?sytrf_rook`. It is the dense-kernel option when L must be bounded without a full column search. — [LAPACK hetf2_rook docs](https://netlib.org/lapack/explore-html/d2/d09/group__hetf2__rook_ga91a1c26a07e7ab02e48a7c1301ed5e6c.html)
- Dense BK and Aasen "are known to suffer from occasional numerical instability due to potentially exponential element growth or unbounded entries". Randomized complete pivoting is proposed as a remedy at comparable cost, per the authors' claim; this is dense-only work. — [Feng, Xiao & Gu, arXiv:1710.00125](https://arxiv.org/abs/1710.00125)
- MA97's 2x2 acceptance conditions (from the SSIDS/APTP report), with Δ = a11·a22 − a21²:
  - (a) |Δ| >= ½|a21|² (nonsingular);
  - (b) |Δ| >= ½|a11||a22|;
  - (c) |Δ| >= ½|a21||a21| (no significant cancellation in Δ).
  Under these, the entries of L are bounded by 4, i.e. equivalent to u = 0.25. — [Duff, Hogg & Lopez 2018, §4.1](https://epubs.stfc.ac.uk/manifestation/40018271/RAL-TR-2018-012.pdf)
- Why TPP parallelises poorly: checking a pivot needs the whole candidate column below the diagonal up to date and scanned. That forces 1D block-column partitioning, with much less parallelism than the 2D tiling used for Cholesky. MA86 and MA97 both use 1D task partitioning; MA97 uses a recursive, bit-compatible kernel. — [Duff, Hogg & Lopez 2018](https://epubs.stfc.ac.uk/manifestation/40018271/RAL-TR-2018-012.pdf)
- Restricted pivoting checks only the r rows of the candidate block, |a_{q+1,q+1}| > u · max over q+1 < i <= r of |a_{i,q+1}|. This enables Level-3 BLAS and parallelism but needs more refinement. In Hogg–Scott's tough set:
  - METIS + restricted pivoting: refinement and FGMRES "were unable, in general, to recover the required accuracy".
  - Matching ordering + restricted pivoting: "generally successful" but failed the accuracy target on 3 problems, so "it does not offer the robustness that we require". — [Hogg & Scott 2013 §2.4, §4.2](https://www.numerical.rl.ac.uk/media/people/jennifer-scott/HoggScot2013_TOMS.pdf)
- Relaxed TPP (lower u toward u_min when no pivot passes; Duff–Pralet 2007, MA86 `umin`): gains were "modest and do not justify the additional complexity" compared with simply using a small u throughout. — [Hogg & Scott 2013 §2.3](https://www.numerical.rl.ac.uk/media/people/jennifer-scott/HoggScot2013_TOMS.pdf)
- Compressed threshold pivoting (Hogg–Scott) builds a small p×p "compressed" matrix C representing A21 and runs TPP on [A11; C]. This cuts messages to O(log P) instead of O(p log P) for an n×p panel on P processors.
  - The "strict" variant is provably stable but can over-delay.
  - The "relaxed" variant can be unstable but was stable in practice with appropriate scaling and ordering. — [arXiv:1305.2353](https://arxiv.org/abs/1305.2353)
- Kim & Eijkhout's earlier "try-it-and-see" approach: restricted pivoting at each node with an a posteriori growth check below the diagonal block. On excessive growth, the front is rebuilt from the children's contribution blocks and pivots are delayed. This is the conceptual ancestor of APTP. — [arXiv:1305.2353 §3](https://arxiv.org/abs/1305.2353)

### Inferences
- For smf's dense front kernel, the choice is effectively among MA27/MA57-style TPP, BK or rook with an a posteriori L-bound check (APTP-like), and supernode-BK with perturbation (PARDISO-like). Rook alone does not remove delays: it still selects only within the fully-summed block, so the delay count is set by the u-test against the off-diagonal (A21) rows, not by the dense pivot rule.
- The MA97/APTP 2x2 test (a)–(c) is cheap and gives a clean u = 0.25-equivalent bound. It is a good drop-in replacement if smf still uses the older MA27 2x2 test.

### Gaps
- No 2015–2026 sparse-solver study was found that compares rook against BK against TPP head to head on GHS_indef (delays or backward error). Evidence is dense-only or indirect.
- The Ashcraft–Grimes–Lewis (1998) paper itself was not fetched; its claims are cited via Hogg–Scott.

---

## Q2. SPRAL SSIDS a posteriori threshold pivoting (APTP): mechanism, performance vs MA97, delays, TPP fallback

### Takeaway
APTP (Duff, Hogg & Lopez 2018; NLAFET WN21; also SISC 2020) gets TPP-level robustness at PARDISO-level speed on multicore. It speculatively factors 2D tiles, checks the L bound after the fact, and leaves failed columns in place, so rows and columns never move between tiles. Failed columns are then refactored with serial TPP or delayed to the parent. On the hard-indefinite set (which contains smf's weak cases) SSIDS matched PARDISO's speed, was up to about 10x faster than MA97, and gave near machine-precision residuals where PARDISO did not. Its weak spot is the serial, unblocked TPP fallback.

### Cited Findings
- **Algorithm.** The front is tiled with outer block size nb. For each block column j:
  - `Factor(A_jj)`: back up A_jj, run a pivoted LDL^T on it, and set nelim_j = block size.
  - `ApplyN(A_ij)` for i > j: back up the block, compute L_ij = A_ij P_j (L_jj D_jj)^{-T}, find the first column with |l| > 1/u, and atomically set nelim_j = min(nelim_j, that column).
  - `ApplyT`: does the same for previously failed entries in blocks to the left.
  - `Adjust`: reduces nelim_j by one more if it would split a 2x2 pivot.
  - `UpdateNN`, `UpdateNT`, `UpdateTN`: apply updates with only the first nelim_j pivots, restoring failed entries from the backups.
  - After the sweep, failed columns are permuted to the back and refactored with APTP or TPP, or passed to the parent. — [Duff, Hogg & Lopez 2018, Alg. 1, §3](https://epubs.stfc.ac.uk/manifestation/40018271/RAL-TR-2018-012.pdf)
- **Two enablers.**
  - "Fail-in-place": failed columns stay where they are and are kept updated, so there is no data movement across tiles.
  - "Speculative execution": tile tasks run assuming no failure elsewhere, which needs backups and backtracking. The paper calls the backup overhead "generally negligible" relative to the block work. — [same, §3](https://epubs.stfc.ac.uk/manifestation/40018271/RAL-TR-2018-012.pdf)
- **Inner kernel.** The approach is two-level: parallel APTP with block size nb recurses to serial APTP with inner block size ib. Diagonal ib-blocks are factored with *complete pivoting*: take the largest remaining entry; use it as a 1x1 if it is diagonal, otherwise test it as the off-diagonal of a 2x2 with |Δ| >= ½|a_mt|², else fall back to a 1x1 on max(|a_mm|, |a_tt|). This is proved at least as stable as TPP with u = 0.25 (|l| <= 4) and pushes failures as late in each block as possible. — [same, Alg. 2, §4](https://epubs.stfc.ac.uk/manifestation/40018271/RAL-TR-2018-012.pdf)
- **Delays and fallback.** By default, columns still uneliminated after the APTP pass go to a *sequential* TPP pass. Passing them directly to the parent was "deemed to be potentially too costly". Fallback cost dominates when many columns fail, because "our implementation of TPP in SSIDS is not only sequential but is also unblocked". Example: on PARSEC/Ga10As10H30, lowering u from 1e-1 to 1e-3 cut failed columns after APTP from 1157 to 0 and greatly reduced time. On GHS_indef/ncvxqp7, lowering u from 1e-1 to 1e-3 "dramatically" reduced delays and halved factor time. — [same, §7](https://epubs.stfc.ac.uk/manifestation/40018271/RAL-TR-2018-012.pdf)
- **Setup.** 28-core Broadwell (2x14). SSIDS default u = 0.01. The hard-indefinite set (TSOPF, GHS_indef cont-201/300, stokes128, ncvxqp1/3/5/7, darcy003, bratu3d, cvxqp3, d_pretok, turon_m, c-big) used matching-based ordering plus scaling; the easy set was unscaled. PARDISO ran with its default iterative refinement, and its refinement cost is *not* included in its factor time. — [same, Tables 7.1–7.2](https://epubs.stfc.ac.uk/manifestation/40018271/RAL-TR-2018-012.pdf)
- **Results on easy matrices (factor time, seconds).**
  - PARSEC/Ga19As19H42: SSIDS 23.9 vs MA86 29.3, MA97 175.4, PARDISO 61.6.
  - GHS_indef/sparsine: SSIDS 7.9 vs MA97 43.9, PARDISO 12.3.
  - PARDISO's refinement failed to converge on Oberwolfach/t2dal and Cunningham/qa8fk unless scaling and ordering preprocessing was enabled. — [same, Table 7.3 and text](https://epubs.stfc.ac.uk/manifestation/40018271/RAL-TR-2018-012.pdf)
- **Results on hard matrices: accuracy.** SSIDS achieved small backward errors on nearly all problems; PARDISO did not always.
  - GHS_indef/stokes128: PARDISO residual 3.88e-3 "despite using scaling and iterative refinement", vs SSIDS near machine precision.
  - GHS_indef/cvxqp3: PARDISO 8.92e-7 vs SSIDS 1.70e-11.
  - The authors conclude APTP gives "similar performance to PARDISO but with the same robustness provided by the traditional TPP algorithm". — [same, §7 text](https://epubs.stfc.ac.uk/manifestation/40018271/RAL-TR-2018-012.pdf)
- **Results on hard matrices: speed** (alignment-checked reading of Table 7.4; verify against the paper). Factor times SSIDS vs MA97 vs PARDISO:
  - ncvxqp3: about 1.0 s vs 11.0 s vs 1.2 s; scaled residuals SSIDS about 2.4e-10, PARDISO about 1.2e-9.
  - ncvxqp7: about 1.6 s vs 19.7 s vs 3.0 s.
  - Schenk_IBMNA/c-big: about 2.3–3.0 s vs (not shown) vs PARDISO.
  - The text confirms "SSIDS performs favourably compared to HSL MA97 … the performance gap increases with the problem size". — [same, Table 7.4](https://epubs.stfc.ac.uk/manifestation/40018271/RAL-TR-2018-012.pdf)
- **SSIDS lineage.**
  - Original GPU SSIDS (Hogg, Ovtchinnikov & Scott, TOMS 2016) combined TPP with a multifrontal method, gave bit-compatible results, and was up to 4.6x faster than a multicore CPU multifrontal solver. — [Reading CentAUR record](https://centaur.reading.ac.uk/74696)
  - SSIDS v2.0 added CPU and hybrid execution via OpenMP tasks with APTP. — [STFC record](https://epubs.stfc.ac.uk/work/30692911)
- **IPOPT's SPRAL interface.**
  - `spral_pivot_method` ∈ {aggressive ("Aggressive a posteori pivoting"), block ("Block a posteori pivoting", the default), threshold ("Threshold partial pivoting (not parallel)")}.
  - `spral_scaling` default is `matching`; options are none/mc64/auction/matching/ruiz/dynamic.
  - In dynamic mode: switch_1 = at_start with scaling_1 = matching; switch_2 = on_demand with scaling_2 = mc64; switch_3 = never.
  - `spral_small` = 1e-20; `spral_small_subtree_threshold` = 4e6 flops. — [IPOPT options](https://coin-or.github.io/Ipopt/OPTIONS.html)
- **Robustness in independent tests.**
  - In a GPU linear-solver review on IPOPT/ACOPF KKT matrices, "SSIDS is able to solve some of the systems that cause all other packages to fail"; STRUMPACK and SSIDS were "the two largely robust" packages. — [Świrydowicz et al., arXiv:2106.13909](https://arxiv.org/abs/2106.13909)
  - Inside IPOPT on CUTEst, SPRAL (CPU) "has difficulty solving easy problems quickly but eventually solves as many problems as PARDISO". — [Tasseff et al., arXiv:1909.08104](https://arxiv.org/abs/1909.08104)

### Inferences
- APTP is the best documented route for smf to close the PARDISO gap without giving up exact inertia. APTP never perturbs: every accepted pivot passes the u-test, so the count of negative D eigenvalues is the exact inertia of the scaled matrix, unlike SBK.
- SSIDS's own measurements say the fallback must be *blocked*. A Level-3 blocked TPP (or a recursive APTP pass on the failed set with a lower u) for failed columns is the obvious improvement over SSIDS. With tens of thousands of failures on ncvxqp/bratu3d, an unblocked fallback would erase the APTP gain.
- The u = 1e-1 → 1e-3 data show that with good scaling and ordering, a smaller u removes most delays and failures cheaply. This matches IPOPT's practice of starting at 1e-8 and escalating to 1e-4 (Q5).

### Gaps
- Default nb/ib block sizes were not given in the report text; check the SPRAL source (`src/ssids/cpu/kernels/ldlt_app.cxx`) for its defaults.
- The exact semantics of "aggressive" versus "block" APTP are not described in the report; IPOPT documents only the option names.
- Per-matrix delay counts for cvxqp3, ncvxqp5 and bratu3d under APTP could not be extracted reliably from Table 7.5.

---

## Q3. PARDISO (Schenk–Gärtner): supernode-Bunch–Kaufman (SBK) + pivot perturbation, matching (iparm 11/13), refinement, inertia for IPMs

### Takeaway
PARDISO restricts BK 1x1/2x2 pivoting to the diagonal block of each supernode and never delays. When no acceptable pivot exists, it replaces tiny pivots with ±ε·‖A‖ (static pivoting; default ε = 1e-8 for symmetric indefinite). It relies on 1–2 steps of iterative refinement and on symmetric weighted matching to put large entries into 2x2 blocks. The result is fast but not backward stable, and the inertia it reports is that of the *perturbed* matrix. This matches smf's observation that MKL is 2–8x faster but sometimes fails a 1e-10 backward-error gate.

### Cited Findings
- **SBK (Schenk & Gärtner, ETNA 23, 2006).** Bunch–Kaufman 1x1/2x2 pivot selection is performed *within the diagonal block of a supernode*, which "has no effect on the overall fill-in". When the supernode cannot be factored with BK pivots, a perturbation is applied: the pivot magnitude is tested against a constant threshold α = ε·‖A‖ with ε half machine precision, and "any tiny pivots … are set to ±α". "The factorization is, in general, not accurate and iterative refinement will be needed". The authors used two refinement steps whenever perturbations occurred. — [Schenk & Gärtner 2006](https://etna.ricam.oeaw.ac.at/vol.23.2006/pp158-179.dir/pp158-179.pdf)
- **Matching variants.** SBK-COMP-1/2 add symmetric maximum-weighted matchings, with matched 2-cycles forced into 2x2 diagonal blocks. "Weighted graph matchings have a strong influence on the overall performance of the augmented symmetric indefinite systems, where the method without symmetric matchings SBK is superior if it works". In their tests, scaling made "an insignificant difference". Profile legends report 0 failures for SBK, SBK-COMP-1 and SBK-COMP-2 on their test set under that protocol: backward-error success threshold, 2 refinement steps. — [same, §5](https://etna.ricam.oeaw.ac.at/vol.23.2006/pp158-179.dir/pp158-179.pdf)
- **MKL PARDISO parameters.**
  - iparm(10): small pivots are perturbed with eps = 10^(−iparm(10)); default 1e-13 for unsymmetric, **1e-8** for symmetric indefinite (mtype −2, −4, 6).
  - iparm(8) = 0 (default): "two steps of iterative refinements when perturbed pivots are obtained".
  - iparm(11): scaling, disabled by default for symmetric indefinite.
  - iparm(13): weighted matching (the old MKL page is internally inconsistent about its symmetric default).
  - iparm(14) (output): number of perturbed pivots.
  - iparm(21): 0 = 1x1 diagonal pivoting, 1 = 1x1 and 2x2 Bunch–Kaufman (default).
  - iparm(22) and iparm(23) (output): numbers of positive and negative eigenvalues (inertia). — [MKL PARDISO parameter docs (older MKL)](https://www.irya.unam.mx/computo/sites/manuales/fce12/mkl/mkl_manual/ssr/ssr_pardiso_parameters.htm)
- **MKL guidance for IPMs.** Use iparm(11) = 1 (scaling) and iparm(13) = 1 (matching) "for highly indefinite symmetric matrices, for example, from interior point optimizations or saddle point problems". Numerical values must be supplied at analysis (phase 11) when matching is on. If complete supernode pivoting fails, PARDISO uses a perturbation strategy "similar to [Li99], [Schenk04]". — [Intel oneMKL PARDISO iparm](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-fortran/2025-2/pardiso-iparm-parameter.html)
- **IPOPT's Pardiso interface.**
  - `pardiso_matching_strategy` default is "complete+2x2" (IPAR(13) = 2); alternatives are complete (1) and constraints (3).
  - `pardiso_redo_symbolic_fact_only_if_inertia_wrong` default "no", meaning *always* redo the symbolic factorization when elements were perturbed.
  - `pardiso_repeated_perturbation_means_singular` default "no".
  - `pardiso_skip_inertia_check` exists ("makes the algorithm non-robust").
  - The default ordering is METIS. — [IPOPT options](https://coin-or.github.io/Ipopt/OPTIONS.html)
- **Critiques.**
  - "Since a perturbed system is solved, the computed inertia of the original matrix A may not be reliable", and the solve can need many refinement steps, sometimes without recovering accuracy. — [Hogg & Scott 2013 §2.5](https://www.numerical.rl.ac.uk/media/people/jennifer-scott/HoggScot2013_TOMS.pdf)
  - SBK "trade[s] stability for performance … the addition of perturbations can make it difficult to preserve inertia which is essential for some optimization applications". — [Duff, Hogg & Lopez 2018 §2](https://epubs.stfc.ac.uk/manifestation/40018271/RAL-TR-2018-012.pdf)
- **Measured accuracy failures** (MKL PARDISO, matching + scaling, default refinement): stokes128 residual 3.88e-3; cvxqp3 8.92e-7, against SSIDS near machine precision. — [Duff, Hogg & Lopez 2018 §7](https://epubs.stfc.ac.uk/manifestation/40018271/RAL-TR-2018-012.pdf)
- **IPOPT benchmark (Tasseff et al. 2019).**
  - 2D PDE problems: "PARDISO vastly outperforms all other solvers, but for a small number of problems, fails".
  - 3D PDE problems: PARDISO and SPRAL "are faster and more robust than other solvers".
  - OPF: PARDISO and SPRAL "are typically much slower" than MA27/MA57.
  - In the paper's characterisation, PARDISO prefers "performance over robustness". — [Tasseff et al., arXiv:1909.08104](https://arxiv.org/abs/1909.08104)

### Inferences
- The 2–8x MKL advantage on cvxqp3/ncvxqp/bratu3d comes from never delaying: the symbolic structure is static, so supernodes keep full Level-3 BLAS efficiency. Its failures to reach 1e-10 are the expected cost of ±ε‖A‖ perturbation followed by only 2 refinement steps.
- When perturbations happen, iparm(22)/(23) count the signs of the pivots *actually used*. A pivot perturbed to +α where the "true" Schur-complement pivot was a tiny negative number flips the reported inertia. IPOPT's Pardiso interface treats any perturbation as suspicious and redoes symbolic analysis rather than trusting the inertia.
- For smf, copying SBK would trade away exact inertia. A safer hybrid is APTP-like or SBK speculation *with* rollback to TPP or delay whenever |l| > 1/u, plus perturbation only as a last resort, *counted and reported* so the IPM can react (as MA57 and MA86 do; Q5).

### Gaps
- The Panua PARDISO 7/8 manual (current IPM-specific options and inertia semantics under perturbation) could not be retrieved. Whether Panua's newer releases report inertia corrected for perturbed pivots is unverified.
- Exact MKL defaults for iparm(13) on mtype −2 vary across documentation versions (the old page contradicts itself); check the current header or release notes.

---

## Q4. MUMPS: delayed pivots, null-pivot detection (ICNTL(24)), static pivoting (CNTL(4)), 2x2 pivots, effect of ICNTL(6)/ICNTL(12)

### Takeaway
MUMPS (guide 5.9.1, newer than the requested 5.7/5.8) does threshold pivoting with delays and 2x2 pivots (CNTL(1) = 0.01 default for general symmetric matrices). For SYM = 2 it implements Duff–Pralet preprocessing, matching-derived 1x1/2x2 recommendations plus compressed-graph ordering, through ICNTL(6) and ICNTL(12). Static pivoting (CNTL(4)) and null-pivot detection (ICNTL(24)) are mutually exclusive. With null-pivot detection, INFOG(12) negative-pivot counts exclude null pivots, so inertia is reported as (pos, neg, null) separately.

### Cited Findings
- **CNTL(1)**, relative pivoting threshold: default −1 (automatic), which gives 0.01 for general symmetric, 0.0 for SPD, and 0.1 with rank-revealing (ICNTL(56) = 1). Values > 0.5 are treated as 0.5 for symmetric matrices. "A larger value … increases fill-in but leads to a more accurate factorization". — [MUMPS 5.9.1 User's Guide](https://mumps-solver.org/doc/userguide_5.9.1.pdf)
- **ICNTL(6)**, symmetric case (SYM = 2):
  - Values 1–6 compute a column permutation that is "internally used to determine a set of recommended 1×1 and 2×2 pivots".
  - 5 = maximise product of diagonal entries plus scaling vectors (|diag| = 1, |off-diag| <= 1 after scaling).
  - 7 = automatic choice (the default for SYM = 0 or 2).
  - Advice: "let MUMPS select the strategy (ICNTL(6)=7) or to set ICNTL(6)=5 if the user knows that the matrix is for example an augmented system (… a large zero diagonal block)". — [same](https://mumps-solver.org/doc/userguide_5.9.1.pdf)
- **ICNTL(12)**, ordering strategy for SYM = 2: 0 = automatic (default), 1 = usual, 2 = ordering on the compressed graph, 3 = constrained ordering (AMF only).
  - ICNTL(12) takes priority over ICNTL(6) and ICNTL(8): if ICNTL(12) = 2 and ICNTL(6) = 0, ICNTL(6) is treated as 5 (with values) or 1.
  - ICNTL(12) = 3 forces ICNTL(6) = 5 and ICNTL(8) = −2. — [same](https://mumps-solver.org/doc/userguide_5.9.1.pdf)
- **INFOG(13)** = total delayed pivots, counted again each time a variable is re-delayed. "A large number (more than 10% of the order of the matrix) indicates numerical problems. Settings related to numerical preprocessing (ICNTL(6), ICNTL(8), ICNTL(12)) might then be modified". — [same](https://mumps-solver.org/doc/userguide_5.9.1.pdf)
- **CNTL(4)**, static pivoting (as in Li–Demmel):
  - < 0 is off (default −1).
  - > 0 sets pivots whose magnitude is below CNTL(4) to CNTL(4).
  - = 0 uses an automatic threshold of √ε·‖A_pre‖.
  - It replaces pivots that TPP would postpone, so estimates are not exceeded.
  - INFOG(25) counts modified pivots.
  - **It is incompatible with ICNTL(24) = 1 and ICNTL(56) = 1, and is then ignored.** — [same](https://mumps-solver.org/doc/userguide_5.9.1.pdf)
- **ICNTL(24) = 1**, null-pivot-row detection:
  - A pivot is null if the ∞-norm of its row/column < thres. CNTL(3) > 0 gives thres = CNTL(3)·‖A_pre‖; = 0 (default) gives thres = ε·‖A_pre‖·N_h (N_h = deepest branch size); < 0 gives |CNTL(3)|.
  - CNTL(5) > 0 sets a null pivot to sign(piv)·CNTL(5)·‖A_pre‖ (recommended about 1e20); CNTL(5) = 0 (SYM = 2) zeroes the L column and sets D = 1.
  - INFOG(28) = null-space size; PIVNUL_LIST lists the rows. — [same](https://mumps-solver.org/doc/userguide_5.9.1.pdf)
- **INFOG(12)** (SYM = 1/2) = number of negative pivots.
  - With ICNTL(24) = 1 or ICNTL(56) = 1 it "excludes the null pivots, even if their sign is negative"; INFOG(40) gives the number of negative pivots among the null ones.
  - With ScaLAPACK on the root (ICNTL(13) = 0, the default) it excludes root pivots, giving only a lower bound; set ICNTL(13) >= 1 or use one process for an exact count.
  - With rank-revealing, inertia "might be incorrect if INFO(1) >= 16". — [same](https://mumps-solver.org/doc/userguide_5.9.1.pdf)
- **ICNTL(56) = 1**: postpones pseudo-singular pivots to the root and factors the root with an SVD-based rank-revealing method; the default CNTL(1) rises to 0.1. — [same](https://mumps-solver.org/doc/userguide_5.9.1.pdf)
- **IPOPT benchmarks.** Within IPOPT on CUTEst, "MA57 and MUMPS appear to be the best performers over the majority of the problem set" (MUMPS 4.10.0 in that study). — [Tasseff et al. 2019](https://arxiv.org/abs/1909.08104)

### Inferences
- MUMPS's design matches smf's (TPP + delays + Duff–Pralet). Its documented heuristic, INFOG(13) > 10% of n meaning bad preprocessing, is a useful trigger for smf to switch scaling or ordering automatically, similar to MA97's 0.05·n "high_delay" switch.
- The CNTL(4)/ICNTL(24) exclusivity is informative: once pivots are silently replaced, MUMPS does not claim a meaningful null-space or inertia. smf's opt-in static pivoting should similarly disable or flag inertia reporting.

### Gaps
- No public quantitative MUMPS-specific study (2018–2026) was found of delayed-pivot counts on GHS_indef under ICNTL(6) = 5 / ICNTL(12) = 2 against defaults; the evidence is Duff–Pralet's older MA57-based study (Q5).
- The guide does not say whether CNTL(4) static pivots preserve the sign of the original pivot.

---

## Q5. MA57 / MA86 / MA97 options: u threshold, static pivoting, small-pivot handling, scaling options and measured effect in IPOPT

### Takeaway
All three HSL codes default to u = 0.01 (double precision), but IPOPT overrides this to u = 1e-8, escalating up to 1e-4 on inaccurate solves or wrong inertia. MA57 and MA86 offer *sign-preserving, delay-budgeted* static pivoting (CNTL(4)/CNTL(5); `static`); MA97 offers none. MA97 instead offers five scalings (none, MC64, MC77, matching-based, MC30). IPOPT's "dynamic" policy turns MC64 scaling on only after on-demand or high-delay triggers (more than 0.05·n new delays). Duff–Pralet's older study found symmetrised MC64 scaling the most robust; MC30 should not be used for indefinite systems.

### Cited Findings
- **MA57** (v3.11.3 spec):
  - CNTL(1) = 0.01, the TPP threshold (used when ICNTL(7) = 1).
  - CNTL(2) = 1e-20, the zero-pivot threshold.
  - **CNTL(4) = 0.0**: if > 0, "small pivots may be replaced by entries of value CNTL(4)" (static pivoting).
  - **CNTL(5) = 0.0**: if > 0, CNTL(4) is treated as 0 (normal delaying) until CNTL(5)·N fully-summed variables have been delayed. That is a delay budget before static pivoting starts.
  - INFO(25) rank: "if static pivoting is invoked … the rank … will always be n".
  - INFO(24) = number of negative eigenvalues; INFO(35) = number of pivots chosen in the static pivoting phase.
  - ICNTL(7) = 4: no pivoting, but pivots altered to share one sign (modified-Cholesky-like).
  - ICNTL(15) = 1 (default): symmetrised MC64 scaling.
  - ICNTL(16) = 1: blocks of entries < CNTL(2) are discarded and the pivots moved to the end (helps highly rank-deficient matrices). — [MA57 spec](https://www.hsl.rl.ac.uk/specs/ma57.pdf)
- **HSL_MA86** (v1.8.0 spec):
  - u = 0.01 (initial).
  - umin = 0.01: relaxed threshold; if fewer than expected pivots pass, the best candidate with tolerance v >= umin is accepted and u is set to v.
  - `static` = 0.0: if > 0 and pivots fail even at umin, diagonal entries are accepted; those with |a| < static are given value sign(a)·static (requires static >= small).
  - `small` = 1e-20.
  - `action` = .true. (continue on singularity with a warning).
  - Info: `num_perturbed`; `usmall` = the threshold actually used. — [HSL_MA86 spec](https://www.hsl.rl.ac.uk/specs/hsl_ma86.pdf)
- **HSL_MA97** (v2.8.1 spec):
  - u = 0.01 (double), range [0, 0.5].
  - small = 1e-20.
  - action = .true.
  - `scaling`: 0 none/user (default), 1 MC64, 2 MC77 (1 iteration ∞-norm + 3 iterations 1-norm), 3 scaling from the matching-based ordering (requires ordering 7/8), 4 MC30.
  - Info: `num_delay` (fully-summed variables passed to the parent), `num_two` (number of 2x2 pivots).
  - The indefinite dense kernel is "based on the same pivoting algorithm as HSL_MA64".
  - No static pivoting option. — [HSL_MA97 spec](https://www.hsl.rl.ac.uk/specs/hsl_ma97.pdf)
- **IPOPT defaults**:
  - ma27_pivtol = ma57_pivtol = 1e-8, pivtolmax = 1e-4.
  - ma86_u = 1e-8, ma86_umax = 1e-4.
  - ma97_u = 1e-8, ma97_umax = 1e-4.
  - ma57_small_pivot_flag (ICNTL(16)) = 0.
  - ma86_scaling = mc64; ma86_order = auto (AMD vs METIS, pick best).
  - linear_scaling default MC19 for MA27/57/77/86. — [IPOPT options](https://coin-or.github.io/Ipopt/OPTIONS.html)
  - Hogg–Scott describe the escalation: IPOPT starts at u = 1e-8; on "unexpected inertia or large backward error" it refactors with a larger u, up to 1e-4. — [Hogg & Scott 2013 §2.2](https://www.numerical.rl.ac.uk/media/people/jennifer-scott/HoggScot2013_TOMS.pdf)
- **IPOPT's MA97 scaling switch.**
  - `ma97_scaling` = dynamic (default), with `ma97_scaling1` = mc64 and `ma97_switch1` = od_hd_reuse.
  - `on_demand`: scale after IPOPT requests a better solution (refinement failed).
  - `high_delay`: scale "after more than 0.05*n delays are present".
  - The `_reuse` variants reuse the previous iteration's scaling unless the trigger fires again.
  - Further stages: ma97_scaling2 = mc64 / ma97_switch2, and so on.
  - Options: none, mc30, mc64, mc77. — [IPOPT options](https://coin-or.github.io/Ipopt/OPTIONS.html)
- **Duff & Pralet (RAL-TR-2004-020 / SIMAX 2005; older classic, flagged), MA57 study on symmetric indefinite sets:**
  - Failures: no scaling 6; MC30 3; MC77inf 3; MC77one 1; MC64SYM 1.
  - MC64SYM was "within a factor of 1.17 of the best on 100% of the problems"; MC77inf was good on 95%.
  - MC64SYM gave "infinite speedup on 5 problems, a speedup greater than 5 on 7 matrices".
  - "We recommend that the MC30 scaling is not used on symmetric indefinite problems."
  - BRATU3D was hard to solve "except with the MC77one scaling". — [Duff & Pralet RAL-TR-2004-020](https://numerical.rl.ac.uk/media/reports/duprRAL2004020.pdf)
- **Hogg & Scott 2013** (TOMS; tough set = more than max(5000, 0.2n) delays with METIS + MC64; includes cvxqp3, ncvxqp1/3/5/7, bratu3d, cont-201/300, stokes128, darcy003, d_pretok, turon_m and TSOPF):
  - (i) Matching-based ordering (HSL_MC80: MC64 matching → compressed graph → METIS_NodeWND with weight 2 for 2x2 super-vertices) gives "substantially fewer delayed pivots" at u = 0.01. Its *predicted* nz(L)/flops are typically 50–100% above METIS.
  - (ii) With matching ordering and **u = 1e-8, delays drop to 0 or near 0 on all test problems**, and one refinement step sufficed. METIS with u = 1e-8 instead needed many solves, sometimes FGMRES.
  - (iii) METIS with u = 0.01: actual nz(L) often more than 50% above prediction; flops up to about 10x above prediction (TSOPF_FS_b39_c30).
  - (iv) Static pivoting (Duff–Pralet style, static = √ε·‖Â‖) on METIS needs thousands of perturbed pivots and many refinement or FGMRES solves; on the matching ordering "far fewer pivots" are perturbed and few refinement steps are needed. "Knowledge of the inertia may be lost."
  - (v) Bridson's constrained ordering with u = 0 (pivot-free) is much denser. Ratios to METIS prediction: cvxqp3 nz(L) 3.51x, flops 9.91x, 22 refinement solves; cont-201 1.74x / 2.75x / 1.
  - (vi) HSL_MA97 full-solve times: the matching ordering "for many problems … gives the best time".
  - Recommendation: a matching-based approach for numerically challenging problems. — [Hogg & Scott 2013](https://www.numerical.rl.ac.uk/media/people/jennifer-scott/HoggScot2013_TOMS.pdf)
- **Matching-ordering cost on easy matrices.** On matrices that are *not* tough, the matching ordering can cost a lot. For Schenk_IBMNA/c-62 and c-64, MC80 gave "significantly more entries in the factor and a higher flop count, leading to a slower solution time", while MC64 scaling alone already removed most delays. — [Hogg & Scott 2013 Table VIII text](https://www.numerical.rl.ac.uk/media/people/jennifer-scott/HoggScot2013_TOMS.pdf)
- **MA97 within IPOPT** (Tasseff et al. 2019, defaults): on CUTEst, "HSL_MA97 and WSMP are generally slower and appear less robust than the aforementioned solvers" (MA57, MUMPS, MA86, PARDISO). — [arXiv:1909.08104](https://arxiv.org/abs/1909.08104)

### Inferences
- smf's tens of thousands of delays on cvxqp3/ncvxqp/bratu3d, despite Duff–Pralet compression, are inconsistent with Hogg–Scott's "0 delays at u = 1e-8 with matching ordering". Likely causes are that smf runs at u = 0.01 in these benchmarks, or that its compressed-ordering or scaling pipeline differs from MC80 (weighted METIS on the compressed graph, MC64 scaling from the *same* matching). In IPOPT the effective u starts at 1e-8, so benchmark smf at u = 1e-8 as well as 0.01.
- MA57's CNTL(5) delay budget (static pivoting only after a fraction of N has been delayed) is a cheap guard worth copying: static pivots stay rare, and smf can report "perturbed" so IPOPT increases δ_c/δ_w or u instead of trusting inertia.
- The sign-preserving static value (MA86 `static`; MA57 CNTL(4)) preserves inertia exactly only if the perturbed pivot's sign equals the true one. That cannot be known for |pivot| near 0, so perturbed counts must invalidate the inertia claim.
- Scaling policy: compute matching scaling with the matching-compressed ordering (MA97 scaling = 3) and reuse it across IPM iterations; rescale on a delay spike (> 0.05·n) or refinement failure. smf's recent "stale reused scale" fix matches this od_hd_reuse model.

### Gaps
- No 2015–2026 controlled IPOPT study isolating MC64 vs MC77 vs auction scaling for MA97/SSIDS (iterations, time, failures) was found; only option semantics and the older Duff–Pralet MA57 data.
- HSL_MA97 "ma97_switch" effect sizes are not published in the sources found.

---

## Q6. Regularization instead of dynamic pivoting in IPMs: quasidefinite, primal-dual regularization, HyKKT / condensed KKT on GPUs, and how inertia is preserved

### Takeaway
- If the KKT matrix is made symmetric quasi-definite (SQD: −δ_d I in the (2,2) block, H + δ_w I positive definite), LDL^T exists for *any* symmetric permutation, so a static ordering with no pivoting works and inertia is fixed by construction. For convex problems infinitesimal δ suffices.
- For nonconvex NLP the augmented system is not SQD without "aggressive" regularization. GPU IPMs therefore condense:
  - LiftedKKT: the primal condensed SPD system, factored with Cholesky (LDL^T in practice).
  - HyKKT: Cholesky of an augmented-Lagrangian (1,1) block plus CG on the Schur complement.
- Inertia is inferred from Cholesky success via Sylvester's law. This is robust on OPF but less robust on degenerate CUTEst problems.

### Cited Findings
- **SQD theory.** If A is SQD, "the LDL factorization exists for any given permutation (strongly factorizable)". Many KKT systems are SQD, become SQD "with infinitesimal regularization, or can be converted to SQD systems". IPMs use primal-dual regularization (δ_p, δ_d), raised until the inertia equals (n + m, m, 0). — [Montoison, Pacaud, Shin & Anitescu, arXiv:2508.16094 (2025)](https://arxiv.org/abs/2508.16094)
- **Convex versus nonconvex.**
  - Convex: all formulations (augmented, condensed) "can become SQD for infinitesimal δ_p, δ_d > 0"; MadIPM uses the augmented system with *fixed* primal-dual regularization.
  - Nonconvex: augmented systems "are not guaranteed to be SQD unless aggressive (beyond what is necessary to ensure the descent condition) regularization" is used. The primal condensed system is the most suitable, since it can be made SPD with (δ_p, δ_d) chosen "based solely on inertia correction"; MadNLP uses this. — [same](https://arxiv.org/abs/2508.16094)
- **Friedlander & Orban** (Math. Prog. Comp. 4(1):71–107, 2012): "exact" primal-dual regularization for LP/convex QP, interpreted as simultaneous proximal-point regularization of the primal and dual. The method still converges to a solution of the original problem for suitable parameters, and each iteration solves an SQD system. — [friedlander.io](https://friedlander.io/publications/2010-regularized-ip-for-qp/); [GERAD G-2018-07](https://www.gerad.ca/en/papers/G-2018-07)
- **HyKKT** (Regev, Chiang, Darve, Petra, Saunders, Świrydowicz, Peleš; OMS 38, 2023; arXiv:2110.03636):
  - Regularize H with δ₁, doubling δ₁ until Cholesky succeeds ("minimal value of regularization (to within a factor of 2)").
  - Form H_γ = H + γ J^T J; run CG on J H_γ^{-1} J^T (+ δ₂ I for rank-deficient J).
  - Inertia: Theorem 3 shows the algorithm yields a descent direction "without computing the inertia of the KKT matrix". Cholesky success certifies H_γ SPD, which implies H is PD on null(J) for large γ.
  - Against MA57 on ACOPF sequences, HyKKT is more than 2x faster on one Eastern Interconnection matrix and more than 3x over the whole series. But "the direct solve is generally more accurate by 2–3 orders of magnitude".
  - "With MA57, all our test cases required substantial permutations even with a lax pivot tolerance of 0.01 (and a tolerance as large as 0.5 may be required to keep the factorization stable)." — [arXiv:2110.03636](https://arxiv.org/abs/2110.03636)
- **Condensed GPU IPMs** (Pacaud, Shin, Montoison, Schanen & Anitescu; arXiv:2405.14236 v3, Jul 2026):
  - Sylvester's law gives inertia(K2) = (n + m_i, m_e + m_i, 0) ⇔ inertia(K1) = (n, m_e, 0), so inertia correction can be run on the condensed matrix.
  - LiftedKKT relaxes equalities to inequalities so that K is SPD, with δ_w chosen dynamically from the inertia. The "relaxation introduces an error … of the same order as the solver tolerance".
  - The condensed systems are more ill-conditioned, but the "structured ill-conditioning" limits accuracy loss.
  - cuDSS: "the 2x2 pivoting strategy … is not supported". The authors used cuDSS LDL^T on condensed matrices because "LDL^T is more robust than Cholesky factorization"; cuDSS numerical factorization was about 2x faster than HSL MA86.
  - Results: up to 10x speedup on large PGLIB-OPF. On CUTEst they report "diminished robustness". HyKKT is less robust than LiftedKKT: on ROCKET (a singular arc), γ = 1e7 was insufficient and γ = 1e12 was needed. — [arXiv:2405.14236](https://arxiv.org/abs/2405.14236)
- **cuDSS static perturbation.** Small diagonal pivots after partial pivoting are replaced by `pivot_epsilon`; the default reported on NVIDIA's forum is 1e-13 (double) and 1e-5 (single). It can be set to 0, and the count is available via CUDSS_DATA_NPIVOTS. — [NVIDIA dev forum](https://forums.developer.nvidia.com/t/cudss-is-sometimes-wrong-where-cusparse-and-umfpack-succeed/342132); [CUDSS.jl docs](https://docs.juliahub.com/General/CUDSS/stable)

### Inferences
- "Robust inertia with no dynamic pivoting" is available only when the matrix is *by construction* SPD or SQD, i.e. convex QP/LP with δ_p, δ_d > 0, or a condensed SPD system with inertia-corrected δ_w. For general nonconvex IPOPT KKT systems the safe pivot-free route is to change the *formulation*, not to bolt static pivoting onto LDL^T.
- For smf inside IPOPT, an optional "SQD mode" is worth having: when IPOPT already applies δ_c > 0 on the (2,2) block, try a static, pivot-free LDL^T with a fixed ordering, and check that the pivot signs match the SQD pattern (n positive, m negative). If any pivot has the wrong sign or is tiny, fall back to TPP/APTP. That gives the PARDISO-like static-structure speed when it is safe and exact inertia either way. This is an inference, not a published smf-like result.

### Gaps
- Vanderbei's 1995 quasidefinite paper and Altman–Gondzio 1999 (regularized LP/QP IPMs; older classics) were not fetched. Their quantitative claims (regularization magnitudes, failure rates) are not included.
- No published head-to-head was found of "SQD static LDL^T" against "TPP LDL^T" inside IPOPT on nonconvex sets (failures, iterations).

---

## Q7. 2018–2026 comparisons of robustness of these strategies on large KKT test sets (failures, delayed pivots, backward error)

### Takeaway
- There are few direct comparisons. The best are:
  - Duff–Hogg–Lopez 2018: APTP (SSIDS) vs TPP (MA86/MA97) vs SBK (PARDISO) on 21 easy + 24 hard indefinite matrices, with times and residuals.
  - Tasseff–Coffrin–Wächter–Laird 2019: 10 linear solvers inside IPOPT on CUTEst, PGLib-OPF and 2D/3D PDE sets.
  - Świrydowicz et al. 2021/22: GPU and CPU solvers on IPOPT ACOPF KKT matrices.
  - Pacaud et al. 2024–2026: condensed GPU IPMs against MA27/MA57.
- Hogg–Scott 2013 (pre-2015, but the definitive delay/perturbation study on exactly smf's hard matrices) remains the main source of delay and perturbation counts.
- Consistent message: perturbation-based SBK is fastest but occasionally inaccurate; APTP gets near-SBK speed with TPP accuracy; matching ordering plus a small u almost eliminates delays.

### Cited Findings
- **Duff, Hogg & Lopez 2018** (Q2): SSIDS gave near machine-precision scaled residuals on nearly all hard problems; PARDISO had 3.88e-3 (stokes128) and 8.92e-7 (cvxqp3); MA97 residuals were "similar to SSIDS". — [RAL-TR-2018-012](https://epubs.stfc.ac.uk/manifestation/40018271/RAL-TR-2018-012.pdf)
- **Tasseff et al. 2019**, IPOPT with MA27, MA57, HSL_MA77, MA86, MA97, MUMPS, PARDISO, WSMP, SPRAL CPU and SPRAL GPU:
  - CUTEst: MA57 and MUMPS best; MA86 and PARDISO solve many quickly but fewer overall; MA27 slower but solves a similar number; MA97 and WSMP slower and "less robust"; SPRAL GPU worst.
  - OPF: MA27/MA57 track the virtual best.
  - 2D PDE: PARDISO "vastly outperforms" but fails on a few problems.
  - 3D PDE: PARDISO and SPRAL fastest and most robust; MA27 "performs most poorly of all".
  - Matching-based ordering and scaling were used for the PDE experiments. — [arXiv:1909.08104](https://arxiv.org/abs/1909.08104)
- **Świrydowicz et al.** (ACOPF KKT from IPOPT/MA57): SSIDS solved systems on which all other packages failed; SSIDS and STRUMPACK were the robust ones; cuSOLVER (QR-based) had the lowest norm-wise backward error. — [arXiv:2106.13909](https://arxiv.org/abs/2106.13909)
- **Pacaud et al.** (v3, 2026): the condensed GPU IPM gives about 10x on large OPF but reduced robustness on CUTEst (Q6). — [arXiv:2405.14236](https://arxiv.org/abs/2405.14236)
- **Hogg & Scott 2013** (TOMS, pre-2015, flagged): on the tough set, ranked by delays, matching (u = 1e-8) ≈ 0 < matching (u = 0.01) < MA47 ordering < METIS (u = 1e-8) ≈ METIS (u = 0.01). Static pivoting with METIS perturbs thousands of pivots and sometimes fails accuracy even with FGMRES. — [HoggScot2013_TOMS.pdf](https://www.numerical.rl.ac.uk/media/people/jennifer-scott/HoggScot2013_TOMS.pdf)

### Inferences
Design recommendation for smf, synthesized from the evidence above:
1. **Keep exact-inertia TPP semantics as the contract**; IPOPT needs them. Make the dense front kernel *APTP-style*: 2D tiles, inner complete pivoting with the MA97 2x2 test (a)–(c), a posteriori |l| <= 1/u check, fail-in-place, and rollback from backups.
2. **Blocked fallback** for failed columns: blocked or recursive TPP at the same u, then delay to the parent. Do not use unblocked serial TPP, which is SSIDS's measured bottleneck.
3. **Preprocessing**: MC64 matching → Duff–Pralet 2x2 compression → weighted METIS on the compressed graph, with the *same matching's* scaling (MA97 scaling = 3 / SSIDS "matching"). Reuse it across IPM iterations; recompute on a delay spike (> 0.05·n, the MA97/SPRAL `high_delay` rule; MUMPS flags > 10%·n) or on a refinement failure.
4. **Threshold policy**: start at u = 1e-8 as IPOPT does, which with matching ordering gave about 0 delays in Hogg–Scott. Escalate toward 1e-4 or 1e-2 only on inertia or backward-error failure.
5. **Static pivoting only as a budgeted last resort** (MA57 CNTL(5)-style), always reported (count of perturbed pivots, like iparm(14)/INFOG(25)/num_perturbed). When it is non-zero, report inertia as untrusted so IPOPT increases regularization rather than accepting the step. Optional SQD fast path when δ_c > 0 (Q6).
6. Targeted gates: the APTP paper shows TPP-class solvers reach about 1e-15–1e-10 scaled residuals on these exact matrices, while SBK reaches 1e-3–1e-7 on some. smf's 1e-10 gate is therefore realistic for APTP but not for SBK.

### Gaps
- No 2020–2026 paper was found that reports delayed-pivot counts *and* backward errors for MUMPS 5.x, MKL PARDISO, SSIDS and MA97 on a common GHS_indef/KKT set with an inertia-correctness metric. That comparison would need to be run in-house.
- The SISC 2020 published version of the APTP paper may include revised tables; only the 2018 technical report was read.
- GPU sparse LDL^T with genuine 1x1/2x2 dynamic pivoting (beyond cuDSS's partial pivoting with epsilon replacement): no 2024–2026 robustness data were found.
