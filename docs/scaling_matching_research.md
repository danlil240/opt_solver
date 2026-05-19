# Matching-Based Scaling — Research Note

**smf solver · Phase 9 Deferred Feature**  
*Author: Agent Beta (M7.B1)*  
*Wave: pg:7 (Phase 7 — Scaling)*

---

## 1. What Is MC64?

MC64 is a Fortran reference implementation by Iain S. Duff and Jacko Koster (Rutherford
Appleton Laboratory) that computes a maximum weighted matching on the bipartite graph of
a sparse matrix and derives a diagonal scaling from that matching.

> **Primary reference:**  
> I. S. Duff and J. Koster, "On Algorithms For Permuting Large Entries to the Diagonal
> of a Sparse Matrix", *SIAM Journal on Matrix Analysis and Applications*, **22**(4),
> pp. 973–996, 2001.  
> DOI: [10.1137/S0895479899358443](https://doi.org/10.1137/S0895479899358443)

The algorithm works on a real or complex sparse matrix `A ∈ ℝⁿˣⁿ` (or the symmetric
lower triangle in our case).  It finds a perfect matching `M` in the weighted bipartite
graph `G = (R ∪ C, E)` where edge `(i, j)` has weight `log|a_{ij}|`.  The matching that
maximises `Σ_{(i,j)∈M} log|a_{ij}|` is equivalent to maximising the product of selected
entries, which tends to place the largest entries on the diagonal of the permuted and
scaled matrix.

From the optimal matching dual variables `(u, v)` (potential vectors for rows and
columns respectively), the diagonal scaling factors are:

```
D_R[i] = exp(u[i])      (row scale)
D_C[j] = exp(v[j])      (column scale)
```

The symmetrically scaled matrix `D_R A D_C` has all diagonal entries with absolute value
1 and all off-diagonal entries with absolute value ≤ 1 (in the matching rows/columns).

For symmetric matrices (as required by smf) the row and column scalings are set equal:
`D[i] = sqrt(D_R[i] * D_C[i])`, which preserves symmetry.

---

## 2. Why Matching Scaling Can Outperform MC77 Equilibration

Equilibration (MC77 / `ScalingMethod::Equilibration`) iteratively normalises rows and
columns so that the ℓ∞ or ℓ₁ norms of each row and column reach unity.  It is cheap
(O(nnz) per iteration, a handful of sweeps) but globally greedy: it can leave large
off-diagonal entries that compete with pivots and degrade numerical stability in
indefinite (LDLᵀ) factorisation.

Matching scaling produces a provably optimal weighting in the bipartite-graph sense:

| Property | MC77 Equilibration | MC64 Matching |
|---|---|---|
| All diagonal entries unit | ✗ (ℓ∞ row/col norm = 1) | ✓ (all diag entries = 1) |
| Off-diagonal entries ≤ 1 | approximately | exactly (per matching) |
| Pivot growth bound | heuristic | theoretically bounded |
| Cost | O(nnz · iters) | O(n · nnz) worst case |
| Indefinite stability | good | better for ill-conditioned problems |

Empirical evidence (e.g., from HSL MA97 and PARDISO comparisons) shows that matching
scaling can reduce the number of delayed pivots and 2×2 pivots during LDLᵀ factorisation
on saddle-point and highly indefinite systems.

---

## 3. Algorithm Outline (MC64 / 5-Phase Shortest-Path Variant)

MC64 uses a sequence of augmenting-path searches, each implemented as a modified
Dijkstra shortest-path computation on the re-weighted bipartite graph.  The five phases
of the reference code correspond to:

1. **Initialisation** — assign a trivial starting matching (e.g., greedy along diagonal).
2. **Relaxation** — compute initial dual potentials from the starting matching.
3. **Augmentation loop** — for each unmatched row, run shortest-path (Dijkstra with
   Johnson re-weighting) to find an augmenting path, extending the matching.
4. **Dual update** — update potentials `u`, `v` after each augmentation step.
5. **Scale extraction** — convert duals to row/column scale factors.

The bottleneck variant (Job = 5 in the original Fortran) maximises the minimum
`|a_{ij}|` on the diagonal, which is appropriate for ill-conditioned problems.

---

## 4. Recommended Implementation Path for Phase 9

### 4.1 Option A — Wrap HSL MC64 (if licensed)
The reference Fortran implementation from HSL (<https://hsl.rl.ac.uk/>) is available
under an academic licence.  A thin C++ wrapper calling the Fortran entry point
`mc64a_` / `mc64ad_` is straightforward.  This is the fastest path to a correct
implementation but introduces an external dependency.

### 4.2 Option B — Self-contained C++ implementation (preferred for smf)
Implement the 5-phase augmenting-path algorithm directly in C++20 using:

- A min-heap priority queue (`std::priority_queue` with a small wrapper, or a
  Fibonacci heap for asymptotically better performance).
- Johnson re-weighting so that Dijkstra edge weights are non-negative.
- `int` index arrays only — no `std::map` per the project style guide.

**Entry point signature (draft):**
```cpp
// Phase 9 target — replace the stub in scaling_matching.cpp
ErrorCode compute_matching_scale(const CscLower& A,
                                 std::vector<double>& scale,
                                 int print_level);
```

**Key internal helpers to write:**
```
matching_init()         — greedy diagonal / cheap heuristic start
dijkstra_shortest_path()— single-source, non-negative re-weighted graph
augment_path()          — walk predecessor array to extend matching
extract_scales()        — dual → D[i] = exp((u[i]+v[i])/2) for symmetry
```

### 4.3 Testing Strategy
- Compare `scale` vectors against the HSL MC64 reference output on a suite of
  small matrices (n ≤ 50) included in the unit tests.
- Verify that the scaled matrix's diagonal entries all have absolute value ≈ 1.0
  (within 1e-10 relative tolerance).
- Run the full solver with `ScalingMethod::Matching` on the existing test matrices
  and confirm that inertia matches Eigen LDLT (same as M3.B2 oracle checks).

---

## 5. Complexity Analysis

| Step | Time | Space |
|---|---|---|
| Greedy initialisation | O(nnz) | O(n) |
| n Dijkstra runs (Johnson) | O(n · nnz · log n) worst | O(n + nnz) |
| Dual update | O(n) per augmentation | O(n) |
| Scale extraction | O(n) | O(n) |
| **Total** | **O(n · nnz · log n)** | **O(n + nnz)** |

In practice, the number of non-trivial augmentation steps is much smaller than n for
matrices arising from optimisation problems (QP/NLP KKT systems), giving near-linear
observed performance.

---

## 6. Phase 9 Scope Note

This stub (`ErrorCode::FeatureNotAvailable`) is intentionally minimal.  Phase 9 will:

1. Replace the stub body with the full augmenting-path implementation.
2. Add a `CMakeLists.txt` option `SMF_MATCHING_SCALE` (default ON) that can be turned
   off to skip the feature on platforms where it is not needed.
3. Add ≥ 3 GoogleTest unit tests in `solver/tests/test_scaling_matching.cpp`.
4. Update `solver/include/smf/scaling.hpp` dispatch to call
   `compute_matching_scale` when `ScalingMethod::Matching` is selected (currently
   guarded by a `FeatureNotAvailable` check in the scaling dispatch table).

---

*End of research note.*
