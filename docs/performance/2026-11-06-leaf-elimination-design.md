# Batched slack/multiplier leaf elimination: design (2026-11-06)

## Problem

IPOPT's augmented system for inequality constraints d(x) - s = 0:

| row/col | x | s | y_d |
|---|---|---|---|
| s   | 0 | Σ_s + δ_w | -I |
| y_d | J_d | -I | -δ_c |

The pair (s_i, y_i) touches the rest of the matrix only through row i of J_d, which has k nonzeros. On the truck-trailer KKT:
- 17.6k such pairs (slack degree 1, multiplier degree k + 1 with k = 3);
- n = 36.6k;
- 17.7k supernodes, almost all of them these pairs as 2-pivot leaf fronts of k + 2 rows.

smf's per-front fixed cost, about 0.8 µs, makes the factorization 1.4x MA27's.

## Idea

Eliminate the pairs before the multifrontal factorization, in one flat loop, and factor the reduced matrix (x, the equality multipliers, and the pairs that cannot be eliminated) with the normal engine.

Eliminating pair i: s_i first, then y_i, each a 1x1 pivot.
- d_s = a_ss.
- y_i's diagonal updates to d_y = a_yy - a_ys^2 / d_s.
- It adds a rank-1 term -(1/d_y) b b^T to the k x k block of its state neighbours, where b is column y_i's J_d entries.
- L entries: l_ys = a_ys / d_s, and l_xy = b / d_y.

## Analysis (once per pattern)

- **Detecting pairs:** a column s with exactly one off-diagonal neighbour y and a stored diagonal, whose neighbour y has a stored diagonal and degree <= k_max + 1 (k_max = 8). Such pairs are found directly on the cleaned pattern.
- **When the leaf path is armed:** only when at least 20% of n is in pairs.
- **Reduced pattern:**
  - A minus the pair rows and columns;
  - plus the k x k neighbour blocks the rank-1 terms fill, a union over the pairs, usually already present in J^T J-like blocks of W.
  - The reduced matrix is analysed by the usual analyse() (ordering, tree, supernodes). Its analysis is kept alongside the full one.
- **Precomputed maps:**
  - each pair's entries (a_ss, a_ys, a_yy, b) and their offsets in the value array;
  - each rank-1 term's target slots in the reduced values;
  - the reduced values' source slots in the full matrix.

## Factorization

1. **Pair pass (flat loop, no fronts):** for each pair, compute d_s, d_y, l_ys and l_xy. Accumulate -b b^T / d_y into a staging array aligned with the reduced value slots.
   - **Threshold test, as the TPP kernel:** accept d_s if |d_s| >= u |a_ys|. Accept d_y if |d_y| >= u * max |b| (the updated column's largest off-diagonal).
   - **Inertia:** add the signs of d_s and d_y (exact; a zero or below-small_pivot value is counted as the general kernel counts it).
2. **If every pair passes:** form the reduced values (the source slots plus the staged updates), then run factor_indef on the reduced analysis. Its inertia adds to the pairs'.
3. **If any pair fails (rare at u = 1e-8; possible at u = 0.01, Σ_s large):** wholesale fallback for this factorization. The general path runs on the full analysis, so the result is exactly today's.
   - A per-pair fallback would need the reduced pattern to keep the failing pair, which a fixed analysis cannot do.
   - The counter Info::leaf_fallbacks records it. After two fallbacks in a row the session stops trying the leaf path.
4. **Determinism:** the pair pass is serial and fixed-order, and the staged sums are accumulated in pair order, so the result is bitwise identical across thread counts. The reduced factorization keeps its own deterministic mode.

## Solve

Order: unknowns (pairs, reduced). With L = [L_pp 0; L_rp L_rr] and D blockwise:
- **Forward:**
  - pairs: z_s = b_s; z_y = b_y - l_ys z_s;
  - then the reduced right-hand side b_r - sum over pairs of l_xy z_y;
  - then the reduced forward solve.
- **Diagonal:** D^-1 on the pairs, and the reduced D.
- **Backward:**
  - reduced backward;
  - then x_y = w_y - l_xy^T x_r; x_s = w_s - l_ys x_y.
- Multiple right-hand sides use the same loops over columns.

## Scope

- **Not handled:** posdef, the static path and the scale-on-delay rescue. These keep the general path, and the leaf path is disarmed when any of them applies, at least in the first version.
- **Gate:** see the coordinator note.
- **Tests:**
  - random small KKTs with inequality slacks, solved with the leaf path against the general path (inertia equal; solutions within 1e-12 relative);
  - forced fallbacks;
  - deterministic mode at 1/2/8 threads.
