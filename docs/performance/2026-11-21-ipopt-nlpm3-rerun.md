# Mittelmann AMPL-NLP rerun after #79-#83 (run `nlpm3`, 2026-11-21)

**Question:** Do the post-`nlpm2` changes on `main` (#79 vectorized TPP panel + OpenMP column-strip updates, #80 OpenBLAS parity, #81 C ABI / native IPOPT interface, #82/#83 review and stability fixes) speed up factorization inside IPOPT on the full Mittelmann AMPL-NLP set without losing any problem?

**Setup:** i9-12900K, `smf8` only (8 threads), 45 problems, 1800 s per-problem limit, plugin `smf_ma97.dll` built from `main` `b5fe202` (threaded Intel MKL, AVX2). Plugin trace on (`SMF_PLUGIN_STATS=2`). Records: `benchmarks/results/remote/nlp_mittelmann/nlpm3/`. POUNCE, MA57, MUMPS, and the 11-08 `smf8` times come from run `nlp1` (cross-day comparison).

**IPOPT `numneg_` check (`coin-or/Ipopt#870`):** Checked all 45 plugin trace logs in `nlpm3/logs/smf8/` — zero problems had a wrong-inertia first factorization (`factor 2: corr=1` count = 0), so no problem was exposed to stock IPOPT's uninitialized `numneg_` branch.

## Result: factorization 5.1% faster, nothing lost

Measure: shifted geometric mean (10 s shift) of wall time over the 39 scored problems, a failed problem counted as 7200 s, IPOPT+smf8 divided by POUNCE (and bridged to Mittelmann's published table via MA57, bridge factor 0.725 over 34 problems).

| | nlp1 (11-08) | nlpm1 (11-10) | nlpm2 (11-11) | nlpm3 (11-21, `b5fe202`) |
|---|---|---|---|---|
| IPOPT+smf8 / POUNCE (39 problems) | 1.317 | 1.265 | 1.181 *(1.21 w/o lucky qcqp)* | **1.222** *(1.194 w/ nlpm2 qcqp)* |
| Problems solved (of 45; scored of 39) | 41 (36) | 42 (36) | 41 (36) | **41 (36)** |
| Mittelmann table, our `smf8` column (KNITRO = 1, our POUNCE = 4.46, published POU = 6.02, IPOPT+MA57 = 16.57) | 6.32 | 6.12 | 5.71 | **5.91** |

- **No problem lost; 31 of 39 scored problems remain faster than POUNCE.** The only unsolved problems are the same four as `nlpm2`: `robot_a`, `robot_b`, `robot_c` (IPOPT 3000-iteration cap with every linear solver) and `WM_CFy` (unscored; 1800 s wall cap, 616 iterations vs 624 in `nlpm2`).
- **Why the headline shifted 1.181 -> 1.222:** Purely `qcqp1500-1nc`, which in `nlpm2` took a lucky 1411-iteration path (752.5 s) vs 2806 iterations (1499.8 s) in `nlpm1`. In `nlpm3` it returned to the ~2700-iteration branch (2716 iterations, 1518.3 s). Holding `qcqp1500-1nc` at its `nlpm2` time puts `nlpm3 / POUNCE` at **1.194**, with the remaining +1.1% from cross-day AMPL evaluation speed (`t_funcs` 1.029x today vs 11-11).
- **Factorization is measurably faster across the board (`t_factor` 0.949x, `t_linsys` 0.975x vs `nlpm2`):**
  Across the 23 equal-iteration solved problems with `t_linsys > 0.5 s` (38 of 45 problems took the exact same iteration count as `nlpm2`):
  - `t_factor` geomean: **0.949x** (-5.1%) — 18 of 20 problems with `t_factor >= 0.5 s` got faster.
  - `t_linsys` geomean: **0.975x** (-2.5%, range `0.920–1.012x`).
  - `t_funcs` geomean (16 problems with `t_funcs > 0.5 s`): **1.029x** (cross-day host variance), giving `wall_s` geomean `1.015x`.
  - Relative to the 11-08 baseline (`nlp1`, 21 equal-iteration problems with `t_linsys > 0.5 s`): `t_factor` **0.630x**, `t_linsys` **0.667x**, `wall_s` **0.881x**.
- **#75 cascade scaling rule verified unchanged:** `scale=out-cascade` fired on the exact same three problems (`qcqp1500-1c`, `qcqp1000-2c`, `cont5_2_4_l`) and nowhere else.

### Notable equal-iteration factorization wins (`nlpm3` vs `nlpm2`, `t_factor >= 0.5 s`)

| Problem | Iterations | `t_factor` (`nlpm2` -> `nlpm3`, s) | Ratio | `t_linsys` (`nlpm2` -> `nlpm3`, s) |
|---|---:|---:|---:|---:|
| `qcqp500-3c` | 329 | 0.79 -> **0.69** | 0.878x | 1.33 -> 1.23 |
| `qcqp500-3nc` | 129 | 0.52 -> **0.46** | 0.886x | 0.62 -> 0.57 |
| `qcqp750-2nc` | 112 | 0.89 -> **0.79** | 0.895x | 1.01 -> 0.96 |
| `ex4_2_320` | 17 | 0.97 -> **0.87** | 0.899x | 1.29 -> 1.24 |
| `henon_120` | 266 | 12.04 -> **10.96** | 0.910x | 13.71 -> 12.72 |
| `ex1_320` | 9 | 0.59 -> **0.54** | 0.912x | 0.67 -> 0.64 |
| `svanberg` | 34 | 1.40 -> **1.30** | 0.933x | 1.82 -> 1.75 |
| `qcqp1000-1nc` | 177 | 1.87 -> **1.75** | 0.935x | 2.02 -> 1.90 |
| `dirichlet_120` | 56 | 6.49 -> **6.16** | 0.950x | 7.12 -> 6.83 |
| `cont5_2_2_l` | 48 | 1.32 -> **1.27** | 0.957x | 1.95 -> 1.94 |
| `cont5_2_3_l` | 55 | 1.47 -> **1.42** | 0.964x | 2.21 -> 2.21 |
| `clnlbeam` | 2556 | 15.61 -> **15.17** | 0.972x | 31.15 -> 31.20 |
| `qcqp1500-1c` | 99 | 4.05 -> 4.09 | 1.010x | 4.82 -> 4.88 |
