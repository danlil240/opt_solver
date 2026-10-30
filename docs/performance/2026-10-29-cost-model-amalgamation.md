# Cost-model amalgamation (2026-10-29)

Lever 1 of `2026-10-29-path-to-number-1.md`: merge supernodes by what the merge saves in contribution-block (CB) traffic against the padding flops it adds, instead of by pivot count alone.

All numbers are smf against smf on one binary, 1 thread, Ryzen desktop, interleaved per case (`tools/bench/ab_run.py`: each case runs every variant once per round, alternating the order, 3 rounds, median). Ratios are B/A of the median; class summaries are shifted geometric means (shift 10 ms, as `score.py`).

## TL;DR

- **New rule `Amalgamation::CostModel`, now the default** (`Control::amalgamation`). It takes every merge the old `Smf` rule takes, plus a child whose CB has at least 128 rows when
  `pc · z · (pc + fp + qc + α) < 3 · α · qc²`, with α = 2 flops per word.
- **Factor time against the old rule** (1 thread, shifted geomean):

  | split | hard_indef | large_kkt | small_kkt | spd |
  |---|---|---|---|---|
  | tuning | **0.78** | 0.96 | 1.00 | 0.93* |
  | held-out | **0.83** | 0.95 | 1.00 | 0.97 |

  \* From the sweep that overlapped another session's 1-thread SSIDS run (section 4). nd24k's 0.60 is confirmed on the quiet machine at 8/16 threads (0.59-0.61). The other SPD cases move ±3%, within that run's noise: read them as neutral.

  Wall time is 0.93 (tuning) and 0.94 (held-out); solve and analyse are unchanged.
- **Headline cases (factor):** flops change by +0-5% and delays fall (ncvxqp3 441 → 70, bratu3d 449 → 33).

  | case | old rule | CostModel | B/A |
  |---|---|---|---|
  | c-big | 15.7 s | **2.8 s** | 0.18 |
  | ncvxqp3 | 2.56 s | 0.92 s | 0.36 |
  | ncvxqp7 (held-out) | — | — | 0.37 |
  | cvxqp3 | — | — | 0.44 |
  | ncvxqp1 / ncvxqp5 | — | — | 0.54 |
  | pde_control_250 (ipopt track) | — | — | 0.59 |
  | bratu3d (held-out) | — | — | 0.59 |
  | nd24k (SPD) | 52 s | 31 s | 0.60 |
- **8/16 threads** (12 case-tracks): factor **0.55**, wall 0.70; c-big 14.0 s → 1.7 s at 16 threads. Narrow fronts could not use threads; merged ones can.
- **Accuracy:**
  - ctest 67/67.
  - No case changes accuracy class in the A/Bs (score.py gate, both splits).
  - IPOPT KKT replays (16 models, every stored step): 0 failing steps on 15, including case9241 0/73. On cops_polygon_200, 1/70 fails by a hair: step 68's componentwise omega is 1.37e-8 against the 1e-8 gate (old rule 9.9e-10), normwise 7e-17.
  - IPOPT end-to-end on cops_polygon_200, cops_elec_200, case9241, pde_control_250 and acopf_grid_160: the same iterations and objectives under both rules.
- **Two guards:**
  - CBs below 128 rows are left to the old rule. Small fronts are cheap to move and merging them pads small KKTs.
  - A **delay guard**: no cost merge across a front that receives a certain delay (a single-column leaf with a zero diagonal). Without it cops_elec_200 at u = 1e-8 was 1.47x slower.
- **Also changed:** the scale-on-delay early stop now checks a front's work *before* factoring it. Same decision, less wasted work.
- **Known regressions:** pde_control_250 (1.69x) and pde_control_120 (1.29x), both on the recommended track (u = 0.01) of recorded IPOPT sequences.
  - Every regular factorization there is faster.
  - The loss is the one step on which scale-on-delay's stopped try trips the work limit later and more slowly (section 5).
  - IPOPT's own u = 1e-8 never arms scale-on-delay; on that track both cases are 0.59-0.72.
- **Not done:** in-place assembly of the last child. smf assembles each front at its factor offset, and CBs live on a separate packed stack, so the two cannot overlay without a layout change. After this rule c-big factors at about 60 GF/s, so data movement no longer dominates there.

## 1. Model

A child front c (pc pivots, CB of qc rows, front fc = pc + qc) merges into its parent group p (front fp rows: its pivots, those of the children merged into it so far, and the rows below). The child's CB lies inside the parent's front, so after the merge every child column carries z = fp − qc explicit zeros.

**Words saved** (data movement):

| term | words |
|---|---|
| the child's front is no longer zeroed | fc²/2 |
| its CB is no longer copied out of the front (read + write) | qc² |
| ... nor extend-added (read, plus read-modify-write of the parent) | 1.5 qc² |
| the parent front grows by pc rows: more to zero | − (pc·fp + pc²/2) |
| **net** | **3 qc² − pc·z** |

**Flops added:** with F(p, q) = p³/3 + p²q + pq² for p pivots over q rows below, the merge costs exactly F(pc, fp) − F(pc, qc) = pc·z·(pc + fp + qc). This is zero for z = 0: a padding-free merge is the same factorization, blocked differently.

**Rule:** merge when `pc·z·(pc + fp + qc) + α·pc·z < 3·α·qc²`.

This fits in the existing bottom-up pass of `amalgamate_relaxed` (`supernode_detection.cpp`, `cb_merge_pays`): every quantity is already there (nelim, the column counts of the last columns). The rule itself is O(1) per tree edge and needs no new data (the delay guard of section 3 adds one byte per column).

Why the old rule missed this: it merges only when *both* fronts have fewer than nemin = 8 pivots. c-big's narrow supernodes (10-100 pivots) carry CBs of thousands of rows, so each one was copied, zeroed and extend-added at roughly 1 flop per word moved. smf ran at 11 GF/s there, with 70% of factor time in assembly and memmove.

## 2. Tuning (tuning split only)

**α (flops per CB word)**, bench_profile, 1 thread:

| α | c-big factor | TSOPF_FS_b300_c2 factor (flops) |
|---|---|---|
| 2 | 2.94 s | 1020 ms (1.89e10) |
| 4 | 2.98 s | 1022 ms (1.96e10) |
| 8 | 2.96 s | 1100 ms (2.23e10) |
| 16 | 3.03 s | 1621 ms (3.63e10) |
| old rule | 17.4 s | 1033 ms (1.88e10) |

c-big's merges have tiny z and are taken at any α. α ≥ 8 over-merges TSOPF's mid-size (in-cache) blocks.

**Minimum CB size** (α = 4): the factor geomean on 16 cases is 0.854 / 0.861 / 0.864 / 0.880 for 32 / 64 / 128 / 256 rows. At 32 rows, kkt_pde_opt stores 50% more entries (64: +29%; 128: none). 128 leaves the fronts of small problems as the old rule builds them.

**Interleaved A/B, all 97 tuning case-tracks** (factor, B/A against the old rule, 3 rounds). This sweep ran from 21:49 to 22:36. Another session's 1-thread SSIDS benchmark ran alongside it from 21:48 to 22:29. The variants were interleaved per case, so they saw the same load, but differences under about 10-15% between them are not resolved:

| variant | hard_indef (42) | large_kkt (21) | small_kkt (26) | spd (8) | all |
|---|---|---|---|---|---|
| α 4, CB ≥ 128 | 0.755 | 0.983 | 1.013 | 0.946 | 0.881 |
| α 4, CB ≥ 64 | 0.743 | 0.975 | 1.011 | 0.943 | 0.873 |
| **α 2, CB ≥ 128** | **0.739** | **0.969** | **1.004** | **0.934** | **0.867** |

α = 2 also stores less: TSOPF_FS_b300 +22% factor entries against +37% at α = 4.

An extra term for the merged pivot block's upper triangle (pc·pp words, stored but never read by the solve) changed no tree: wide merges also have large CBs. It was dropped.

## 3. The delay guard

The model counts no delays. On cops_elec_200 at u = 1e-8 (IPOPT's threshold), 200 zero-diagonal constraint rows are single-column leaves. Each one delays into its parent, a node of a padding-free chain (z = 0).

- **Old rule:** it groups that chain 8 columns at a time and absorbs most leaves into the groups.
- **Cost rule without the guard:** it merges the whole chain into the root, locks the leaves out, and all 200 delays land in one 800-row front. Flops rise 37%, entries 75%, factor time 47%. At u = 0.01 the same tree was 3% *faster* (delays 543 → 200).

The guard: a width-1 leaf whose column has a zero diagonal in the analysed values delays for certain. The cost rule takes no merge into or out of a group that receives such a delay.

- `symbolic_analysis.cpp` passes the zero diagonals.
- The analysis has values: the IPOPT plugin defers it to the first factorization.
- Pattern-only analyses get no guard.

On the dev set at u = 1e-8 the guard changes only elec's tree: same supernode counts as the unguarded rule on ncvxqp3/5, cvxqp3, helm3d01, acopf_grid_80/160, pde_control_250 and copter2. elec returns to the old tree.

## 4. Results (final binary: α 2, CB ≥ 128, delay guard, work-limit pre-check)

Interleaved A/B, 1 thread, 3 rounds, smf against smf.

- **SPD:** neither the delay guard nor the pre-check applies to SPD (both are indefinite-only), so the SPD rows come from the α 2 / CB ≥ 128 variant of the tuning sweep. That sweep overlapped the SSIDS run, so only nd24k's large gain (0.60, confirmed clean at 8/16 threads) is a measured win. The other SPD cases (±3%) count as neutral.
- **Clean run:** the final runs (held-out 22:37-22:43, tuning 22:43-22:55, threads 22:56-23:14) started after the SSIDS run ended (22:29) and had the machine to themselves. Part of the sweep in section 2 overlapped another session's SSIDS benchmark. Interleaving kept that comparison fair, but its absolute times are noisier: c-big's Smf-rule baseline read 18.5-20.7 s there against 15.7 s here.

**Shifted geomean B/A (shift 10 ms):**

| split | metric | hard_indef | large_kkt | small_kkt | spd | all |
|---|---|---|---|---|---|---|
| tuning | factor | **0.782** (42) | 0.963 (21) | 0.996 (26) | 0.934 (8) | 0.881 (89, no spd) |
| tuning | wall | 0.870 | 0.972 | 0.999 | 0.940 | 0.930 |
| tuning | solve / analyse | 0.99 / 0.99 | 0.99 / 1.00 | 1.00 / 1.00 | — | 0.99 / 1.00 |
| held-out | factor | **0.826** (20) | 0.947 (8) | 1.001 (9) | 0.969 (4) | 0.899 (41) |
| held-out | wall | 0.887 | 0.978 | 1.003 | 0.978 | 0.937 |

No case-track changes accuracy class (score.py's gate, both splits). The IPOPT KKT replay gate is in section 4b.

**Per case, factor time (ms, median of 3), delays and factor entries (B/A):**

| case | track | old rule | CostModel | B/A | delays | entries |
|---|---|---|---|---|---|---|
| c-big | ipopt | 15746 | 2810 | **0.18** | 0 → 0 | 1.12 |
| c-big | recom | 15737 | 2814 | **0.18** | 0 → 0 | 1.12 |
| ncvxqp3 | ipopt | 2568 | 918 | 0.36 | 247 → 17 | 1.17 |
| ncvxqp3 | recom | 2555 | 917 | 0.36 | 441 → 70 | 1.17 |
| cvxqp3 | ipopt | 216 | 95 | 0.44 | 26 → 2 | 1.21 |
| ncvxqp1 | recom | 99 | 53 | 0.54 | 165 → 20 | 1.22 |
| ncvxqp5 | recom | 638 | 351 | 0.55 | 87 → 48 | 1.16 |
| pde_control_250 | ipopt | 696 | 409 | 0.59 | 0 → 0 | 1.07 |
| acopf_grid_160 | ipopt | 798 | 616 | 0.77 | — | 1.07 |
| acopf_grid_80 | ipopt | 103 | 88 | 0.85 | — | 1.05 |
| helm3d01 | recom | 184 | 158 | 0.85 | 27 → 27 | 1.08 |
| TSOPF_FS_b300 | recom | 299 | 265 | 0.89 | 4852 → 4270 | 1.02 |
| copter2 | recom | 256 | 245 | 0.96 | 136 → 132 | 1.04 |
| pglib case4661 | ipopt | 111 | 105 | 0.94 | — | 1.01 |
| TSOPF_FS_b300_c2 | recom | 894 | 891 | 1.00 | 30406 → 29079 | 1.01 |
| cops_elec_200 | ipopt | 23 | 23 | 1.00 | 200 → 200 | 1.00 |
| stokes128 | recom | 117 | 118 | 1.01 | — | 1.01 |
| **pde_control_250** | **recom** | 1484 | 2513 | **1.69** | 4086 → 2226 | 1.07 |
| nd24k (spd) | — | 52155 | 31417 | **0.60** | — | 1.13 |
| apache2 / G3_circuit (spd) | — | 4763 / 5255 | 4698 / 5088 | 0.99 / 0.97 | — | 1.01 / 1.02 |
| *held-out:* ncvxqp7 | recom | 3367 | 1248 | 0.37 | 170 → 16 | 1.19 |
| aug3dcqp | recom | 23 | 12 | 0.51 | 3 → 3 | 1.11 |
| bratu3d | recom | 216 | 127 | 0.59 | 449 → 33 | 1.17 |
| cops_polygon_200 | recom | 35 | 24 | 0.67 | 348 → 1691 | 1.14 |
| pde_control_120 | ipopt | 82 | 59 | 0.72 | 0 → 0 | 1.06 |
| finan512 (spd) | — | 30 | 26 | 0.88 | — | 1.03 |
| ncvxqp9 | recom | 40 | 40 | 0.98 | 3641 → 3580 | 1.00 |
| TSOPF_FS_b300_c3 | recom | 1269 | 1313 | 1.03 | 34107 → 33122 | 1.01 |
| **pde_control_120** | **recom** | 192 | 248 | **1.29** | 982 → 631 | 1.06 |

The small-front guard holds: stokes128 and ncvxqp9 (+10-13% at nemin 128) are at 0.98-1.01, and the small KKTs and optim20 do not move.

**optim20 (bench_profile, best of 7, two rounds):** the Smf rule's second round reads 0.989 against its first (the noise floor). CostModel reads 0.958 and 0.946. It moves only the cases where it fires: indef_rand_1500 0.65-0.68, spd_elasticnet_1500 0.83-0.87, spd_normal_1200 0.86-0.91, kkt_eq_qp 0.95. No case is slower.

### 4b. IPOPT KKT replays (`tools/bench/replay_gate.py`, u = 1e-8, every stored step)

Every recorded factorization of the 16 `ipopt_hard` captures is replayed. The gate per step: eta ≤ 1e-10 and omega ≤ 1e-8 after up to 3 refinement steps.

| model (steps) | old rule | final rule | factor ms, old → final |
|---|---|---|---|
| pglib case9241 (73) | 0 failing | 0 failing | 2503 → 2541 |
| pglib case4661 (60) | 0 | 0 | 1304 → 1239 |
| pglib case2869 (52) | 0 | 0 | 511 → 498 |
| pglib case2000 / 1354 / 500 / 300 / 118 | 0 | 0 | ±2% |
| cops_elec_200 (744) | 0 | 0 | 3368 → 3402 |
| cops_glider_400 (1000), rocket, steering, catmix, pinene | 0 | 0 | ±5% |
| cops_chain_800 | Singular (as before) | Singular | 429 → 429 |
| **cops_polygon_200 (70)** | **0** | **1** (step 68) | 269 → 225 |

Inertia at the last step is identical on every model.

**The one failing step: cops_polygon_200 step 68.**

- It is a late, near-singular iterate, and the worst step under every tree.
- Normwise it is fine (eta 7e-17); componentwise omega after refinement is:

  | tree | omega |
  |---|---|
  | old rule | 9.9e-10 |
  | cost rule without the delay guard | 4.8e-9 (passes) |
  | final | 1.37e-8 (over the 1e-8 gate) |

- Under the final tree refinement stagnates at 1.37e-8: 3 and 5 steps give the same value, where the old tree's error keeps falling to 4e-11.
- So at u = 1e-8, where pivoting is almost unconstrained, merging makes this one step's factor a few times less accurate, and the guard's tree tips it over the line.
- IPOPT's own residual test is normwise. The end-to-end IPOPT check is below.

**IPOPT end-to-end** (smf_ma97 plugin built from this commit with each default, 1 thread, `max_iter 3000`):

| model | iterations | objective | status | factorization s, old → final |
|---|---|---|---|---|
| cops_polygon_200 | 40 = 40 | −0.674981234810125 (same digits) | Optimal | 0.335 → 0.274 |
| cops_elec_200 | 180 = 180 | 306.000663982863 (same) | Optimal | 2.34 → 2.31 |
| pglib case9241 | 68 = 68 | 46010.1276390406 (same) | Optimal | 2.84 → 2.86 |
| pde_control_250 | 11 = 11 | 0.956329703063882 (1e-16 rel.) | Optimal | **1.60 → 0.96** |
| acopf_grid_160 | 23 = 23 | 118949.253432325 (3e-16 rel.) | Optimal | **3.83 → 2.90** |

IPOPT takes the same path. The replay step that fails the componentwise gate does not change the solve.

That leaves the gate itself as a decision for the user: accept the merged tree on this evidence, or tighten the delay guard further (which would trade away merges).

## 5. Interplay with scale-on-delay (recommended track, u = 0.01)

Scale-on-delay (`Control::scale_on_delay_ratio`, 16) stops a factorization once its front work (Σ eligible columns × rows² per front) passes 16× the tree's own delay-free work. It then redoes the factorization with a matching scaling. That ratio was tuned on the old tree, and merged fronts change both sides of it:

- The delay-free reference grows in work units: pde_control_250, 2.38e9 against 1.88e9.
- It shrinks in time: 78 ms against 135 ms.
- Merged trees absorb delays better: step 2 of pde_control_250 has 97,870 delays at 5× the work, accepted, where the old tree tripped.
- A try that does trip runs longer first: 1.9e10 work units/s against 4.2e10 in the old tree's cascades.

**pde_control_250 / pde_control_120 (recommended track only):**

- Every factorization that does not rescale is faster under the cost tree: 80 against 134 ms and 12.5 against 16.5 ms.
- The switch to matching scaling comes one step later, through a costlier stopped try. pde_control_250 step 3 takes 1.9 s, pde_control_120 step 2 takes 204 ms against 115.
- Sequence factor time: 1.87x (250) and 1.29x (120).
- With scale-on-delay off, the cost tree wins every step. Delays fall 2-4x; pde_control_250 step 3 takes 11.8 s against 30.3 s.

**The pre-check:** the early stop now tests each front's work before its kernel runs (`process_front`) and counts it. Work is only ever added, so a front that would cross the limit means the finished run crosses it too: the same decision. That cut the stopped try on pde_control_250 from 2.9 s to 1.9 s. It is included in all final numbers.

**pglib case4661 (recommended track), before the delay guard:**

- The factorizations were equally fast under both rules (~30 ms).
- The kept-scale probe's backward errors sat at the 1e-9 reuse cap: 1.2-1.9e-9 on the cost tree against 0.1-0.9e-9 on the old one.
- So three more steps recomputed the matching (0.25 s each): 1.16x.

With the guard, the OPF trees barely change (entries +1%) and the case is 0.97. Still, the reuse cap is a threshold that any change of pivot order can cross.

**Scope:** neither effect exists on the ipopt track. u = 1e-8 is below `kWeakPivotRatio`, so scale-on-delay is never armed there; pde_control_250 is 0.60 and case4661 0.95 on that track. The unit tests in `test_scale_on_delay.cpp` pin `Amalgamation::Smf`: their KKTs are sized for the trigger on the old tree (work ratio 24+), and on the cost tree they delay to only 9.5×.

## 6. Threads

8 and 16 threads, 12 tuning case-tracks: the ipopt track for the KKT and hard cases, plus nd24k. Interleaved, 3 rounds, final binary.

| threads | hard_indef (8) | large_kkt (3) | nd24k | all 12 | wall, all |
|---|---|---|---|---|---|
| 8 | 0.499 | 0.710 | 0.591 | **0.553** | 0.703 |
| 16 | 0.504 | 0.712 | 0.593 | **0.557** | 0.702 |

Per case at 16 threads (factor):

| case | B/A |
|---|---|
| c-big (14.0 s → 1.69 s) | **0.12** |
| ncvxqp3 | 0.26 |
| cvxqp3 | 0.38 |
| ncvxqp5 | 0.40 |
| pde_control_250 | 0.53 |
| nd24k | 0.59 |
| acopf_grid_160 | 0.70 |
| helm3d01 | 0.72 |
| copter2 | 0.95 |
| cops_elec_200 | 0.94 |
| TSOPF_FS_b300 | 1.05 (0.93 at 8 threads) |
| stokes128 (11-13 ms runs) | 1.07 (1.04 at 8 threads) |

The gain is larger than at 1 thread, as the path-to-#1 plan expected: narrow fronts cannot use threads.

- c-big on the old tree barely scaled: 15.7 s at 1 thread, 14.0 s at 16 threads.
- On the merged tree it goes from 2.8 s to 1.7 s.
- That is still only 1.7x at 16 threads, so lever 2 (cross-front tiling at the top of the tree) remains.

## 7. Memory

Merging stores more factor (entries count the f × p rectangle of every front, the upper triangle of each pivot block included) and keeps fewer CBs alive on the stack at once.

| class | factor entries | peak RSS |
|---|---|---|
| hard KKT with merges (ncvxqp*, cvxqp3, bratu3d) | +17-22% | +9-15% |
| c-big | +12% | +5% |
| TSOPF_FS_b300 | +22% | +4% |
| large_kkt (acopf, pde_control) | +5-7% | ±1% |
| SPD (nd24k, apache2, G3_circuit, cfd2, bmwcra_1, pwtk, ...) | +1-13% | **−20 to −39%** |
| small KKT / untouched trees | 0 | 0 |

One outlier: pde_control_120 (recommended track) at +45% peak RSS. That is the stopped scale-on-delay try's delay-inflated fronts (section 5).

## Follow-ups

1. **Re-tune scale-on-delay on the new tree.** That means the ratio, and possibly a trigger in flops instead of eligible × rows². The two recommended-track losses in section 5 come from there.
   - *2026-10-30:* the ratio alone does not do it. Ratios 8, 6 and 4 fix pde_control (2.77 s → 1.97, 1.53 and 0.99 s), but they start rescaling cops_elec_200 (1.6x → 2.7x behind MA97) and case300 (+33%).
   - The ratio stays at 16, and the cops_polygon_200 replay step is accepted. See section 4 of `2026-10-30-branches-and-scoreboard.md`.
2. **TPP pivot search in wide, delay-heavy fronts.** Merged fronts process delay cascades at half the work units per second of the old tree's. Failed pivot attempts are not counted as work and look quadratic in the eligible columns. A profile of pde_control_250 step 3 (bench_profile `--sample factor`) would show it.
3. **A delay-aware cost term.** Treating expected delays as extra pivots and rows of a front would generalize the zero-diagonal guard, which only knows certain delays.
4. **In-place assembly of the last child** (task item 2, not done). Fronts are assembled at their factor offset, and CBs live on a separate packed stack. Overlaying them needs a full-square CB layout or a front stack. After this rule c-big runs at ~60 GF/s against 11, so the measured gain left is smaller than when it was proposed.

## Reproduce

Run from the main checkout, where the fixtures live. Records are in `benchmarks/results/amalg_2026-10-29/`.

Interleaved A/B on one binary; the rule is picked by environment:

```bash
python tools/bench/ab_run.py --manifest benchmarks/manifests/fronttier.json --split tuning --rounds 3 --output <dir> --binary <build>/benchmarks/Release/bench_replay.exe --variant smf=SMF_BENCH_AMALGAMATION=smf --variant cost=SMF_BENCH_AMALGAMATION=cost
```

Per case and per class B/A; `--metric` is factor, wall, solve or analyse:

```bash
python tools/bench/ab_compare.py --a <dir>/smf --b <dir>/cost --manifest benchmarks/manifests/fronttier.json --split tuning --metric factor
```

IPOPT KKT replays, failing steps per model:

```bash
python tools/bench/replay_gate.py --binary <build>/benchmarks/Release/bench_replay.exe --variant smf=SMF_BENCH_AMALGAMATION=smf --variant cost=SMF_BENCH_AMALGAMATION=cost
```

In-process quick look:

```bash
<build>/benchmarks/Release/bench_profile.exe --manifest <m> --solvers smf --amalg smf|cholmod|cost -v
```
