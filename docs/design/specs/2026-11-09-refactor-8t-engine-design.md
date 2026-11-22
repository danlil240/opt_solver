# smf 8-thread refactorization engine design

Date: 2026-11-09. Branch: `feat/engine-8t` (from `main` `0a7e8d6`).
Status: design approved in conversation (approach A, Sections 1-4), spec waiting for Daniel's review. Nothing implemented.

## Goal

On the 12900K at 8 threads, smf's projected full-run factorization time beats MKL PARDISO (8 threads) on **each** of the 8
cases of the PR #64 probe. This is piece 1 of "best sparse solver first, then best optimizer" (Daniel, 2026-11-08).

**Measure:** the PR #64 projection, per case, the same as it was measured there:
analyse + the replayed factorizations + (N - replayed) x median refactor, where N is the number of factorizations
in the case's real IPOPT run.

**What Daniel set:**
- the success bar: win all 8, not just the aggregate;
- approach A: staged fixes on the current engine, not a new engine for the top of the tree and not static pivoting.

**Assumptions (stated by me 2026-11-08, not contested):**
- no loss at 1 or 16 threads;
- no loss of accuracy;
- accurate threshold pivoting stays the default, and there is no static pivoting by default;
- deterministic mode keeps working;
- CPU only and MIT licence.

**Out of scope:**
- analyse time (1-6% of these runs);
- the qcqp1500-1c loss from the NLP run;
- IPOPT itself (robot_a/b/c failures, AMPL evaluation cost).

## Where the time goes today

### PR #64: per-case ratios

8-thread projected run time, smf / PARDISO; 12900K, PARDISO with the probe's iparm.

| case | N | smf ms | PARDISO ms | ratio | smf refactor 1T -> 8T ms | PARDISO refactor 1T -> 8T ms |
|---|---:|---:|---:|---:|---|---|
| pglib case2000_goc__sad | 75 | 1005 | 603 | 1.67 | 12.6 -> 13.0 | 23.0 -> 4.8 |
| acopf_grid_80 | 25 | 243 | 163 | 1.49 | 19.2 -> 9.0 | 23.0 -> 5.3 |
| acopf_grid_160 | 24 | 1153 | 787 | 1.46 | 139.4 -> 44.7 | 159.6 -> 27.3 |
| obstacle_surface_200 | 136 | 814 | 693 | 1.17 | 13.6 -> 5.9 | 20.7 -> 4.8 |
| pde_control_120 | 13 | 93 | 84 | 1.10 | 11.6 -> 6.0 | 15.6 -> 4.2 |
| pde_control_250 | 12 | 462 | 434 | 1.07 | 89.1 -> 32.8 | 93.2 -> 21.4 |
| pglib case1354_pegase | 46 | 286 | 298 | 0.96 | 5.8 -> 5.9 | 10.9 -> 4.1 |
| portfolio_20k | 17 | 133 | 138 | 0.96 | 6.8 -> 7.0 | 5.5 -> 4.7 |

### Front trace (2026-11-09)

Data: 12900K, branch `probe/front-trace-8t`, results in `benchmarks/results/remote/ftrace8t`, `seg8t`, `mw0_noauto`.

**1. PGLib and portfolio run on one thread at 8T.**
- Cause: `kTreeParallelMinWork = 2e7` in `thread_plan.hpp` is a hard floor, and the planner prices flops (f^2 p) only.
  PGLib's work is many tiny fronts whose time is per-front overhead.
- Forced to 8 threads (floor 0, `auto_threads` off) the existing engine gives, with inertia identical and residuals
  equal or better:

  | case | today ms | forced 8T ms | speedup |
  |---|---:|---:|---:|
  | case1354 | 5.98 | 2.69 | 2.2x |
  | case2000 | 12.8 | 5.75 | 2.2x |
  | portfolio | 6.9 | 5.6 | 1.2x |

- With `auto_threads` on, the planner picks 2-4 threads (4.12 / 6.56 / 6.23 ms).
- PR #64's claim that delays serialise PGLib was wrong.

**2. The other five cases spend 20-34% of an 8-thread refactorization in serial work around the tree.**
- acopf has 1-4k delays per factorization (acopf_grid_80 969, acopf_grid_160 3994). That is too few to serialise
  anything, but the factor's size still changes from one factorization to the next.
- The other three cases have no delays.
- Median 8T refactor split, ms:

  | case | total | subtrees | top | gather | rest (serial) |
  |---|---:|---:|---:|---:|---:|
  | acopf_grid_160 | 43.59 | 10.85 | 22.08 | 3.42 | 7.25 |
  | acopf_grid_80 | 8.71 | 1.78 | 4.41 | 0.61 | 1.91 |
  | obstacle_surface_200 | 6.08 | 1.52 | 2.60 | 0.56 | 1.39 |
  | pde_control_120 | 5.83 | 1.41 | 2.55 | 0.36 | 1.51 |
  | pde_control_250 | 31.91 | 7.85 | 12.57 | 2.55 | 8.94 |

- The serial "rest" breaks down as follows, ms:

  | case | map + scale | engine setup | gather | post-gather bookkeeping |
  |---|---:|---:|---:|---:|
  | acopf_grid_160 | 3.34 | 0.50 | 3.34 | 2.13 |
  | acopf_grid_80 | 0.82 | 0.08 | 0.71 | 0.53 |
  | obstacle_surface_200 | 0.31 | 0.09 | 0.48 | 0.73 |
  | pde_control_120 | 0.27 | 0.16 | 0.40 | 0.84 |
  | pde_control_250 | 2.32 | 1.13 | 2.40 | 4.08 |

  pde_control_250 also has pre-map 0.30, factor-keep setup 0.36 and return 0.34 ms.

- The gather copies 70-100 MB of factor (acopf_grid_160, pde_control_250) at close to the machine's memory bandwidth.
  Only not copying removes it.

**3. The top of the tree is one chain and scales 1.2-2.7x on 8 threads.**
- The critical path is 0.94 of the 8T top span in all five cases.
- Speedup of the top part (1T work / 8T span):

  | case | speedup |
  |---|---:|
  | acopf_grid_160 | 2.67x |
  | acopf_grid_80 | 1.71x |
  | obstacle_surface_200 | 1.33x |
  | pde_control_120 | 1.20x |
  | pde_control_250 | 2.11x |

- Kernel time summed over the top fronts on acopf_grid_160, ms, 1T -> 8T:

  | kernel | 1T | 8T | runs on |
  |---|---:|---:|---|
  | assembly | 7.92 | 7.99 | one thread |
  | panel | 6.85 | 6.78 | one thread |
  | deferred update | 15.34 | 6.21 | BLAS |
  | Schur update | 28.98 | 14.03 | BLAS |

  So about 15 ms of the ~21 ms critical path is serial.
- Some fronts on the critical path get 2 BLAS threads. For example, a 380 MFlop front takes 7.78 ms at 1T and
  5.37 ms at 8T.
- The share is fixed when the front starts, at free threads / ready fronts (`top_scheduler.cpp`).

## Design

Two stages with seven parts between them. Each part is a separate unit with its own on/off control and its own A/B
test, and is its own PR. Every part does nothing at 1 thread, except where noted.

### Stage 1: the work around the tree

#### Part 1: thread planner priced in time

Files: `parallel/thread_plan.hpp/.cpp`, the planning block of `multifrontal_serial.cpp`, `control.hpp`,
`workspace.hpp` (`ThreadPlanCache`), `tools/bench/fit_thread_costs.py`.

**Front cost.** Each front's work in the tree view becomes

    w = f^2 p + c_asm f^2 + c_front

(flops, assembly, fixed overhead), in the existing work units.
- `c_asm` and `c_front` become new `Control` fields: `thread_cost_per_front_entry` and `thread_cost_per_front`.
- They are fitted on the per-front 1T times of the front trace, all 8 cases plus the board's ipopt track.
- The existing per-thread, per-top-front and split-front costs are refitted in the same pass.

**Realized sizes.**
- The engine records each front's realized row count and pivot count, delays included, in the workspace.
- A refactorization of the same analysis re-plans from those sizes when the tree's total realized cost differs from
  the planned one by more than 10%.
- Otherwise the cached plan stands (`ThreadPlanCache` gains the realized-cost key).

**Floor.**
- `kTreeParallelMinWork` no longer gates the thread count or the scheduling partition; `plan_is_serial` uses the
  costs alone.
- The floor stays, under its own name, for the delay-budget regions only. That path is opt-in (`delay_budget = -1`
  by default), and its regions decide numerics, so they must not move.

**Target:** the planner picks 8 threads for both PGLib cases (or whatever the forced-thread sweep shows is fastest) and
keeps 1 thread for small trees where 8 is slower. Projected ratios:

| case | projected ratio |
|---|---:|
| case1354 | about 0.46 |
| case2000 | about 0.75 |
| portfolio | about 0.80 |

#### Part 2: factor written in place, no gather

Files: new `include/smf/factor_store.hpp` and `src/factor_store.cpp`, `FactorKeep` in `factor_posdef.hpp`,
`multifrontal_serial.cpp` (passes, phase 3), and the factor's readers.

**`FactorStore<T>`.**
- One reserved virtual address range, cut into slices, one per thread of the factorization.
- Memory is committed in chunks as a slice grows (Windows `VirtualAlloc` reserve and commit; POSIX `mmap`).
  Pages nobody writes are never backed by RAM.
- Each slice's address space is sized so that any one thread could hold the whole factor: 4x the predicted factor plus
  the largest front squared. Address space is cheap: 16 slices of a 4 GB factor reserve about 260 GB of the 128 TB
  available.
- If a slice still runs out (delays more than quadrupled the factor), that slice moves to a private reservation and
  the factorization ends with today's copy into one contiguous range ("regather"), counted in the stats. There is no
  restart. The next factorization reserves from the realized size.
- After a factorization, a slice whose committed memory exceeds its use by more than 25% (and more than 16 MB)
  releases the excess.

**What uses it.**
- `FactorKeep::factor_values` (double) and `solve_row_indices` (Int) become `FactorStore`s.
- Each thread's pass appends its fronts to its own slice, as it appends to its private buffer today. The pass buffer
  *is* the final storage. A front is still assembled and factored in place at its slice's tail.
- Offsets written to `factor_col_ptr` and to solve steps are absolute from the store's base. The relocation loop and
  `gather_copy` go away.
- The factor has gaps between slices. Nothing may assume it is contiguous.

**Interface.** The store provides:
- `data()`;
- `size()`, the extent: end of the highest used byte;
- `empty()`, `clear()`, and `assign(n, 0)` for the old non-lean paths (`factor_indef.cpp`, `factor_posdef.cpp`);
- iteration over the used ranges, for the determinism hash and the diagnostics;
- move. A copy keeps every offset.

Readers that use `data()` plus an offset (the solves) don't change. Readers that use `size()` as an entry count use a
new `used()`.

**One thread.** One slice, exactly today's serial path.

**Fallback.** If the reservation fails, every slice is private and the factorization regathers, which is today's
per-thread buffers plus the gather. That path
stays, and a test switch forces it.

**Target:**
- gather 0;
- peak RSS lower by about one copy of the factor (today's thread buffers plus the gathered array);
- saves 2.4-3.4 ms per factorization on acopf_grid_160 and pde_control_250.

#### Part 3: solve preparation off the serial path

Files: `solve_diag.cpp` (`finish_solve_layout`, `build_inline_dinv`), the post-gather block of
`multifrontal_serial.cpp`, `TreeParallelBuffers`, `smf_ma97_plugin.cpp`, `pivot_order.cpp`, `solve.cpp`.

This part starts by timing the post-gather sub-steps at 8T on the 8 cases. A sub-step under 0.1 ms on every case is
left alone. The expected changes:
- **D^-1 per front.** Each front's D^-1 entries (`dinv`, 3 per pivot) are computed right after its kernel, by the
  thread that factored it, into a third slice store. Each step carries its absolute `dinv_offset`. The serial pass
  over all pivots goes away.
- **Fixed order for the top of the solve.**
  - The top part of the solve takes its steps in the postorder of the top fronts instead of completion order.
  - `solve_top_parent` and the top step positions then depend only on the analysis and the partition, so they are
    cached.
  - `solve_top_rows` / `solve_top_slot` and the solve's thread plan are rebuilt only when the realized row lists of
    the top steps differ from the cached ones (a sequential compare). Otherwise they are reused.
- **`diag_entries` on demand.**
  - Built on first read (`pivot_order.cpp`, the non-inline solve, `solve.cpp`).
  - The plugin's count of 2x2 pivots comes from a new per-factorization counter in the stats instead of a scan of
    `diag_entries`.
- **Kept per factorization:** concatenating the per-thread step lists (O(fronts), small structs) and ordering the
  subtrees by work.

**Target:** post-gather bookkeeping under 0.5 ms on every case. Today it is 0.5-4.1 ms.

#### Part 4: value preparation in parallel, and per-analysis setup reuse

Files: `workspace.cpp` (`map_values_to_permuted`), `factor_indef.cpp` (scale loop), `multifrontal_serial.cpp` (setup
block).

**Map and scale.**
- `map_values_to_permuted` and the fused scale-and-scatter loop run over column ranges balanced by entry count, on the
  factorization's pool, when there are at least 2 threads and at least 1e5 entries (threshold to be swept).
- The one-to-one scatter (`direct`) writes each permuted slot exactly once, so there is no race. Each thread keeps
  its own max |value| and the results are reduced at the end.
- The scatter with duplicates (`+=`) stays serial.
- Results are bit-for-bit those of the serial pass.

**Setup reuse.**
- This part starts by timing the engine setup and factor-keep setup sub-steps.
- Work that is provably unchanged since the last factorization into the same `FactorKeep` with the same analysis is
  skipped: permutation copies, buffer resets, and vectors re-sized to the same size. The check is the analysis
  serial, stored in the `FactorKeep`.
- Each skip has a test that a different analysis does reset it.

**Target:**
- map + scale: 3.3 -> under 1 ms on acopf_grid_160, 2.3 -> under 0.7 ms on pde_control_250;
- setup: under 0.3 ms on every case.

#### Stage 1 projection

These are estimates, not measurements.

| case | today | after Stage 1 |
|---|---:|---:|
| case1354 | 0.96 | about 0.46 |
| case2000 | 1.67 | about 0.75 |
| portfolio_20k | 0.96 | about 0.80 |
| obstacle_surface_200 | 1.17 | about 0.86 |
| pde_control_120 | 1.10 | about 0.86 |
| pde_control_250 | 1.07 | about 0.75 |
| acopf_grid_80 | 1.49 | about 1.09 |
| acopf_grid_160 | 1.46 | about 1.14 |

### Stage 2: the top of the tree

Stage 2 targets the serial ~15 ms of acopf_grid_160's ~21 ms critical path. The parts are ordered by that share.

#### Part 5: parallel assembly of large top fronts

Files: `process_front` in `multifrontal_serial.cpp`, `TopScheduler` helper offers (`top_scheduler.hpp`).

A top front with at least `f_min` rows (to be swept; start at 1024) and a BLAS share of at least 2 assembles in
parallel:
- the front's columns are split into blocks;
- each helper takes a block and does, for it, the zero fill, the original entries and every child's extend-add in
  the usual child order.

Each entry receives the same additions in the same order as today, so results are bit-for-bit identical at every
thread count. That also holds in deterministic mode, where the part may run.

Helpers come through the scheduler's existing `offer`/`withdraw` mechanism, the one tiled fronts already use, so idle
pool threads do the work.

**Target:** assembly on acopf_grid_160's critical path 7.9 -> about 2 ms.

#### Part 6: lookahead in the TPP kernel

Files: `ldlt_kernel.cpp` (blocked path of `ldlt_factor_front`).

This applies to large top fronts only, in non-deterministic mode, at a share of 2 or more threads. Deterministic mode
and every other front keep today's kernel.

**Order of work.** After panel k is factored and its W is built:
1. Panel k's deferred update is applied to the next panel's columns first.
2. Then, concurrently:
   - one thread searches and factors panel k+1;
   - the other threads run panel k's deferred update on the remaining eligible columns, plus panel k's chunk of the
     Schur update.

**Schur update in chunks.** The Schur update, today one lower GEMM over all pivots after the last panel, is done in
per-panel chunks so it can overlap.
- This changes the order in which each Schur entry is summed. Rounding differs from today; pivots do not.
- On the 8 cases the gate checks residuals and inertia.

**Row interchanges.**
- Panel k+1's symmetric interchanges touch rows of the columns being updated concurrently.
- They are recorded and applied to those columns after the concurrent update finishes, as LAPACK does with deferred
  interchanges.
- The pivot test sees fully updated panel columns, so pivot choices are those of today's kernel.

**Stall.** When panel k+1 stalls and must search outside the panel, it waits for the concurrent update and takes
today's path.

**Target:** visible panel time on acopf_grid_160's critical path 6.9 -> about 2.5 ms. The panel is hidden behind the
deferred update (about 2.5 ms at 7 threads) plus the Schur chunks (about 5 ms).

#### Part 7: thread shares that follow the critical path

Files: `parallel/top_scheduler.hpp/.cpp`, the BLAS share plumbing (`blas_thread_guard.cpp`).

- **Priority.** Each top front gets a priority: its remaining path cost to the root, from Part 1's costs and realized
  sizes, computed once per plan. Ready fronts are taken highest priority first, instead of last-in first-out.
- **Share at start.** A front taken while others are ready gets threads in proportion to its priority among the ready
  fronts, at least 1 each.
- **Growing share.** A running front's share is an atomic. Threads freed when another front finishes go to the
  running front with the highest priority, unless a ready front needs one. The running front's next BLAS call uses
  the new share.
- **Continuation** (the thread that readies a parent runs it) stays.

**Target:** deferred-update and Schur speedup on the acopf_grid_160 critical path about 6x at 8 threads instead of
about 2x.

#### Stage 2 projection

Also estimates.

| case | after Stage 2 |
|---|---:|
| acopf_grid_80 | about 0.91 |
| acopf_grid_160 | about 0.94 |

The other six stay at or below their Stage 1 values.

## Testing, switches and fallbacks

**Switches.**
- Every part has a `Control` flag and an `SMF_*` environment override, so a single binary can A/B it. This follows
  the PR #55 lesson: one binary for layout-sensitive A/B.
- A flag defaults to on only after its part's A/B. The flag stays as the off-switch.

**Fallbacks.**

| condition | what happens |
|---|---|
| slice reservation fails | per-thread buffers plus gather (Part 2) |
| top row lists change | rebuild the solve layout (Part 3) |
| panel stalls | synchronous path (Part 6) |
| factorization aborts (inertia limit, work limit, error) | clear the layout and plan caches |

**Unit tests.**

| part | checks |
|---|---|
| 1 | tiny trees still plan 1 thread; a PGLib-shaped tree plans 8; re-plan on a realized-size jump; budget regions unchanged |
| 2 | slice growth and commit; forced fallback; forced reservation failure; offsets and solve correct with gaps; a moved and a copied `FactorKeep` still solve |
| 3 | layout cache reused on an unchanged sequence, rebuilt across a PGLib delay sequence; `diag_entries` on demand equals today's; 2x2 counter equals the scan |
| 4 | parallel map/scale bit-for-bit equal to serial; setup skips reset on a new analysis |
| 5 | parallel assembly bit-for-bit equal to serial, at 2/4/8 helpers |
| 6 | same pivot sequence and inertia as today's kernel on the adversarial, delayed-pivot and 2x2-pairing tests; stall path covered |
| 7 | priorities on a synthetic chain; share growth; no oversubscription (sum of shares <= threads) |

**Existing suites.** Every existing suite must pass, in particular the parallel determinism, tree-parallel,
repeated-factor and IPM session suites.

**Benchmarks.**
- Runs go to the 12900K via the remote scripts, one at a time, after checking the machine is idle.
- **Per part:** A/B on the 8 cases (`bench_replay`, the PR #64 sequences) with peak RSS, plus that part's own target
  above.
- **Before Stage 1:** a fresh board baseline (1T/8T/16T) at the branch base.

**Gates.**

Stage 1 gate, after Parts 1-4 are merged:
- **The 8-case projection against PARDISO 8T.** PARDISO is re-measured once on the same day to guard against drift.
  Pass: case1354, case2000, portfolio_20k, obstacle_surface_200, pde_control_120 and pde_control_250 all below 1.00.
  The acopf ratios are recorded.
- **Board at 1T, 8T and 16T against the baseline.** Pass:
  - each track's geomean no more than 0.5% worse;
  - no single case more than 5% slower on a rerun.
- **Accuracy.** Pass:
  - every case's inertia identical;
  - validation outcome identical;
  - no backward error more than 2x worse than baseline on a case that passes today.
- **Peak RSS on the 8 cases.** Pass: no higher than baseline.
- **IPOPT end-to-end on the 8 cases.** Pass:
  - same termination;
  - identical iteration counts, or a difference traced to rounding and not to a change of pivot policy before the
    gate passes.

Stage 2 gate, after Parts 5-7:
- the same checks;
- pass: all 8 cases below 1.00.

**Stop rules.**
- If after Parts 1-4 any of Stage 1's six target cases is still at 1.00 or above, I stop and re-plan with Daniel before
  Stage 2.
- If Parts 5-7 together do not cut acopf_grid_160's 8T top span by at least 30%, I stop and re-plan before more
  kernel work.

## Delivery

- Each part is its own PR from `main`, in the order 1, 4, 2, 3, then 5, 7, 6.
  - Parts 1 and 4 are the smallest and safest.
  - Part 3 stores D^-1 in Part 2's store, so it follows Part 2.
  - Part 6 is the riskiest and goes last.
- Daniel merges.
- `STATUS.md` is updated after every PR, gate and plan change.
- PR #64's doc gets the PGLib correction (the floor, not delays) in Part 1's PR.
