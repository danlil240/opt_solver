# Benchmark tracks

`bench_replay --track` (set per case by the manifest's `"track"`, or for a
whole run by `run.py --track-override`) decides how a case is factorized.

| track | smf Control | path | solvers |
|---|---|---|---|
| `recommended` | smf defaults (`pivot_u` 0.01, AutoSerial ordering, Auto scaling, `expected_refactors` 1), `num_threads` = the run's thread count | plain `smf::Solver` adapter | all |
| `ipopt` | as `recommended` but `pivot_u` = 1e-8 (IPOPT's `ma*_pivtol` / `ma97_u` default); harness refinement after every 1-RHS solve, as IPOPT refines | plain `smf::Solver` adapter | all (each rival at IPOPT's pivot threshold) |
| `ipopt-plugin` | the IPOPT plugin's own Control: `smf::ipopt_plugin_control()` (`solver/include/smf/ipopt_plugin_control.hpp`), the function `libsmf_ma97` maps every MA97 control block through. Under IPOPT's defaults: `pivot_u` 1e-8, AutoParallel ordering, no scaling (IPOPT's dynamic `ma97_scaling` factors unscaled first), `expected_refactors` 10, `nemin` 8, indefinite matrix type for every case (IPOPT's KKT type), `num_threads` 1 (the plugin never sets it); same refinement as `ipopt` | `smf::IpmSession` with the plugin's `IpmControl` (`smf::ipopt_plugin_ipm_control()`, the `SMF_IPM_*` variables); analysis deferred to the first factorization, as IPOPT analyses the pattern without values; a singular factor in a replayed sequence is skipped (IPOPT rejects it and refactors) | **smf only** for now: other solvers get an `Unavailable` record from `bench_replay`, and `run.py` skips them |

`ipopt-plugin` measures what IPOPT users of the plugin get, including the
plugin-side analysis cost (e.g. stokes128: matching phase ~180 ms against
~33 ms on `ipopt`). The `SMF_BENCH_*` environment knobs and `--pivot-u`,
`--scaling`, `--ordering` still apply on top of the plugin's Control, for
experiments. With `SMF_IPM_REANALYSE=1` a re-analysis runs inside a
factorization and its time is counted in `factor_ms`, not `analyse_ms`.

**IPOPT's dynamic scaling is emulated** over the replayed steps of a
sequence, by the rule of IPOPT 3.14's Ma97SolverInterface under its defaults
(`ma97_scaling = dynamic`, `ma97_switch1 = od_hd_reuse`, `ma97_switch2/3 =
never`; `solver/benchmarks/common/ipopt_dynamic_scaling.hpp`):
- IPOPT factors unscaled until a factorization returns successfully with
  `num_delay - numdelay_ > 0.05 n`. Here `numdelay_` is the delay count of
  the last MC64 factorization, and 0 before the first one.
- It then requests MC64 (`scaling = 1`) on the next factorization and keeps
  the scale handed back. It records that factorization's delays as
  `numdelay_`, then passes the scale back (`scaling = 0`, reuse) on later
  factorizations, until the rule fires again. Each new firing requests MC64
  again.
- A singular factorization returns before the check, so a pending request
  stays pending.
- A failed call (plugin flag -5) hands no scale back.

The plugin's answer to each call is its own per-call decision,
`smf::ma97_scaling_call()` (unit scale after a rejected matching ordering,
reuse of a passed-back scale), followed by its stale-scale retry.

Limits:
- The "next factorization" is the next *replayed* step. The default
  `--seq-steps 5` replays 5 evenly spaced recorded steps, so requests land
  later in IPOPT's iteration count than they really would. Use
  `--seq-steps all` (bench_replay) for the exact order.
- Recorded values come from the run that recorded them, whose own scaling
  history may differ from the replay's.
- IPOPT's on-demand ("od") quality requests are not modelled.
- Cold cases run one factorization, so they never request MC64. Their
  result is exact for IPOPT's first factorization.

Per-case `"ordering"` pins in the manifest are dropped on override tracks:
the track runs the plugin's own ordering.

## Running a manifest on another track

The front-tier manifest lists most indefinite matrices twice (`recommended`
and `ipopt`). `--track-override ipopt-plugin` (the only override track;
run.py and ab_run.py reject other values) runs every selected matrix **once**
on that track, taking mode, recorded sequence and per-case settings from its
`ipopt` entry when it has one (else its first entry). Selecting no smf
solver is an error (exit 2). With `--threads` above 1, run.py and
bench_replay warn that the plugin runs smf with `Control::num_threads = 1`.
Until the plugin sets the thread count, N-thread rows on this track are
1-thread-tree rows (only BLAS is threaded).

    python tools/bench/run.py --manifest benchmarks/manifests/fronttier.json \
        --output benchmarks/results/plugin_t1 --track-override ipopt-plugin \
        --solvers smf --threads 1

    python tools/bench/ab_run.py --manifest benchmarks/manifests/fronttier.json \
        --output benchmarks/results/plugin_ab --track-override ipopt-plugin \
        --binary build/benchmarks/Release/bench_replay.exe \
        --variant base=SMF_BENCH_MATCHER=ssp --variant cand=SMF_BENCH_MATCHER=hybrid

Records carry `"track": "ipopt-plugin"`. `score.py` and `ab_compare.py` match
them to the case's override entry (class and split from the manifest), so a
self-vs-self A/B works as on the other tracks; `score.py` has no rivals to
rank against on this track, so its ratio column is n/a.
