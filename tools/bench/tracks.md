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
sequence. IPOPT's default `ma97_scaling = dynamic` with `ma97_switch1 =
od_hd_reuse` factors unscaled until a successful factorization delays more
than 0.05 n pivots; it then requests MC64 (`scaling = 1`) on the next
factorization and passes the returned scale back on every later one
(robot_arm trace, docs/performance/2026-10-26-solve-phase.md). The adapter
does the same, answering each call through the plugin's own per-call
decision, `smf::ma97_scaling_call()` (unit scale after a rejected matching
ordering, reuse of a passed-back scale), and the plugin's stale-scale retry.
Limits:
- The "next factorization" is the next *replayed* step. The default
  `--seq-steps 5` replays 5 evenly spaced recorded steps, so the switch
  lands later in IPOPT's iteration count than it really would. Use
  `--seq-steps all` (bench_replay) for the exact order.
- Recorded values come from the run that recorded them, whose own scaling
  history may differ from the replay's.
- Modelled: one switch, then reuse for the rest of the sequence, as in the
  robot_arm trace. Not modelled: IPOPT's on-demand ("od") quality requests
  and its later switch levels.
- Cold cases run one factorization, so they never switch. Their result is
  exact for IPOPT's first factorization.

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
