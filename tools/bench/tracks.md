# Benchmark tracks

`bench_replay --track` (set per case by the manifest's `"track"`, or for a
whole run by `run.py --track-override`) decides how a case is factorized.

| track | smf Control | path | solvers |
|---|---|---|---|
| `recommended` | smf defaults (`pivot_u` 0.01, AutoSerial ordering, Auto scaling, `expected_refactors` 1), `num_threads` = the run's thread count | plain `smf::Solver` adapter | all |
| `ipopt` | as `recommended` but `pivot_u` = 1e-8 (IPOPT's `ma*_pivtol` / `ma97_u` default); harness refinement after every 1-RHS solve, as IPOPT refines | plain `smf::Solver` adapter | all (each rival at IPOPT's pivot threshold; MA57 also with IPOPT's other `ma57_*` defaults: no MC64 scaling, `ICNTL(15)` = 0, where `recommended` keeps MA57's own scaling) |
| `ipopt-plugin` | the IPOPT plugin's own Control: `smf::ipopt_plugin_control()` (`solver/include/smf/ipopt_plugin_control.hpp`), the function `libsmf_ma97` maps every MA97 control block through. Under IPOPT's defaults: `pivot_u` 1e-8, AutoParallel ordering, no scaling (IPOPT's dynamic `ma97_scaling` factors unscaled first), `expected_refactors` 10, `nemin` 8, indefinite matrix type for every case (IPOPT's KKT type), `num_threads` the plugin's (`smf::plugin_num_threads()`: `SMF_NUM_THREADS`, else `OMP_NUM_THREADS`, else the performance cores; see below); same refinement as `ipopt` | `smf::IpmSession` with the plugin's `IpmControl` (`smf::ipopt_plugin_ipm_control()`, the `SMF_IPM_*` variables); analysis deferred to the first factorization, as IPOPT analyses the pattern without values; a singular factor in a replayed sequence is skipped (IPOPT rejects it and refactors) | **smf only** for now: other solvers get an `Unavailable` record from `bench_replay`, and `run.py` skips them |

`ipopt-plugin` measures what IPOPT users of the plugin get, including the
plugin-side analysis cost (e.g. stokes128: matching phase ~180 ms against
~33 ms on `ipopt`). The `SMF_BENCH_*` environment knobs and `--pivot-u`,
`--scaling`, `--ordering` still apply on top of the plugin's Control, for
experiments. With `SMF_IPM_REANALYSE=1` a re-analysis runs inside a
factorization and its time is counted in `factor_ms`, not `analyse_ms`.

**Fast mode.** `SMF_PIVOTING=static` (the IPOPT plugin's variable; the older
`SMF_BENCH_PIVOTING` wins when both are set) runs smf with
`Pivoting::Static` on every track: PARDISO-like static pivoting, opt-in, not
guaranteed accurate on hard indefinite systems, and with the inertia of the
perturbed matrix when a pivot is perturbed. It disables the delay and
scale-on-delay rescue. `environment.json` records it as `smf_pivoting_env`
(the reliable marker: records' `static_fallbacks` > 0 also comes from
`SMF_BENCH_STATIC_FALLBACK=1`). A fast-mode board is a separate
column, never a headline number. The default (unset or `threshold`) is the
accurate threshold pivoting.

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
Exception to the unit answer: a delay cascade (`smf::Ma97CascadeState`).
When the factorization before the request did at least 16 times the
delay-free front work, the request is honoured with the matching scaling,
and so is every later request of that analysis. `--print-steps` shows
`work=` (that ratio) and `mc64=` (0 no request, 1 unit answer, 2 computed,
3 computed after a cascade); records carry `mc64_cascade_scales`.
`SMF_PLUGIN_MC64=never` keeps the unit answer, `=always` honours every
request, `SMF_PLUGIN_MC64_RATIO=<x>` changes the limit.

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
solver is an error (exit 2).

Threads on this track are the plugin's, as inside IPOPT. The plugin sets
`Control::num_threads` to `smf::plugin_num_threads()`
(`solver/include/smf/ipopt_plugin_control.hpp`): `SMF_NUM_THREADS` (a
positive int), else `OMP_NUM_THREADS` (a positive int; of a list `a,b` the
first), else the machine's performance core count (below). Real MA97 has no thread
option (IPOPT has none for it either) and factors on OpenMP's default team,
so the plugin follows `OMP_NUM_THREADS` like the library it replaces. run.py
exports `OMP_NUM_THREADS` (and `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`) as
each run's `--threads` value, so an N-thread row really runs N threads
(tree parallelism, `auto_threads`, BLAS budget N, as on the `recommended`
track at N threads). `SMF_NUM_THREADS` in the environment overrides that:
run.py warns, and bench_replay warns whenever the plugin's count differs
from `--threads` (also when it runs without run.py and `OMP_NUM_THREADS` is
unset: then the performance cores) and runs and records the plugin's count.
The performance core count (`smf::performance_core_count()`) is that of the
physical cores the process affinity allows (cpusets of Docker
`--cpuset-cpus`, Slurm, `taskset`; Windows job objects and
`start /affinity`), and on a hybrid CPU only those of the top performance
class among them, the P-cores (Windows: each core's `EfficiencyClass`;
Linux: `/sys/devices/cpu_core/cpus` when `/sys/devices/cpu_atom/cpus` exists
too). On non-hybrid CPUs that is every physical core; with the P-cores
outside the affinity, the allowed E-cores. It is read once per process, like
both variables, which are taken as given (also above it). Why: on an
i9-12900K (8 P + 8 E-cores) 16 threads were slower than 8 on 51 of the 72
plugin-track cases (geomean time vs serial 0.832 at 8 threads, 0.844 at 16;
cont-300 and pde_control_250 1.16x, TSOPF_FS_b39_c30 1.12x, thermal1 1.10x
slower at 16; none more than 10% faster), so the default there is 8.

More than one thread changes the analysis as well as the speed. Under
IPOPT's default ordering (AutoParallel) large matrices are ordered by
parallel nested dissection instead of serial METIS, and the matching trial
runs concurrently. Orderings, delays and fill, and so IPOPT's iterates, can
therefore differ from the old serial plugin (`num_threads` 1) and depend on
the thread count. `SMF_NUM_THREADS=1` restores the serial plugin;
`Control::deterministic` makes results independent of the thread count.
Several IPOPT instances on one machine (multistart, parallel batches) now
take all performance cores each unless `SMF_NUM_THREADS` / `OMP_NUM_THREADS`
splits them.

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
