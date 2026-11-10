# qcqp1500-1c: a delay cascade the plugin refused to rescale (2026-11-11)

**Question.** On Mittelmann's `qcqp1500-1c`, IPOPT + smf spent 120 s in the linear solver against MA57's 36 s
at the same 99 iterations (4.9x at 1 thread). Is it factor size, delayed pivots, a repeated rescue, or a
re-analysis?

**Answer.** Delayed pivots, in the last 30 of 100 factorizations only. IPOPT notices and asks for MC64 scaling
four times; the plugin answered every request with the unit scale because the analysis had rejected the matching
ordering. With the request honoured after a real cascade, the linear-solver time is **120.4 s -> 4.8 s** at 8
threads (9.4 s at 1 thread; MA57 36.3 s the same day), with the same 99 iterations.

All runs: i9-12900K, P-cores, IPOPT 3.14.20. Data: `benchmarks/results/remote/qcqp1500_cascade/`
(`qcap1` capture run, `qrep1` and `qrep2` replays, `qipopt1` IPOPT confirmation, `casc_*` regression runs).

## 1. What the statistics say

One IPOPT run with main's plugin (`63bdc35`), 8 threads, `SMF_PLUGIN_STATS=2` and the KKT capture on:

| IPOPT timer | seconds |
|---|---:|
| LinearSystemSymbolicFactorization | 0.011 |
| LinearSystemFactorization | 118.9 |
| LinearSystemBackSolve | 1.1 |

One analysis (144 ms at 8 threads in the replay, counted by IPOPT inside its first factorization), no re-analysis,
no rescue. The ordering is AMD throughout. Per factorization (plugin trace):

| factorizations | delayed pivots | factor entries | flops | time each (8T) |
|---|---:|---:|---:|---:|
| 1-70 | 498 | 4.36M | 2.7e9 | 0.02-0.03 s |
| 71 / 72 / 73 | 730 / 1150 / 1696 | 5.3M / 7.4M / 10.6M | 3.9e9 .. 1.1e10 | 0.04 / 0.07 / 0.12 s |
| 74 / 75 / 77 | 3042 / 4861 / 6256 | 21M / 41M / 61M | 2e10 .. 1.6e11 | 0.5 / 1.3 / 2.3 s |
| 80-92 | 10,494 | 144M | 5.8e11 | 6.4-7.3 s |
| 93-100 | 9048 .. 2285 | 112M .. 15M | 3.9e11 .. 1.8e10 | 5.6 .. 0.27 s |

The first 70 factorizations take 1.6 s in total, the last 30 take 116.7 s.

## 2. Why the pivots are delayed

The KKT matrix (n = 22,016, 649,570 lower entries) has four blocks: x (1500, a 39% dense Hessian), slacks s
(10,008, diagonal Sigma, one entry -1 to their multiplier), equality multipliers (500, zero diagonal) and
inequality multipliers lambda (10,008, zero diagonal, about 17 Jacobian entries of size ~3.5 each).

AMD eliminates each slack s_i, then its multiplier lambda_i, and leaves the x block as one dense top front. That
is the cheapest order: 2.7e9 flops. After s_i the pivot of lambda_i is -1/Sigma_i. The 498 delays of the early
factorizations are the equality multipliers (zero diagonal, eliminated in the top front): harmless.

From iteration 70 the slack diagonals grow, and they keep growing (median Sigma: 3e3 at factorization 70, 2.5e6
at 74, 4.7e8 at 77, 4.7e11 at 81, 7.6e13 at 91). Once Sigma_i exceeds about 3e7, |-1/Sigma_i| is below
u x 3.5 = 3.5e-8 and lambda_i fails the threshold test (u = 1e-8, IPOPT's `ma97_u`). It is delayed, and its parent
is the dense top front, which grows from 2000 rows to 12,000. The count of multipliers that fail this test in the
captured values equals the measured delays at every sampled factorization (2545 at 74, 5761 at 77, 10,000 at 81).

So the fill is not an ordering fault: the order is the right one, the unscaled matrix just cannot be pivoted in it
at u = 1e-8.

## 3. Why nothing rescued it

IPOPT's default MA97 strategy (`ma97_scaling = dynamic`, `ma97_switch1 = od_hd_reuse`) asks for MC64 scaling when
a factorization's delays exceed the last scaled factorization's by 5% of n. It did, at factorizations 73, 75, 78
and 80. The plugin answered all four with the unit scale: since the PGLib triage, an MC64 request that follows a
rejected matching ordering gets no scaling, because a matching scale under an AMD order made PGLib case2869 and
case4661 inaccurate and costs about 10 refactorizations each time. smf's own scale-on-delay rescue is never armed
at u = 1e-8 or under MA97's scaling = none.

MA57 in Mittelmann's IPOPT options runs with `ma57_automatic_scaling=yes`, so it never sees the problem.

## 4. Replay of the capture

`bench_replay --seq-steps all` on the 100 captured matrices (fixture `mittelmann_kkt/qcqp1500-1c`, manifest
`benchmarks/manifests/mittelmann_kkt.json`). The `ipopt-plugin` track emulates IPOPT's MC64 requests. Total
factorization time over the 100 steps:

| solver | threads | total | first 70 | last 30 | largest factor | accurate |
|---|---:|---:|---:|---:|---:|---|
| smf, unit answer (main's behaviour, `SMF_PLUGIN_MC64=never`) | 8 | 118.3 s | 1.6 s | 116.7 s | 144.3M | yes |
| smf, unit answer | 1 | 183.8 s | 4.7 s | 179.1 s | 144.3M | yes |
| **smf, cascade rule (this PR)** | 8 | **3.5 s** | 1.6 s | 1.9 s | 21.0M (one step) | yes |
| **smf, cascade rule** | 1 | **8.5 s** | 4.7 s | 3.8 s | 21.0M | yes |
| smf, every request honoured (`SMF_PLUGIN_MC64=always`) | 8 | 4.0 s | 1.6 s | 2.3 s | 17.9M | yes |
| MA57, own scaling (`recommended` track, as Mittelmann's options) | 1 | 36.0 s | 24.4 s | 11.6 s | 2.5M | yes |
| MA57, no scaling (`ipopt` track) | 1 | 254 s | 6.9 s | 247 s | 72.2M | no (13 steps, wrong inertia at 81) |
| MKL PARDISO | 8 | 11.9 s | 8.3 s | 3.6 s | n/a | no (20 steps, backward error 1) |
| MKL PARDISO | 1 | 72.6 s | 50.7 s | 21.9 s | n/a | no (20 steps) |

The replay reproduces IPOPT: 118.3 s against IPOPT's 118.9 s for smf at 8 threads, 36.0 s against 35.7 s for MA57.

Factor size against MA57: smf's delay-free factor has 4.36M entries and 2.7e9 flops, MA57's 2.37-2.50M entries.
smf's is larger because the 498 equality multipliers join the top front (2000 rows instead of 1500). It does not
cost time: smf needs 21-25 ms per factorization at 8 threads and about 67 ms at 1 thread, MA57 120-420 ms, PARDISO
117 ms at 8 threads. The whole loss was the cascade.

Accuracy with the rule: every step passes the gate, worst backward error after the harness' 3 refinement steps
7e-11 (9e-11 at 1 thread), the same as without it (6e-11 / 8e-11). Inertia 11508 / 10508 / 0 at every step.
Without refinement the rule is the more accurate of the two on this capture: worst backward error 9e-9 (all 100
steps pass the gate unrefined), against 6.5e-3 for the cascade factorizations (12 steps fail it unrefined). After
one refinement step the rule is at 6e-11.

That is not general. A scaled factor that pivots a multiplier right after its slack forms H + J^T Sigma J, and its
unrefined componentwise backward error is about eps x Sigma on rows where H still matters. The plugin test's
synthetic KKT (1200 of 4960 slack diagonals at 1e12, the rest at 1) shows it: 3e-4 unrefined with the matching
scale against 3e-15 for the delayed factor, 1e-11 to 3e-9 after three refinement steps. IPOPT refines every solve, and
MA57 with scaling pivots the same way.

With the rule, IPOPT's first request (factorization 73) still gets the unit scale: the factorization before it did
5.4x the delay-free front work. The second request (75) follows one at 27x and is honoured: 0.57 s for the matching,
after which the delays are back at 498 and stay there to the end on that one scale (IPOPT reuses it, and never
asks again).

## 5. The fix

`solver/include/smf/ipopt_plugin_control.hpp`, `Ma97CascadeState` and `ma97_scaling_call()`; used by the plugin
(`ma97_factor_d`) and by `bench_replay`'s `ipopt-plugin` track.

- The plugin remembers, per analysis, the front work of the latest factorization over the delay-free work
  (`Info::front_work` / `delay_free_front_work`).
- An MC64 request after a rejected matching ordering is honoured when that ratio is at least
  `kPluginCascadeWorkRatio` = 16 (the limit `Control::scale_on_delay_ratio` uses: easy cases stay below 9,
  cascades start at 28), and from then on for every later request of that analysis. Otherwise it gets the unit
  scale as before.
- `SMF_PLUGIN_MC64=never` restores the old answer, `=always` honours every request (as before),
  `SMF_PLUGIN_MC64_RATIO=<x>` changes the limit for experiments.
- The plugin trace (`SMF_PLUGIN_STATS=2`) shows `work=<ratio>(prev <ratio>)` and `scale=out-cascade` for a request
  honoured this way; replay records carry `mc64_cascade_scales`.

IPOPT's request alone is not enough (PGLib asks too and is better off unscaled), and the work ratio alone is not
acted on (the plugin never scales unasked). Both together mean: the delays jumped, and they are expensive.

## 6. IPOPT, same day (2026-11-11)

99 iterations and "Optimal Solution Found" in every run.

| problem | solver | linear solver | factorization | wall |
|---|---|---:|---:|---:|
| qcqp1500-1c | smf 8T, main | 120.4 s | 118.9 s | 187.1 s |
| | **smf 8T, this PR** | **4.8 s** | 4.1 s | **71.7 s** |
| | smf 1T, this PR | 9.4 s | 8.9 s | 77.0 s |
| | MA57 | 36.3 s | 35.7 s | 105.4 s |
| qcqp1000-2c | smf 8T, main | 21.3 s | 20.9 s | 38.2 s |
| | **smf 8T, this PR** | **2.4 s** | 2.1 s | **19.3 s** |
| | smf 1T, this PR | 3.5 s | 3.3 s | 20.3 s |
| | MA57 | 16.5 s | 16.8 s | 35.9 s |

What is left of the wall time is AMPL function evaluation (63.7 s and 16.1 s), which is not ours. On qcqp1000-2c
the requests come at factorizations 78 (unit answer) and 80 (honoured).

## 7. The 1.08x slowdown of main's plugin

Not from the two changed defaults (`kIpoptExpectedRefactors` 10 -> 25, parallel front assembly). Another session
bisected it in IPOPT by building the plugin at each merge commit: it starts with #68 (factor written in place),
which moved a factor that outgrew its reservation into freshly allocated memory on every cascade factorization.
PR #74 fixes that (`docs/performance/2026-11-10-ipopt-main-plugin-check.md`). I did not repeat the bisect in the
replay. With the cascade gone the oversized factors occur on two or three factorizations per run instead of 28.

IPOPT's iteration log is the same line for line through iteration 88 with main's plugin and with the rule (14 of
those iterations run on the scaled factor); from 89 the last printed digits differ, as they do between main's
plugin and MA57. Final objective 3.88297948667e6 in all three. The plugin served 284 solves in both smf runs, so
IPOPT needed no extra refinement steps.

## 8. Regression checks

- Tests: 83/83 on `build-mtt` and `build-seq` (12900K), including the new
  `Ma97PluginRuntime.Mc64RequestAfterDelayCascadeIsHonoured` and `BenchmarkAdapters.Ma97CascadeStateRemembersACascade`.
- The 8 PR #64 probe cases, plugin track, 8 threads, 5 rounds interleaved, old answer against the rule: the rule
  never fires (`mc64_cascade_scales` 0 everywhere), same orderings and delay counts, factor time geomean 0.987
  (noise: no decision differs when the rule does not fire).
- Front tier on the plugin track (78 case-tracks, 8 threads) and `ipopt_hard` (16 sequences, every recorded
  factorization): the rule fires nowhere. The Inaccurate cases there (TSOPF_FS_b300, d_pretok, turon_m,
  cops_polygon_200) are the same with the old answer.
- qcqp1000-2c: section 6.

## 9. Not done

- The full Mittelmann set was not rerun with this plugin. Any problem whose trace shows no `scale=out-cascade`
  behaves exactly as with main's plugin; the rerun would show whether other problems cascade.
- The first request of a cascade still gets the unit scale (0.6 s lost on qcqp1500-1c). A try under a work limit
  at the request itself would catch it one request earlier.
- `cont5_2_4_l` (the other suspected laggard) was not looked at.
