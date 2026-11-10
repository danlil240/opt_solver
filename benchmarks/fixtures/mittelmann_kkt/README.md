# Mittelmann NLP KKT captures

IPOPT KKT sequences recorded by the `smf_ma97` plugin on problems of Mittelmann's AMPL-NLP benchmark.
Manifest: `benchmarks/manifests/mittelmann_kkt.json`. The matrix files derive from Mittelmann's models and are
**not committed** (this folder ignores everything but this file); they live on the i9-12900K under
`C:\smfkit\smfbench\benchmarks\fixtures\mittelmann_kkt\`.

| case | n | nnz (lower) | factorizations | size | what it shows |
|---|---:|---:|---:|---:|---|
| `qcqp1500-1c` | 22,016 | 649,570 | 100 | 520 MB | delay cascade from iteration 70: slack diagonals 1e8..1e16 (see `docs/performance/2026-11-11-qcqp1500-delay-cascade.md`) |

Layout per case, as the other captured sequences (`ipopt_hard`, `ipopt_kkt`):
`<case>/<case>.mtx` is the pattern with the first factorization's values (Matrix Market, lower triangle, entries in
column-major order) and `<case>/<case>.mtx.vals` holds every factorization's values in that order as raw doubles
(`nnz` per factorization).

KKT layout of `qcqp1500-1c` (IPOPT's augmented system): x 0..1499 (dense Hessian block, 437,489 off-diagonal
entries), slacks s 1500..11507 (diagonal Sigma, one entry -1 to their multiplier), equality multipliers
11508..12007 (zero diagonal, about 16 Jacobian entries each), inequality multipliers 12008..22015 (zero diagonal,
about 17 Jacobian entries each).

## Recreate a capture

On the 12900K, with the NLP kit in `C:\smfkit\nlp` (`nl\<problem>.nl` made by
`benchmarks/fixtures/mittelmann_nlp/mod2nl.py` from Mittelmann's models):

```bat
cd /d C:\smfkit\nlp
set SMF_PLUGIN_STATS=2
set SMF_KKT_DUMP_DIR=C:\smfkit\runs\qcap1\kkt
set SMF_KKT_DUMP_MAX=200
mkdir %SMF_KKT_DUMP_DIR%
C:\smfkit\python\python.exe nlp_run.py --config cfg_main.json --only smf8 --problems "qcqp1500-1c" --nl-dir C:\smfkit\nlp\nl --out C:\smfkit\runs\qcap1\records.jsonl
```

Run one problem and one solver configuration per capture (a second process would append to the same files).
`kkt\kkt_s0.mtx` and `kkt\kkt_s0.vals` are then copied to `<case>\<case>.mtx` and `<case>\<case>.mtx.vals` here.
The plugin's per-factorization trace (delays, factor entries, time, scaling answer) is in
`logs\smf8\<problem>.log` next to the records.

## Replay

Every recorded factorization, with the per-step lines on stderr:

```bat
bench_replay.exe --case qcqp1500-1c --mtx <case>.mtx --mtype indefinite --seq-values <case>.mtx.vals ^
  --mode warm_sequence --track ipopt-plugin --solver smf --threads 8 --seq-steps all --print-steps
```

`--track ipopt-plugin` runs smf as the plugin does and emulates IPOPT's dynamic MA97 scaling (its MC64 requests);
rivals run on `--track ipopt` (MA57 without scaling) or `--track recommended` (MA57 with its own scaling, as in
Mittelmann's IPOPT options). `SMF_PLUGIN_MC64=never` restores the plugin's unit answer to every MC64 request that
follows a rejected matching ordering (the behaviour before the cascade rule).
