# SPRAL SSIDS in WSL2 (benchmark competitor)

SSIDS (CPU) does not build natively on Windows, so the harness runs it inside
WSL2 Ubuntu through `solver/benchmarks/wsl/ssids_replay.c`, a small C driver with
`bench_replay`'s command line and JSONL record schema.

## One-time setup

```powershell
wsl --install -d Ubuntu --no-launch          # only if there is no Ubuntu distro yet
wsl -d Ubuntu -u root -- bash -c "apt-get update && apt-get install -y gfortran gcc g++ build-essential meson ninja-build libmetis-dev libopenblas-dev libhwloc-dev pkg-config curl ca-certificates"
wsl -d Ubuntu -u root -- bash /mnt/c/Users/<you>/projects/opt_solver/solver/benchmarks/wsl/build.sh
```

`build.sh` downloads the SPRAL release (`v2025.09.18`, override with
`SPRAL_VERSION`), builds it CPU-only with OpenMP (`meson -Dgpu=false`), installs it
to `~/spral-install` and compiles the driver to `~/ssids_bench/ssids_replay`.
All of this stays inside WSL; nothing is committed. In Git Bash prefix the
commands with `MSYS_NO_PATHCONV=1` so `/mnt/c/...` is not rewritten.

Rebuild the driver after editing `ssids_replay.c` by re-running `build.sh`
(it skips the SPRAL build when `libspral.so` is already installed).

## Running

```
python tools/bench/run_wsl.py --manifest benchmarks/manifests/smoke.json
python tools/bench/run.py --manifest <m> --output <dir> --runner wsl --solvers smf,ssids ...
```

`--runner wsl` sends the solver ids in `run_wsl.WSL_SOLVERS` (`ssids`) through
`wsl.exe -d Ubuntu --exec env ... ssids_replay`; every other solver id keeps
running natively. Windows paths are translated to `/mnt/<drive>/...`, and the
built-in generated cases (smoke, tridiag, KKT stage models) are first written
to `build/wsl_cases/*.mtx` with `bench_replay --export-mtx`.

The driver is started with `OMP_NUM_THREADS=<threads>`, `OPENBLAS_NUM_THREADS`,
and the two settings SSIDS requires, `OMP_CANCELLATION=TRUE` and
`OMP_PROC_BIND=TRUE`. If WSL, the distro or the driver is missing, records come
back as `Unavailable` (with the reason) and the case is skipped, like any other
unavailable solver.

Environment overrides: `SMF_WSL_DISTRO` (default `Ubuntu`),
`SMF_WSL_SSIDS_DRIVER` (default `~/ssids_bench/ssids_replay`).

## SSIDS settings

| matrix | factorization | ordering | scaling | pivot u |
|---|---|---|---|---|
| SPD | Cholesky | METIS (`ordering=1`) | none | n/a |
| indefinite | LDL^T | matching-based (`ordering=2`) | its own (`scaling=3`) | 0.01; 1e-8 on the `ipopt` track |

`--ssids-ordering N` / `--ssids-scaling N` override the ordering/scaling.
Timing is `clock_gettime(CLOCK_MONOTONIC)` around the SPRAL calls inside WSL
(same CPU as the Windows runs, under Hyper-V; the `.mtx` files are read from
`/mnt/c`, but parsing is outside the timed region and is cached next to the
matrix as `<mtx>.smfbin`, the same file `bench_replay` writes).
