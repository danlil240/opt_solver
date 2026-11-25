# Enabling native `linear_solver=smf` in `conda-forge/ipopt-feedstock`

Once `libsmf` is published on conda-forge, enabling built-in `linear_solver=smf`
in `conda-forge/ipopt-feedstock` requires only three small additions:

1. **Add the patch** `tools/ipopt/smf-linear-solver/ipopt-3.14.19-smf.patch` to
   `recipe/patches/` and list it under `source.patches:` in `meta.yaml`.

2. **Add `libsmf` to `requirements.host`** in `meta.yaml`:
   ```yaml
   requirements:
     host:
       - libblas
       - liblapack
       - libsmf
       - mumps-seq
       # ...
   ```
   (`libsmf` exports `run_exports`, so `ipopt` automatically gets the matching
   runtime dependency.)

3. **Pass `IPOPT_HAS_SMF` flags to `./configure`** in `build.sh`:
   ```bash
   SMF_CFLAGS="$(pkg-config --cflags smf-solver)"
   SMF_LIBS="$(pkg-config --libs smf-solver)"
   ./configure \
     --prefix="${PREFIX}" \
     CPPFLAGS="${CPPFLAGS} -DIPOPT_HAS_SMF ${SMF_CFLAGS}" \
     LIBS="${LIBS:-} ${SMF_LIBS}" \
     ...
   ```

## Resulting behavior for conda-forge users

- `IpoptGetAvailableLinearSolvers(1)` includes `IPOPTLINEARSOLVER_SMF`.
- `linear_solver=smf` is built-in (no `smflib` or `hsllib` path needed) and
  works cleanly alongside MUMPS, SPRAL, or user-provided CoinHSL.
- Uses conda-forge's generic `libblas`/`liblapack` (OpenBLAS by default, or
  MKL / Apple Accelerate via `libblas` build strings).
- Downstream conda packages (`cyipopt`, `casadi`, `pyomo`, `do-mpc`) gain
  `linear_solver="smf"` automatically with zero code changes.
