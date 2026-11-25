# Native `linear_solver=smf` for IPOPT

This directory provides a first-class IPOPT `SparseSymLinearSolverInterface` (`IpSmfSolverInterface`) backed by smf's versioned C ABI (`libsmf_c`, `smf/smf_c.h`).

## Why native `linear_solver=smf` instead of the `hsllib`/`ma97` shim?

1. **Works alongside full CoinHSL:** IPOPT removes the `hsllib` option when compiled against full CoinHSL (`No such IPOPT option: hsllib`). Native `linear_solver=smf` uses compile-time linking (`IPOPT_HAS_SMF`) or its own `smflib` option (`libsmf_c.so` / `libsmf_c.dylib` / `smf_c.dll`), completely independent of `hsllib` and HSL.
2. **Immune to `coin-or/Ipopt#870` (`numneg_` bug):** `SmfSolverInterface` initializes `numneg_(0)` in its constructor and updates `numneg_ = info.num_neg` before every inertia check.
3. **Zero format conversion:** `MatrixFormat()` returns `CSR_Format_0_Offset`, which for symmetric matrices is identical to smf's 0-based lower-triangular CSC — no triplet-to-CSR sort or index shifting.
4. **Native options & honest logs:** Exposes `smf_u`, `smf_umax`, `smf_small`, `smf_nemin`, `smf_ordering`, `smf_scaling`, `smf_num_threads`, `smf_pivoting`, `smf_slack_elimination`, `smf_deterministic`, `smf_print_level`, and `smflib`, and credits `smf` in IPOPT's startup banner (`running with linear solver smf ...`).

## Building IPOPT 3.14.19 with native `smf`

### 1. Build and install `libsmf_c`

```bash
cmake -S solver -B build -DCMAKE_BUILD_TYPE=Release -DSMF_USE_MKL=OFF -DSMF_MARCH=x86-64-v3 -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build build -j
cmake --install build
```

### 2. Patch IPOPT 3.14.19

```bash
python3 tools/ipopt/smf-linear-solver/apply_to_ipopt.py /path/to/Ipopt-3.14.19
# or: git -C /path/to/Ipopt-3.14.19 apply /path/to/ipopt-3.14.19-smf.patch
```

### 3a. Compile-time linked (`smf` becomes a built-in default linear solver)

```bash
./configure \
  --prefix=/usr/local \
  CPPFLAGS="-DIPOPT_HAS_SMF $(pkg-config --cflags smf-solver)" \
  LDFLAGS="$(pkg-config --libs-only-L smf-solver) -Wl,-rpath,$(pkg-config --variable=libdir smf-solver)" \
  LIBS="$(pkg-config --libs-only-l smf-solver)"
make -j && make install
```

### 3b. Runtime dynamic loading via `smflib` (works alongside CoinHSL or MUMPS)

```bash
./configure --prefix=/usr/local
make -j && make install
```

Then at runtime in IPOPT / CasADi / cyipopt / JuMP:

```text
linear_solver smf
smflib /path/to/libsmf_c.so   # optional if libsmf_c is on LD_LIBRARY_PATH / rpath
smf_num_threads 0             # 0 = auto (SMF_NUM_THREADS / OMP_NUM_THREADS / P-cores)
smf_scaling auto              # auto | dynamic | none | matching | equilibration
```
