# Julia `smf_jll` and `Ipopt.jl` Integration

## 1. Using `smf_jll` with stock `Ipopt.jl` (Julia 1.11+)

`smf_jll` is registered in Julia's General registry (18 platforms: Linux glibc/musl, macOS, Windows, FreeBSD) and ships both `libsmf_c` and the `libsmf_ma97` drop-in plugin.

```julia
import Pkg
Pkg.add(["JuMP", "Ipopt", "smf_jll"])

using JuMP, Ipopt
import smf_jll

model = Model(Ipopt.Optimizer)
set_attribute(model, "linear_solver", "ma97")
set_attribute(model, "hsllib", smf_jll.libsmf_ma97_path)
```

Verify that `smf` is serving the factorizations by running with `SMF_PLUGIN_STATS=1`, or run the included smoke test:

```bash
SMF_PLUGIN_STATS=1 julia packaging/ipopt_jll/test_ipopt_smf.jl
```

## 2. Building `smf_jll` via Yggdrasil

`build_tarballs.jl` mirrors the merged Yggdrasil recipe (`S/smf/build_tarballs.jl`), building `libsmf_c` (versioned C ABI + `smf-solver.pc`) and `libsmf_ma97` against `OpenBLAS32_jll` and `SuiteSparse32_jll`.

## 3. Native `linear_solver="smf"` in `Ipopt_jll` (once upstreamed to `coin-or/Ipopt`)

Once `IpSmfSolverInterface` lands in upstream `coin-or/Ipopt`, `Ipopt_jll` can link `smf_jll` directly with `CPPFLAGS="-DIPOPT_HAS_SMF -I${includedir}"` and `LIBS="-L${libdir} -lsmf_c"`:

```julia
using JuMP, Ipopt

model = Model(Ipopt.Optimizer)
set_attribute(model, "linear_solver", "smf")
set_attribute(model, "smf_num_threads", 0)  # 0 = auto
```
