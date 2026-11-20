# Julia `smf_jll` and `Ipopt_jll` / `Ipopt.jl` Integration

## 1. Building `smf_jll` via Yggdrasil

`build_tarballs.jl` builds `libsmf_c` (versioned C ABI + `smf.pc`) and
`libsmf_ma97` against `OpenBLAS32_jll` and `SuiteSparse32_jll` for Linux
(`x86_64`, `aarch64`), macOS (`x86_64`, `aarch64`), and Windows (`x86_64`).

## 2. Wiring `linear_solver="smf"` into `Ipopt_jll`

In Yggdrasil `I/Ipopt/build_tarballs.jl`:

1. Add `Dependency("smf_jll")` to `dependencies`.
2. Apply `tools/ipopt/smf-linear-solver/ipopt-3.14.19-smf.patch` to the Ipopt
   source tree.
3. Pass `CPPFLAGS="-DIPOPT_HAS_SMF -I${includedir}"` and `LIBS="-L${libdir} -lsmf_c"`
   to Ipopt's `./configure`.

Then from Julia (`JuMP` + `Ipopt.jl`):

```julia
using JuMP, Ipopt

model = Model(Ipopt.Optimizer)
set_attribute(model, "linear_solver", "smf")
set_attribute(model, "smf_num_threads", 0)  # 0 = auto
```

## 3. Drop-in usage with stock `Ipopt.jl` today (before `Ipopt_jll` updates)

```julia
using JuMP, Ipopt, smf_jll

model = Model(Ipopt.Optimizer)
set_attribute(model, "linear_solver", "ma97")
set_attribute(model, "hsllib", smf_jll.libsmf_ma97)
```
