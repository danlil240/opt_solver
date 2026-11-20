# `smf-solver` Python Package (`cyipopt` & `CasADi`)

Bundles `libsmf_c` (for native `linear_solver="smf"`) and `libsmf_ma97` (for
stock `cyipopt` and `CasADi` IPOPT wheels) with zero-boilerplate Python helpers.

## Building from source

```bash
pip install ./packaging/python
```

For portable Linux/macOS x86_64 wheels, pass `-DSMF_MARCH=x86-64-v3`:

```bash
CMAKE_ARGS="-DSMF_MARCH=x86-64-v3" pip wheel ./packaging/python -w dist/
```

## Usage with `cyipopt`

```python
import cyipopt
import smf_solver

nlp = cyipopt.Problem(...)

# With IPOPT built with native linear_solver=smf:
for k, v in smf_solver.ipopt_options(native=True, num_threads=0).items():
    nlp.add_option(k, v)

# With stock PyPI cyipopt / IPOPT (before native smf lands upstream):
for k, v in smf_solver.ipopt_options(native=False).items():
    nlp.add_option(k, v)
```

## Usage with `CasADi`

```python
import casadi as ca
import smf_solver

opts = smf_solver.casadi_options(native=True)   # or native=False for stock CasADi wheels
solver = ca.nlpsol("solver", "ipopt", nlp, opts)
```
