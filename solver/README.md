# smf — Sparse Multifrontal Solver

`smf` is a C++20 sparse symmetric multifrontal direct solver inspired by HSL_MA97, supporting positive-definite Cholesky and indefinite LDLᵀ factorization with OpenMP tree-level parallelism and a clean adapter for IPOPT.

## Dependencies

- CMake ≥ 3.20
- GCC ≥ 11 or Clang ≥ 14 with C++20 support
- BLAS and LAPACK (e.g. `libopenblas-dev`)
- OpenMP (included with GCC/Clang)
- SuiteSparse AMD (e.g. `libsuitesparse-dev`) — optional but recommended
- METIS 5 (e.g. `libmetis-dev`) — optional but recommended
- GoogleTest (e.g. `libgtest-dev`) — required for tests

## How to build

Configure and build from the repository root with an out-of-source build directory:

```bash
cmake -S solver -B solver/build
cmake --build solver/build -- -j$(nproc)
```

To enable address/UB sanitizers:

```bash
cmake -S solver -B solver/build -DSMF_SANITIZE=ON
cmake --build solver/build -- -j$(nproc)
```

Available CMake options (all `ON` unless noted):

| Option                  | Default | Description                              |
|-------------------------|---------|------------------------------------------|
| `SMF_ENABLE_OPENMP`     | ON      | Enable OpenMP parallelism                |
| `SMF_USE_MKL`           | OFF     | Use Intel MKL instead of OpenBLAS        |
| `SMF_USE_METIS`         | ON      | Enable METIS 5 fill-reducing ordering    |
| `SMF_USE_SUITESPARSE_AMD` | ON   | Enable SuiteSparse AMD ordering          |
| `SMF_DETERMINISTIC`     | OFF     | Enable reproducible (deterministic) mode |
| `SMF_BUILD_TESTS`       | ON      | Build unit tests                         |
| `SMF_BUILD_BENCHMARKS`  | ON      | Build benchmarks                         |
| `SMF_SANITIZE`          | OFF     | Enable `-fsanitize=address,undefined`    |

## How to run tests

```bash
ctest --test-dir solver/build --output-on-failure
```
