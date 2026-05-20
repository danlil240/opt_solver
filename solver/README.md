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

### Install all apt dependencies in one command

```bash
sudo apt-get install -y \
    cmake build-essential \
    libopenblas-dev liblapack-dev \
    libsuitesparse-dev libmetis-dev \
    libgtest-dev libeigen3-dev
```

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

| Option                    | Default | Description                                         |
|---------------------------|---------|-----------------------------------------------------|
| `SMF_ENABLE_OPENMP`       | ON      | Enable OpenMP parallelism                           |
| `SMF_USE_MKL`             | OFF     | Use Intel MKL instead of OpenBLAS                  |
| `SMF_USE_METIS`           | ON      | Enable METIS 5 fill-reducing ordering               |
| `SMF_USE_SUITESPARSE_AMD` | ON      | Enable SuiteSparse AMD ordering                     |
| `SMF_DETERMINISTIC`       | OFF     | Enable reproducible (deterministic) mode            |
| `SMF_BUILD_TESTS`         | ON      | Build unit tests                                    |
| `SMF_BUILD_BENCHMARKS`    | ON      | Build benchmarks                                    |
| `SMF_SANITIZE`            | OFF     | Enable `-fsanitize=address,undefined`               |
| `SMF_REQUIRE_MA27`        | OFF     | Fail configure with install hint if CoinHSL absent  |
| `SMF_BUILD_IPOPT_BENCHMARK` | OFF   | Build `bench_ipopt_compare` (requires IPOPT installed) |

## How to run tests

```bash
ctest --test-dir solver/build --output-on-failure
```

## Installing smf

Install to a prefix (e.g. `/usr/local` or a custom path):

```bash
cmake -S solver -B solver/build -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build solver/build -- -j$(nproc)
cmake --install solver/build
```

This installs:
- `lib/libsmf.a` and (if enabled) `lib/libsmf_ma97.so`
- `include/smf/` headers
- `lib/cmake/smf/smfConfig.cmake`, `smfConfigVersion.cmake`, `smfTargets.cmake`
- `lib/cmake/smf/Find{METIS,CHOLMOD,SuiteSparseAMD,CoinHSL}.cmake` (for transitive deps)

### Using smf in a downstream CMake project

After installation, a downstream project can consume smf with standard CMake:

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_solver CXX)

find_package(smf CONFIG REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE smf::smf)
```

**Minimal smoke-test `main.cpp`:**

```cpp
#include "smf/solver.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/control.hpp"
#include "smf/info.hpp"
#include <cstdio>
#include <vector>
int main() {
    // 2x2 identity: lower triangle (CSC format)
    smf::CscLower A;
    A.n = 2;
    A.col_ptr = {0, 1, 2};
    A.row_idx = {0, 1};
    A.values  = {1.0, 1.0};
    smf::Control ctrl;
    smf::Info    info;
    smf::Solver  solver;
    std::vector<double> rhs = {3.0, 7.0};
    // factor_solve: combined analyse + factor + solve in one call
    auto fkeep = solver.factor_solve(A, ctrl, info, rhs.data(), A.n);
    if (!fkeep) { std::puts("factor_solve failed"); return 1; }
    std::printf("smoke_test: x = [%.1f, %.1f]  (expect [3.0, 7.0])\n",
                rhs[0], rhs[1]);
    return 0;
}
```

Configure with (using a custom prefix):

```bash
cmake -S . -B build -Dsmf_DIR=/usr/local/lib/cmake/smf
cmake --build build
./build/my_app
```

## MA27 comparison benchmark (`bench_compare`)

`bench_compare` times smf against CHOLMOD and, optionally, HSL MA27 (via CoinHSL).
MA27 is a reference-quality indefinite sparse direct solver from the Harwell Subroutine Library.

### Why CoinHSL cannot be installed with apt-get

CoinHSL (which packages MA27, MA57, MA97 and other HSL routines) is licensed by
STFC Rutherford Appleton Laboratory.  It is **not freely redistributable** and therefore
does not appear in standard Debian/Ubuntu package repositories.  Running
`apt-get install coinor-libhsl-dev` or similar will fail — no such package exists on
stock runners or workstations.

### Getting CoinHSL (libcoinhsl) — fastest supported path

1. **Register and download** the CoinHSL source (free academic or commercial licence):
   <https://www.hsl.rl.ac.uk/>

2. **Build and install** via ThirdParty-HSL (recommended for IPOPT users):

```bash
git clone https://github.com/coin-or-tools/ThirdParty-HSL
cd ThirdParty-HSL
# copy or symlink your coinhsl/ source directory here, then:
./configure --prefix=/usr/local
make -j$(nproc)
sudo make install
# installs /usr/local/lib/libcoinhsl.so and headers
```

3. **Configure smf** using one of:

```bash
# Auto-discover via prefix:
cmake -S solver -B solver/build -DCOINHSL_ROOT=/usr/local

# Point directly at the library:
cmake -S solver -B solver/build \
    -DCOINHSL_LIBRARY=/usr/local/lib/libcoinhsl.so

# Require it (configure fails with detailed install instructions when absent):
cmake -S solver -B solver/build -DSMF_REQUIRE_MA27=ON

# Combine with root override:
cmake -S solver -B solver/build \
    -DSMF_REQUIRE_MA27=ON -DCOINHSL_ROOT=/usr/local
```

`FindCoinHSL.cmake` also honours the `COINHSL_ROOT` environment variable and probes
`pkg-config` (ThirdParty-HSL installs `coinhsl.pc`), so the library is found
automatically once it is in any standard prefix.

### Building bench_compare with MA27 enabled

```bash
cmake -S solver -B solver/build -DCOINHSL_ROOT=/usr/local
cmake --build solver/build --target bench_compare -- -j$(nproc)
```

### Running the benchmark

```bash
./solver/build/benchmarks/bench_compare
```

Output columns: solver name, matrix size, factor time (ms), solve time (ms), residual.
CHOLMOD is auto-enabled when `libsuitesparse-dev` is installed.
MA27 columns appear only when `SMF_HAS_MA27=ON` in the build.

### Expected output (CHOLMOD only, no MA27)

```
[smf vs CHOLMOD vs MA27 — 5 SPD matrices]
...
CoinHSL/MA27 not found — MA27 column disabled
...
avg smf/CHOLMOD speedup: ~0.39x   (smf is work-in-progress; CHOLMOD is highly tuned)
```

The current ~0.39× figure vs CHOLMOD is expected — smf is a research prototype;
MA27 comparison requires CoinHSL as described above.
