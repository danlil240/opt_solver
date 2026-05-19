# MA97-Class Sparse Symmetric Multifrontal Solver — Implementation Plan

> **Goal:** Implement a C++20 sparse symmetric direct solver inspired by the public HSL_MA97 specification.  
> **Important:** Do **not** attempt a line-by-line clone of HSL_MA97. Build an **MA97-class solver** with a similar conceptual workflow: `analyse → factor → solve → enquire/alter/finalize`.

---

## 0. Project Goal

Implement a C++20 sparse symmetric direct solver supporting:

```cpp
analyse(...)
factor(...)
solve(...)
factor_solve(...)
enquire_indef(...)
enquire_posdef(...)
alter_d(...)
finalise()/clear()
```

Primary target:

```text
Real double precision
Sparse symmetric matrices
Lower-triangular CSC input
Positive-definite: sparse Cholesky
Indefinite: sparse LDLᵀ with 1x1 and 2x2 pivots
Multiple RHS
Repeated numeric factorization with same sparsity pattern
OpenMP parallelism
BLAS/LAPACK dense kernels
```

Secondary target:

```text
Single precision
Complex Hermitian
Complex symmetric
Sparse RHS forward solve
Fredholm/inconsistent-system handling
Strict bit-compatible parallel mode
```

Do **not** start with complex numbers. Implement `double` first.

---

## 1. High-Level Architecture

Create a library called:

```text
symmetric_multifrontal_solver
```

Suggested directory layout:

```text
solver/
  CMakeLists.txt
  include/
    smf/solver.hpp
    smf/types.hpp
    smf/csc_matrix.hpp
    smf/control.hpp
    smf/info.hpp
    smf/analysis.hpp
    smf/factor.hpp
    smf/ordering.hpp
    smf/scaling.hpp
    smf/dense_kernel.hpp
    smf/threading.hpp
    smf/ipopt_adapter.hpp       # optional later
  src/
    csc_matrix.cpp
    check_matrix.cpp
    ordering_amd.cpp
    ordering_metis.cpp
    symbolic_analysis.cpp
    etree.cpp
    assembly_tree.cpp
    supernode_detection.cpp
    factor_posdef.cpp
    factor_indef.cpp
    pivoting.cpp
    frontal_matrix.cpp
    solve.cpp
    scaling.cpp
    dense_kernel_lapack.cpp
    diagnostics.cpp
  tests/
    test_matrix_check.cpp
    test_ordering.cpp
    test_etree.cpp
    test_symbolic.cpp
    test_cholesky_small.cpp
    test_ldlt_small.cpp
    test_kkt_ipopt_like.cpp
    test_repeated_factor.cpp
    test_inertia.cpp
    test_parallel_determinism.cpp
  benchmarks/
    bench_poisson.cpp
    bench_kkt_ocp.cpp
    bench_random_symmetric.cpp
    bench_suite_sparse_matrix_market.cpp
```

Core components:

```text
CscMatrixLower
Control
Info
Analysis
Factorization
Solver
OrderingBackend
ScalingBackend
DenseKernelBackend
ThreadPool/OpenMP utilities
```

---

## 2. Public C++ API

Implement an API similar to MA97 but idiomatic C++.

```cpp
#pragma once

#include <cstdint>
#include <span>
#include <vector>
#include <memory>
#include <optional>

namespace smf {

enum class MatrixType {
    RealSymmetricPositiveDefinite,
    RealSymmetricIndefinite
};

enum class OrderingMethod {
    User,
    AMD,
    METIS,
    AutoSerial,
    AutoParallel,
    MatchingAMD,
    MatchingMETIS
};

enum class ScalingMethod {
    None,
    User,
    Matching,      // MC64-like later
    Equilibration, // MC77-like later
    LogSum         // MC30-like later
};

enum class FactorStatus {
    Success,
    Singular,
    NotPositiveDefinite,
    InvalidInput,
    OutOfMemory,
    NumericalFailure
};

struct Control {
    int print_level = 0;

    OrderingMethod ordering = OrderingMethod::AutoParallel;
    int nemin = 8;

    ScalingMethod scaling = ScalingMethod::None;

    bool continue_on_singular = true;

    double pivot_tolerance = 0.01;
    double small_pivot = 1e-20;
    double factor_memory_multiplier = 1.1;

    std::int64_t factor_parallel_min_flops = 20'000'000;
    std::int64_t solve_parallel_min_entries = 100'000;

    bool solve_multifrontal_forward = false;
    bool solve_use_blas3_single_rhs = false;

    bool deterministic = true;
    int num_threads = 0; // 0 means OpenMP default
};

struct Info {
    FactorStatus status = FactorStatus::Success;

    int matrix_duplicates = 0;
    int matrix_missing_diag = 0;
    int matrix_out_of_range = 0;

    int structural_rank = 0;
    int numerical_rank = 0;

    int ordering_used = 0;

    int max_tree_depth = 0;
    int max_front_size = 0;
    int max_supernode_size = 0;

    std::int64_t predicted_factor_entries = 0;
    std::int64_t actual_factor_entries = 0;
    std::int64_t predicted_flops = 0;
    std::int64_t actual_flops = 0;

    int delayed_pivots = 0;

    int num_negative = 0;
    int num_zero = 0;
    int num_positive = 0;

    double analyse_seconds = 0.0;
    double factor_seconds = 0.0;
    double solve_seconds = 0.0;
};

struct CscLower {
    int n = 0;
    std::vector<int> col_ptr;  // size n + 1
    std::vector<int> row_idx;  // lower triangle only: row >= col
    std::vector<double> values;
};

struct AnalysisHandle;
struct FactorHandle;

class Solver {
public:
    explicit Solver(Control control = {});

    AnalysisHandle analyse(
        const CscLower& pattern,
        bool check_input,
        std::optional<std::span<const int>> user_order = std::nullopt);

    FactorHandle factor(
        const AnalysisHandle& analysis,
        const CscLower& matrix,
        MatrixType matrix_type,
        std::optional<std::span<const double>> user_scale = std::nullopt);

    void solve(
        const AnalysisHandle& analysis,
        const FactorHandle& factor,
        int nrhs,
        std::span<const double> b,
        int ldb,
        std::span<double> x,
        int ldx) const;

    FactorHandle factor_solve(
        const AnalysisHandle& analysis,
        const CscLower& matrix,
        MatrixType matrix_type,
        int nrhs,
        std::span<double> rhs_in_out,
        int ld_rhs);

    const Control& control() const noexcept;
    const Info& info() const noexcept;

private:
    Control control_;
    mutable Info info_;
};

} // namespace smf
```

---

## 3. Input Format and Matrix Checking

### 3.1 Accepted Input

Accept only lower-triangular CSC first:

```text
col_ptr[j] ... col_ptr[j+1]-1 are row indices for column j
row_idx[k] >= j
values[k] corresponds to A(row_idx[k], j)
```

Later add coordinate input.

### 3.2 `check_input = true`

Implement a matrix-cleaning phase:

```cpp
struct CleanedPattern {
    CscLower clean;
    std::vector<int> original_to_clean;
    int duplicates;
    int out_of_range;
    int missing_diag;
};
```

Rules:

```text
Discard out-of-range rows/columns
Discard upper-triangle entries row < col
Sort each column by row
Sum duplicate entries
Detect missing diagonal entries
Allow implicit zero diagonal
```

Implementation sketch:

```cpp
CleanedPattern clean_lower_csc(const CscLower& a) {
    CleanedPattern out;
    out.clean.n = a.n;
    out.clean.col_ptr.assign(a.n + 1, 0);

    std::vector<std::vector<std::pair<int, double>>> cols(a.n);

    for (int j = 0; j < a.n; ++j) {
        for (int p = a.col_ptr[j]; p < a.col_ptr[j + 1]; ++p) {
            const int i = a.row_idx[p];

            if (i < 0 || i >= a.n || j < 0 || j >= a.n) {
                ++out.out_of_range;
                continue;
            }

            if (i < j) {
                ++out.out_of_range;
                continue;
            }

            cols[j].push_back({i, a.values.empty() ? 0.0 : a.values[p]});
        }

        std::sort(cols[j].begin(), cols[j].end(),
                  [](auto a, auto b) { return a.first < b.first; });

        bool has_diag = false;
        std::vector<std::pair<int, double>> merged;

        for (const auto& [i, v] : cols[j]) {
            if (i == j) {
                has_diag = true;
            }

            if (!merged.empty() && merged.back().first == i) {
                merged.back().second += v;
                ++out.duplicates;
            } else {
                merged.push_back({i, v});
            }
        }

        if (!has_diag) {
            ++out.missing_diag;
        }

        cols[j] = std::move(merged);
        out.clean.col_ptr[j + 1] = out.clean.col_ptr[j] +
                                   static_cast<int>(cols[j].size());
    }

    const int nnz = out.clean.col_ptr.back();
    out.clean.row_idx.resize(nnz);
    out.clean.values.resize(nnz);

    for (int j = 0; j < a.n; ++j) {
        int p = out.clean.col_ptr[j];
        for (const auto& [i, v] : cols[j]) {
            out.clean.row_idx[p] = i;
            out.clean.values[p] = v;
            ++p;
        }
    }

    return out;
}
```

---

## 4. Analysis Phase

The `analyse()` phase must build all symbolic data needed for repeated numerical factorization.

### 4.1 Analysis Pipeline

Implement:

```text
1. Validate/clean CSC lower pattern
2. Build undirected symmetric graph G(A)
3. Compute ordering
4. Apply permutation P
5. Compute symbolic elimination tree
6. Detect supernodes
7. Amalgamate small neighboring nodes using nemin
8. Build assembly tree
9. Predict front sizes, factor entries, memory, flops
10. Allocate AnalysisHandle
```

### 4.2 Ordering

Start with:

```text
Phase 1: Natural ordering
Phase 2: AMD through SuiteSparse AMD
Phase 3: METIS_NodeND
Phase 4: Auto choice: AMD for small/serial, METIS for large/parallel
Phase 5: matching-based ordering later
```

Ordering backend:

```cpp
class OrderingBackend {
public:
    virtual ~OrderingBackend() = default;

    virtual std::vector<int> compute_ordering(
        int n,
        const std::vector<int>& xadj,
        const std::vector<int>& adjncy) = 0;
};
```

For the first production target, use:

```text
SuiteSparse AMD
METIS 5 NodeND
```

### 4.3 Elimination Tree

```cpp
struct EliminationTree {
    std::vector<int> parent;
    std::vector<std::vector<int>> children;
    std::vector<int> postorder;
    int max_depth = 0;
};
```

### 4.4 Supernodes and Assembly Tree

```cpp
struct Supernode {
    int id = -1;

    std::vector<int> elim_cols;      // fully summed variables
    std::vector<int> row_indices;    // frontal row set
    std::vector<int> children;

    int parent = -1;
    int depth = 0;

    std::int64_t predicted_flops = 0;
    std::int64_t predicted_factor_entries = 0;
};
```

Supernode detection:

```text
Group adjacent columns with identical or near-identical structure below diagonal.
Then amalgamate small neighboring nodes if both have fewer than control.nemin eliminations.
```

---

## 5. Numerical Factorization

### 5.1 Factorization Handle

```cpp
struct Pivot {
    enum class Type { OneByOne, TwoByTwo };
    Type type;
    int col0;
    int col1;       // valid only for 2x2
    double d00;
    double d10;
    double d11;
};

struct FactorNode {
    int node_id;

    std::vector<int> elim_cols;
    std::vector<int> row_indices;

    // Dense frontal data, column-major.
    // Store only L factors and D block data needed for solve.
    std::vector<double> l_block;

    std::vector<Pivot> pivots;

    // Contribution block metadata.
    std::vector<int> contribution_rows;
};

struct FactorHandle {
    MatrixType matrix_type;
    std::vector<FactorNode> nodes;

    std::vector<double> scale;
    std::vector<int> perm;
    std::vector<int> inv_perm;

    int numerical_rank = 0;
    int num_positive = 0;
    int num_negative = 0;
    int num_zero = 0;

    std::int64_t actual_factor_entries = 0;
    std::int64_t actual_flops = 0;
};
```

### 5.2 Multifrontal Factorization Algorithm

For each node in postorder:

```text
1. Allocate dense frontal matrix F.
2. Scatter original matrix entries belonging to this front into F.
3. Assemble contribution blocks from child nodes into F.
4. Identify fully summed variables of this node.
5. Factor fully summed part:
   - SPD: dense Cholesky, no pivoting.
   - Indefinite: threshold LDLᵀ with 1x1/2x2 pivots.
6. Store L block and D pivots.
7. Form contribution block / Schur complement for parent.
8. Free child contribution memory.
9. Delay unstable pivots to parent if needed.
```

### 5.3 Positive-Definite Path

Use LAPACK/BLAS:

```text
dpotrf
dtrsm
dsyrk/dgemm
```

Pseudo-code:

```cpp
void factor_front_spd(FrontalMatrix& F, FactorNode& out) {
    const int k = F.num_fully_summed();
    const int m = F.size();

    // F = [ F11 F21ᵀ
    //       F21 F22  ]
    // Factor F11 = L11 L11ᵀ.
    lapack_dpotrf_lower(k, F.ptr(0, 0), F.ld());

    // L21 = F21 * inv(L11ᵀ)
    blas_dtrsm_right_lower_transpose(
        k, m - k, F.ptr(k, 0), F.ld(), F.ptr(0, 0), F.ld());

    // Schur update: F22 -= L21 L21ᵀ
    blas_dsyrk_lower(
        m - k, k, -1.0, F.ptr(k, 0), F.ld(), 1.0, F.ptr(k, k), F.ld());

    out.store_l_from_front(F, k);
}
```

If a nonpositive pivot appears in SPD mode, return `NotPositiveDefinite`.

### 5.4 Indefinite Path: LDLᵀ with 1x1/2x2 Pivots

Implement a bounded Bunch-Kaufman-style threshold pivoting algorithm for each frontal matrix.

Pivot acceptance:

```text
1x1 pivot |a_kk| >= u * max_offdiag_in_col(k)
2x2 pivot accepted if the 2x2 block is nonsingular and gives acceptable growth
Treat |pivot| < small as numerically zero
```

Use `control.pivot_tolerance` as relative pivot tolerance `u`.

Required behavior:

```text
Accepted 1x1 pivot:
    store scalar D
    compute L column
    update trailing block

Accepted 2x2 pivot:
    store D block [[d00, d10], [d10, d11]]
    compute two L columns using inverse of 2x2 D
    update trailing block

Rejected pivot:
    delay variable to parent
```

Pseudo-code skeleton:

```cpp
struct DenseLdltResult {
    std::vector<Pivot> pivots;
    std::vector<int> accepted;
    std::vector<int> delayed;
    int positive = 0;
    int negative = 0;
    int zero = 0;
};

DenseLdltResult factor_front_indefinite(
    FrontalMatrix& F,
    int num_fully_summed,
    double u,
    double small)
{
    DenseLdltResult result;

    int k = 0;
    while (k < num_fully_summed) {
        const auto candidate = choose_pivot(F, k, num_fully_summed, u, small);

        if (candidate.type == PivotCandidate::Type::Reject) {
            result.delayed.push_back(candidate.col0);
            F.move_column_to_delayed_region(candidate.col0);
            --num_fully_summed;
            continue;
        }

        if (candidate.type == PivotCandidate::Type::OneByOne) {
            apply_1x1_pivot(F, k, candidate.col0, result);
            k += 1;
        } else {
            apply_2x2_pivot(F, k, candidate.col0, candidate.col1, result);
            k += 2;
        }
    }

    return result;
}
```

Inertia calculation:

```cpp
void add_inertia_1x1(double d, Info& info) {
    if (std::abs(d) <= info_small) {
        ++info.num_zero;
    } else if (d > 0.0) {
        ++info.num_positive;
    } else {
        ++info.num_negative;
    }
}

void add_inertia_2x2(double a, double b, double c, Info& info) {
    // D = [a b; b c]
    // Eigenvalue signs from trace/determinant.
    const double det = a * c - b * b;
    const double tr = a + c;

    if (std::abs(det) <= info_small) {
        // One zero, one sign from trace.
        ++info.num_zero;
        if (tr > 0.0) ++info.num_positive;
        else if (tr < 0.0) ++info.num_negative;
        else ++info.num_zero;
    } else if (det < 0.0) {
        ++info.num_positive;
        ++info.num_negative;
    } else {
        if (tr > 0.0) info.num_positive += 2;
        else info.num_negative += 2;
    }
}
```

For IPOPT, accurate inertia is mandatory.

---

## 6. Scaling

Implement scaling in stages.

### Stage 1: No Scaling and User Scaling

```text
A_scaled = S A S
b_scaled = S b
x = S x_scaled
```

### Stage 2: Simple Equilibration

Implement max-norm symmetric equilibration:

```cpp
scale[i] = 1.0 / sqrt(max_abs_row_col_i);
```

### Stage 3: Matching-Based Scaling

Later implement MC64-like maximum product matching or use an external package.

---

## 7. Solve Phase

Implement:

```text
Permutation/scaling apply
Forward solve through L
Diagonal/block-diagonal D solve
Backward solve through Lᵀ
Inverse permutation/scaling
```

### 7.1 Multiple RHS

Prioritize multiple RHS path:

```cpp
void solve_dense_rhs(
    const AnalysisHandle& analysis,
    const FactorHandle& factor,
    int nrhs,
    const double* b,
    int ldb,
    double* x,
    int ldx);
```

Use BLAS-3 when `nrhs > 1`:

```text
dtrsm
dgemm
```

---

## 8. Parallelism and CPU Architecture

### 8.1 Baseline

Use OpenMP first.

CMake option:

```cmake
option(SMF_ENABLE_OPENMP "Enable OpenMP parallel factorization" ON)
option(SMF_DETERMINISTIC "Enable deterministic assembly order" ON)
```

### 8.2 Tree-Level Parallelism

Factor independent subtrees using tasks:

```cpp
#pragma omp parallel
#pragma omp single
{
    factor_node_task(root);
}
```

```cpp
void factor_node_task(int node_id) {
    for (int child : tree.children[node_id]) {
        #pragma omp task shared(...)
        factor_node_task(child);
    }

    #pragma omp taskwait
    factor_node_after_children(node_id);
}
```

### 8.3 Node-Level Parallelism

For large frontal matrices, use:

```text
Parallel dsyrk/dgemm via threaded BLAS
OR explicit OpenMP tiling if deterministic mode is required
```

Recommended policy:

```text
Small fronts: single-thread BLAS, parallelize over tree
Large fronts near root: allow threaded BLAS or custom tiled updates
Avoid nested oversubscription
```

Implement a BLAS thread guard:

```cpp
class BlasThreadGuard {
public:
    explicit BlasThreadGuard(int threads);
    ~BlasThreadGuard();
};
```

Rules:

```text
If OpenMP task parallelism active, set BLAS threads = 1 inside each task.
For root/fronts above threshold, optionally set BLAS threads = control.num_threads and serialize tree work.
```

### 8.4 Determinism

Add `control.deterministic`.

If true:

```text
Sort children by node id before assembly.
Use fixed assembly order.
Avoid OpenMP reductions with undefined summation order.
Use deterministic tiled update order.
Use single-thread BLAS if strict bitwise reproducibility is required.
```

### 8.5 CPU Performance Priorities

Optimize for:

```text
Cache locality
BLAS-3 kernels
Avoiding small heap allocations in inner loops
NUMA locality on large systems
Reducing scatter/gather overhead
Minimizing delayed pivot reallocation
```

Implementation requirements:

```text
Use column-major dense frontal storage.
Align dense buffers to 64 bytes.
Use std::pmr or custom arena for per-factorization allocations.
Avoid std::map/unordered_map in numeric kernels.
Precompute scatter maps during analyse().
Use int32 indices by default; support int64 later.
```

Frontal matrix class:

```cpp
class FrontalMatrix {
public:
    FrontalMatrix(int nrows, int ncols, AlignedArena& arena);

    double* data() noexcept;
    const double* data() const noexcept;

    double& operator()(int i, int j) noexcept {
        return data_[j * ld_ + i];
    }

    int ld() const noexcept { return ld_; }
    int rows() const noexcept { return rows_; }
    int cols() const noexcept { return cols_; }

private:
    int rows_;
    int cols_;
    int ld_;
    double* data_;
};
```

---

## 9. Memory Management

Implement memory prediction in analysis:

```text
predicted factor entries
predicted flops
max front size
max supernode size
tree depth
```

Use:

```cpp
class FactorWorkspace {
public:
    explicit FactorWorkspace(std::size_t bytes);
    void* allocate(std::size_t bytes, std::size_t alignment = 64);
    void reset_node_scope();
    void clear();
};
```

Required behavior:

```text
Preallocate predicted memory * control.factor_memory_multiplier.
If exceeded, grow by 1.5x.
Track number of grows.
Expose memory diagnostics.
```

---

## 10. Error Handling and Diagnostics

No exceptions in hot numeric kernels. Use status codes internally.

Public API may throw only for programmer errors such as wrong array size.

```cpp
enum class ErrorCode {
    Ok,
    InvalidN,
    InvalidColPtr,
    InvalidRowIndex,
    DuplicateInUncheckedInput,
    OrderingFailed,
    SymbolicFailed,
    FactorNonPositivePivot,
    FactorSingular,
    FactorOutOfMemory,
    SolveBeforeFactor,
    DimensionMismatch
};
```

Diagnostics:

```text
flag/status
duplicates
out-of-range entries
missing diagonals
chosen ordering
structural rank
numerical rank
number of delayed pivots
number of 1x1 pivots
number of 2x2 pivots
inertia
factor entries
flops
peak memory
timings
```

---

## 11. Testing Plan

### 11.1 Matrix Input Tests

```text
valid lower CSC
upper triangle rejection
duplicate summation
missing diagonal
bad col_ptr
empty matrix
1x1 matrix
2x2 matrix
singular matrix
```

### 11.2 SPD Correctness

Generate SPD matrices:

```text
A = B Bᵀ + αI
2D Poisson
block diagonal SPD
random sparse SPD
```

Check:

```text
relative residual ||Ax-b|| / (||A||||x|| + ||b||) < 1e-10
matches Eigen/SuiteSparse/CHOLMOD on small cases
fails cleanly on non-SPD when matrix type is SPD
```

### 11.3 Indefinite Correctness

Test:

```text
small hand-computed indefinite matrices
KKT matrices
saddle-point systems
singular systems
near-singular pivots
matrices requiring 2x2 pivots
matrices causing delayed pivots
```

Check:

```text
relative residual < 1e-9 to 1e-10 when nonsingular
inertia equals Eigen dense LDLT for small matrices
rank detection reasonable
```

### 11.4 Repeated Factorization

Same sparsity, different values:

```cpp
auto analysis = solver.analyse(pattern, true);
for (int k = 0; k < 100; ++k) {
    auto factor = solver.factor(analysis, A_k, MatrixType::RealSymmetricIndefinite);
    solver.solve(analysis, factor, nrhs, b, ldb, x, ldx);
}
```

### 11.5 Parallel Tests

Run with:

```text
OMP_NUM_THREADS=1
OMP_NUM_THREADS=2
OMP_NUM_THREADS=4
OMP_NUM_THREADS=8
OMP_NUM_THREADS=16
```

Check:

```text
residual stability
same pivot sequence in deterministic mode if possible
same inertia
same numerical rank
no data races under ThreadSanitizer where feasible
```

### 11.6 Performance Benchmarks

Benchmark against:

```text
Eigen SimplicialLDLT for small SPD
SuiteSparse CHOLMOD for SPD
MUMPS/PARDISO if available
IPOPT linear solver timing if adapter is implemented
```

Datasets:

```text
SuiteSparse Matrix Collection
synthetic KKT systems
trajectory optimization block-KKT systems
2D/3D Poisson
random sparse indefinite systems
```

Metrics:

```text
analyse time
factor time
solve time
factor memory
flop rate
delayed pivots
front sizes
parallel speedup
residual
inertia correctness
```

---

## 12. CMake and Dependencies

Use:

```text
C++20
OpenMP
BLAS/LAPACK: OpenBLAS/MKL/BLIS + LAPACK
SuiteSparse AMD
METIS
GoogleTest
```

CMake sketch:

```cmake
cmake_minimum_required(VERSION 3.22)
project(smf_solver LANGUAGES CXX)

option(SMF_ENABLE_OPENMP "Enable OpenMP" ON)
option(SMF_USE_MKL "Use Intel MKL" OFF)
option(SMF_USE_METIS "Use METIS" ON)
option(SMF_USE_SUITESPARSE_AMD "Use SuiteSparse AMD" ON)

add_library(smf_solver
    src/csc_matrix.cpp
    src/check_matrix.cpp
    src/ordering_amd.cpp
    src/ordering_metis.cpp
    src/symbolic_analysis.cpp
    src/etree.cpp
    src/assembly_tree.cpp
    src/supernode_detection.cpp
    src/factor_posdef.cpp
    src/factor_indef.cpp
    src/pivoting.cpp
    src/frontal_matrix.cpp
    src/solve.cpp
    src/scaling.cpp
    src/dense_kernel_lapack.cpp
    src/diagnostics.cpp
)

target_compile_features(smf_solver PUBLIC cxx_std_20)
target_include_directories(smf_solver PUBLIC include)

if(SMF_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(smf_solver PUBLIC OpenMP::OpenMP_CXX)
endif()

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
target_link_libraries(smf_solver PUBLIC BLAS::BLAS LAPACK::LAPACK)
```

---

## 13. Milestones

### Milestone 1 — Dense Correctness Core

Goal:

```text
Implement dense SPD Cholesky and dense indefinite LDLᵀ with inertia.
```

Deliverables:

```text
Dense LDLᵀ unit tests
1x1/2x2 pivot support
Inertia tests against Eigen
Residual tests
```

Do not touch sparse multifrontal yet.

---

### Milestone 2 — Sparse Input and Symbolic Analysis

Goal:

```text
Clean CSC input, compute ordering, etree, supernodes, assembly tree.
```

Deliverables:

```text
CscLower checker
AMD ordering
METIS ordering
Elimination tree
Postorder traversal
Supernode grouping
Memory/flop prediction
```

---

### Milestone 3 — Left-Looking/Simple Multifrontal SPD

Goal:

```text
Implement SPD sparse Cholesky through frontal nodes.
```

Deliverables:

```text
factor SPD
solve SPD
2D Poisson benchmark
Repeated factorization
```

---

### Milestone 4 — Indefinite Frontal LDLᵀ

Goal:

```text
Implement sparse symmetric indefinite factorization with delayed pivots.
```

Deliverables:

```text
1x1/2x2 pivots inside fronts
delayed pivots to parent
inertia
KKT test suite
singularity handling
```

This is the hardest milestone.

---

### Milestone 5 — Robust Solve Phase

Goal:

```text
Production-grade solve for multiple RHS.
```

Deliverables:

```text
forward solve
D block solve
backward solve
multiple RHS BLAS-3 path
scaling/permutation correctness
```

---

### Milestone 6 — Parallel Factorization

Goal:

```text
OpenMP task parallelism over assembly tree.
```

Deliverables:

```text
tree task scheduling
BLAS threading guard
deterministic assembly order
parallel benchmark
thread-count tests
```

---

### Milestone 7 — Scaling and Ordering Improvements

Goal:

```text
Improve robustness for hard indefinite systems.
```

Deliverables:

```text
equilibration scaling
matching-based scaling/order research prototype
auto ordering heuristic
delayed-pivot memory tuning
```

---

### Milestone 8 — IPOPT/Custom Optimizer Integration

Goal:

```text
Expose a custom linear solver interface for optimization workloads.
```

Deliverables:

```text
C ABI wrapper
IPOPT linear solver adapter if required
inertia reporting
regularization support
warm repeated factorization
trajectory-optimization benchmark
```

---

## 14. Minimal First Implementation Target

Do **not** attempt everything at once.

First useful version:

```text
double only
lower CSC only
AMD ordering
serial multifrontal SPD
serial indefinite LDLᵀ
multiple RHS solve
inertia
basic scaling none/user
```

Then add:

```text
METIS
OpenMP tree parallelism
equilibration scaling
delayed pivot optimization
BLAS-3 solve path
```

---

## 15. Definition of Done

The solver is acceptable for first serious internal use when:

```text
1. It solves SPD sparse matrices with residual < 1e-10.
2. It solves small/medium indefinite KKT matrices with residual < 1e-9.
3. It reports correct inertia on dense-checkable matrices.
4. It supports analyse once, factor many.
5. It handles missing diagonal, duplicate input, and invalid input safely.
6. It gives useful diagnostics: rank, inertia, delayed pivots, factor entries, flops, timing.
7. It has deterministic single-thread behavior.
8. It is benchmarked against at least Eigen and one established sparse solver.
```

---

## 16. Engineering Warning

The implementation risk is concentrated in these areas:

```text
Indefinite pivoting stability
Delayed pivot propagation
Memory layout for contribution blocks
Correct assembly/scatter maps
Inertia correctness
Parallel task scheduling without races
Avoiding catastrophic fill-in
```

The agent should implement this incrementally and preserve a working solver at every milestone.

A toy sparse LDLᵀ can be written quickly; an MA97-class solver is a serious numerical software project.

---

## 17. Recommended Agent Execution Rules

The implementation agent must follow these rules:

```text
1. Never implement multiple milestones at once.
2. Keep every milestone buildable and testable.
3. Add tests before optimizing.
4. Use dense reference solvers for small correctness cases.
5. Never hide numerical failures.
6. Report residual, inertia, pivot counts, delayed pivots, and memory for every benchmark.
7. Prefer correctness and diagnostics before parallel performance.
8. Avoid clever sparse data structures until the algorithm is validated.
9. Preserve deterministic serial behavior at all times.
10. Document every pivoting and scaling decision.
```

---

## 18. Suggested First Agent Prompt

Use this prompt to start implementation:

```text
You are implementing a C++20 sparse symmetric multifrontal solver inspired by HSL_MA97.

Start with Milestone 1 only.

Implement:
- Dense SPD Cholesky wrapper using LAPACK dpotrf.
- Dense symmetric indefinite LDLᵀ prototype with 1x1 and 2x2 pivots.
- Inertia calculation for 1x1 and 2x2 D blocks.
- Residual checker.
- GoogleTest unit tests comparing small matrices against Eigen dense solutions.

Do not implement sparse analysis yet.
Do not implement OpenMP yet.
Do not implement METIS/AMD yet.

The code must be clean C++20, testable, and organized under the proposed project layout.
```
