# MA97-Like Sparse Symmetric Solver — Full Implementation Plan (C++)

---

## 0. Overview

Build a **multifrontal sparse symmetric solver** (positive-definite and indefinite) in modern C++17/20, matching the feature set of HSL MA97:
- Analyse phase (symbolic, ordering)
- Numerical factorization (Cholesky / LDLᵀ)
- Solve phase (forward/backward substitution)
- OpenMP parallelism with bit-compatible results
- Scaling support (MC64/MC77 analogues)
- Both CSC and coordinate input formats

---

## 1. Repository & Build Structure

```
solver/
├── CMakeLists.txt
├── include/
│   └── solver/
│       ├── types.hpp          # Core type aliases, enums
│       ├── control.hpp        # SolverControl struct
│       ├── info.hpp           # SolverInfo struct
│       ├── akeep.hpp          # AnalysisKeep (symbolic data)
│       ├── fkeep.hpp          # FactorKeep (numerical data)
│       └── solver.hpp         # Public API (free functions)
├── src/
│   ├── analyse/
│   │   ├── check.cpp          # Input validation, CSC/coord conversion
│   │   ├── ordering_amd.cpp   # AMD ordering
│   │   ├── ordering_metis.cpp # METIS wrapper
│   │   ├── ordering_match.cpp # Matching-based ordering
│   │   ├── etree.cpp          # Elimination tree construction
│   │   ├── supernode.cpp      # Supernodal amalgamation
│   │   └── assembly_tree.cpp  # Assembly tree + symbolic fronts
│   ├── factor/
│   │   ├── factorize.cpp      # Driver
│   │   ├── front_posdef.cpp   # Dense Cholesky kernel (potrf wrapper)
│   │   ├── front_indef.cpp    # Dense LDLᵀ kernel with Bunch-Kaufman
│   │   ├── stack.cpp          # Contribution block stack manager
│   │   ├── pivot.cpp          # Pivot selection (1×1 and 2×2)
│   │   └── scaling.cpp        # MC64/MC77/MC30 analogues
│   ├── solve/
│   │   ├── solve.cpp          # Forward + backward substitution driver
│   │   ├── solve_supernodal.cpp
│   │   ├── solve_multifrontal.cpp
│   │   └── solve_sparse.cpp   # Sparse forward solve
│   ├── parallel/
│   │   ├── task_tree.cpp      # OpenMP task-based tree traversal
│   │   └── task_node.cpp      # Node-level parallelism
│   └── utils/
│       ├── memory.cpp         # Aligned allocation, pool allocator
│       ├── blas_wrap.cpp      # BLAS/LAPACK wrappers
│       └── logger.cpp
├── tests/
│   ├── unit/
│   └── integration/
└── benchmarks/
```

---

## 2. Core Data Structures

### 2.1 Public Types (`types.hpp`)
```cpp
#pragma once
#include <cstdint>

namespace solver {

enum class MatrixType {
    RealPosDef  =  3,
    RealIndef   =  4,
    HermPosDef  = -3,
    HermIndef   = -4,
    ComplexIndef = -5
};

enum class SolveJob {
    Full       = 0,   // AX = B
    Forward    = 1,   // PLX = SB
    DiagOnly   = 2,   // DX = B (indef)
    Backward   = 3,   // (PL)ᵀS⁻¹X = B
    DiagBack   = 4    // D(PL)ᵀS⁻¹X = B (indef)
};

enum class OrderingMethod {
    UserSupplied = 0,
    AMD          = 1,
    MinDeg       = 2,
    METIS        = 3,
    MA47         = 4,
    AutoParallel = 5,
    AutoSerial   = 6,
    MatchAMD     = 7,
    MatchMETIS   = 8
};

using Int    = int32_t;
using LongInt = int64_t;
using Real   = double;

} // namespace solver
```

### 2.2 SolverControl (`control.hpp`)
```cpp
struct SolverControl {
    // Printing
    int  print_level       = 0;
    int  unit_diagnostics  = 6;  // stderr fd
    int  unit_error        = 6;
    int  unit_warning      = 6;

    // Analyse
    OrderingMethod ordering = OrderingMethod::AutoParallel;
    int  nemin             = 8;  // Node amalgamation threshold

    // Factor
    int  scaling           = 0;
    bool action            = true;  // Continue on singularity
    LongInt factor_min     = 20'000'000LL;
    Real multiplier        = 1.1;
    Real small             = 1e-20;
    Real u                 = 0.01;  // Pivot tolerance

    // Solve
    Real consist_tol       = std::numeric_limits<Real>::epsilon();
    bool solve_mf          = false;
    bool solve_blas3       = false;
    LongInt solve_min      = 100'000LL;
};
```

### 2.3 Assembly Tree Node
```cpp
struct SuperNode {
    Int  id;
    Int  parent;                  // -1 = root
    Int  ncols;                   // # fully summed (pivot) columns
    Int  nrows;                   // total rows in front
    std::vector<Int> col_map;     // global col indices
    std::vector<Int> row_map;     // global row indices
    std::vector<Int> children;
    LongInt contrib_offset;       // offset into contribution stack
    // Numerical data (filled during factor)
    std::vector<Real> L_data;     // column-major dense block
    std::vector<Real> D_data;     // 2-array: D⁻¹ diagonal + off-diag
    bool is_delayed = false;
    int  num_delayed = 0;
};
```

### 2.4 AnalysisKeep (`akeep.hpp`)
```cpp
struct AnalysisKeep {
    Int n;
    // CSC data (checked & cleaned)
    std::vector<Int>  ptr;
    std::vector<Int>  row;
    std::vector<Real> val;           // present only if match-based ordering
    // Ordering
    std::vector<Int>  perm;          // perm[i] = position of variable i
    std::vector<Int>  iperm;         // inverse permutation
    // Supernodal structure
    std::vector<SuperNode> snodes;
    Int num_sup;
    Int maxfront;
    Int maxsupernode;
    Int maxdepth;
    LongInt num_factor;              // predicted #entries in L
    LongInt num_flops;               // predicted flop count
    Int matrix_rank;
    // Scaling (if match-based)
    std::vector<Real> scale;
};
```

### 2.5 FactorKeep (`fkeep.hpp`)
```cpp
struct FactorKeep {
    MatrixType matrix_type;
    std::vector<Real> L_values;      // packed L factor storage
    std::vector<Real> D_values;      // 2n entries (D⁻¹)
    std::vector<Int>  piv_order;
    std::vector<Real> scale;
    LongInt num_factor;
    LongInt num_flops;
    Int matrix_rank;
    Int num_neg;
    Int num_two;
    Int num_delay;
    Int maxfront;
    Int maxsupernode;
};
```

---

## 3. Phase 1 — Analyse

### 3.1 Input Validation & Cleaning

```cpp
// src/analyse/check.cpp
void check_csc(Int n, const Int* ptr, const Int* row,
               std::vector<Int>& clean_ptr,
               std::vector<Int>& clean_row,
               SolverInfo& info);
```

- Walk CSC structure, accumulate lower-triangular only
- Sum duplicates, discard out-of-range
- Report counts in `info.matrix_dup`, `info.matrix_outrange`
- For coordinate input: sort by column, convert to CSC first

### 3.2 Elimination Ordering

**AMD (Approximate Minimum Degree)**
```cpp
// src/analyse/ordering_amd.cpp
// Implement Amestoy-Davis-Duff AMD (1996)
// Reference: ACM TOMS 22(3), 1996
void amd_order(Int n, const Int* ptr, const Int* row,
               std::vector<Int>& perm);
```

Key AMD steps:
1. Build adjacency graph from lower triangle
2. Maintain degree lists with bucket sort
3. Eliminate min-degree node, update neighbors' degrees
4. Use approximate degree: d(v) ≈ |Le(v) ∪ Lp(v)| − 1
5. Apply mass elimination for indistinguishable nodes

**METIS Wrapper**
```cpp
// src/analyse/ordering_metis.cpp
// Wrapper around METIS_NodeND
void metis_order(Int n, const Int* ptr, const Int* row,
                 std::vector<Int>& perm,
                 std::vector<Int>& iperm);
```

**Matching-Based (MC80 analogue)**
```cpp
// Use maximum weighted matching to find good pivots
// before ordering; requires matrix values
void matching_order(Int n, const Int* ptr, const Int* row,
                    const Real* val,
                    std::vector<Int>& perm,
                    std::vector<Real>& scale);
// Algorithm: Duff & Koster (2001) maximum weight matching
// Augmenting path method on bipartite graph
```

### 3.3 Elimination Tree Construction

```cpp
// src/analyse/etree.cpp
// Compute elimination tree from symbolic Cholesky
// Algorithm: Liu (1986), O(n α(n))
void build_etree(Int n, const Int* ptr, const Int* row,
                 const Int* perm, const Int* iperm,
                 std::vector<Int>& parent,
                 std::vector<Int>& first_child,
                 std::vector<Int>& next_sibling);
```

Steps:
1. Apply fill-reducing permutation
2. Symbolic Cholesky factorization using path compression
3. Record `parent[j]` = first row index > j in column j of L

### 3.4 Supernode Detection & Amalgamation

```cpp
// src/analyse/supernode.cpp
// Fundamental supernodes: consecutive cols j, j+1 where
//   - parent(j) == j+1
//   - |struct(col j+1)| == |struct(col j)| + 1
// Then amalgamate: merge nodes where both have < nemin cols
void find_supernodes(Int n, const std::vector<Int>& parent,
                     const std::vector<Int>& col_counts,
                     Int nemin,
                     std::vector<Int>& snode_id,  // col -> supernode
                     std::vector<SuperNode>& snodes);
```

### 3.5 Symbolic Assembly Tree

```cpp
// src/analyse/assembly_tree.cpp
// For each supernode: compute row_map (union of children's
// non-fully-summed rows + own pivot rows), post-order traversal
void build_assembly_tree(Int n,
                         const Int* ptr, const Int* row,
                         const Int* perm, const Int* iperm,
                         std::vector<SuperNode>& snodes,
                         SolverInfo& info);
```

Output statistics: `num_factor`, `num_flops`, `maxfront`, `maxdepth`, `num_sup`.

---

## 4. Phase 2 — Numerical Factorization

### 4.1 Memory Layout & Contribution Stack

Two stacks alternate (as in MA97) to avoid copying:

```cpp
// src/factor/stack.cpp
class ContribStack {
    std::vector<Real> buffer_a, buffer_b;
    size_t top_a = 0, top_b = 0;
    bool use_a = true;
public:
    Real* alloc(size_t n);
    void  free_top(size_t n);
    void  flip();   // swap active stack
};
```

Rationale: When computing a node's contribution from children, reading from one stack and writing the merged result to the other avoids in-place aliasing.

### 4.2 Positive-Definite Front (Cholesky)

```cpp
// src/factor/front_posdef.cpp
// Front matrix F is nrows × nrows (column-major)
// Pivot block: ncols × ncols (top-left corner)
void factor_front_posdef(SuperNode& node,
                         Real* F,       // nrows × nrows front
                         int  ncols,
                         int  nrows,
                         SolverInfo& info) {
    // 1. LAPACK dpotrf on top-left ncols × ncols block
    //    → produces Cholesky factor L11
    int info_lapack = 0;
    dpotrf("L", &ncols, F, &nrows, &info_lapack);
    if (info_lapack > 0) { info.flag = -8; return; }

    // 2. BLAS dtrsm: solve L11 * L21ᵀ = A21
    //    (update subdiagonal block)
    dtrsm("R","L","T","N", &nrows_sub, &ncols,
          &one, F, &nrows, F+ncols, &nrows);

    // 3. BLAS dsyrk: update Schur complement
    //    C22 -= L21 * L21ᵀ
    dsyrk("L","N", &nrows_sub, &ncols,
          &m_one, F+ncols, &nrows,
          &one, F+ncols*(1+nrows), &nrows);

    // 4. Store L11 and L21 into node.L_data
}
```

### 4.3 Indefinite Front (LDLᵀ with Bunch-Kaufman pivoting)

```cpp
// src/factor/front_indef.cpp
// Uses the same pivoting strategy as HSL MA64
// 1×1 pivot: |a_jj| >= u * max_col
// 2×2 pivot: off-diagonal block is used when 1×1 fails

struct PivotResult {
    int  pivot_type;   // 1 or 2
    bool delayed;
};

PivotResult try_pivot_1x1(Real* F, int j, int nrows,
                           Real u, Real small);
PivotResult try_pivot_2x2(Real* F, int j, int nrows,
                           Real u, Real small);

void factor_front_indef(SuperNode& node,
                        Real* F,
                        int ncols, int nrows,
                        const SolverControl& ctrl,
                        SolverInfo& info);
```

**Pivot test (Bunch-Kaufman)**:
```
α = (1 + √17) / 8 ≈ 0.6404
1×1 pivot at position j if |F[j,j]| ≥ α * max_{i>j} |F[i,j]|
else 2×2 pivot on rows/cols {j, p} where p = argmax |F[i,j]|
```

**Delayed pivots**: If no suitable pivot found at node, pass column(s) to parent. Track in `node.num_delayed`.

### 4.4 Assembly (Scatter-Add)

```cpp
// Scatter child contribution into parent front
// Uses node.row_map for index translation
void assemble_contribution(const SuperNode& child,
                            SuperNode& parent,
                            Real* parent_front,
                            const Real* child_contrib);
```

### 4.5 Factorization Driver

```cpp
// src/factor/factorize.cpp
void factorize(const AnalysisKeep& akeep,
               FactorKeep& fkeep,
               const Real* val,
               const SolverControl& ctrl,
               SolverInfo& info) {
    // Post-order traversal of assembly tree
    // For each supernode in post-order:
    //   1. Allocate front from stack
    //   2. Scatter original matrix entries
    //   3. Assemble contributions from children
    //   4. Factor the pivot block (posdef or indef)
    //   5. Compute contribution block (Schur complement)
    //   6. Free children's contribution memory
    //   7. Store L data into fkeep
}
```

---

## 5. Phase 3 — Solve

### 5.1 Full Solve Driver

```cpp
// AX = B  ⟺  S⁻¹ P L D Lᵀ Pᵀ S⁻¹ X = B
// Steps:
//   1. Apply scaling:  B ← S⁻¹ B  (if scaling enabled)
//   2. Forward subst:  L y = P S⁻¹ B   (job=1)
//   3. Diagonal solve: D z = y          (job=2, indef only)
//   4. Backward subst: Lᵀ x = z        (job=3)
//   5. Apply scaling:  X ← S⁻¹ x  (if scaling enabled)
void solve(Int nrhs, Real* X, Int ldx,
           const AnalysisKeep& akeep,
           const FactorKeep& fkeep,
           const SolverControl& ctrl,
           SolverInfo& info,
           SolveJob job = SolveJob::Full);
```

### 5.2 Supernodal Forward Solve

```cpp
// Traverse supernodes in post-order
// For each node: BLAS dtrsv / dtrsm on dense L block
// Direct updates into X (no stack)
// Does NOT parallelize easily (data dependency)
void forward_supernodal(Int nrhs, Real* X, ...);
```

### 5.3 Multifrontal Forward Solve

```cpp
// Uses the contribution stack to pass partial sums up
// Enables task-parallel forward solve
// For each node: apply L⁻¹, push result to parent via stack
void forward_multifrontal(Int nrhs, Real* X, ...);
```

### 5.4 Backward Solve

```cpp
// Both supernodal and multifrontal use same backward pass
// Traverse in reverse post-order
// Bit-compatible parallel backward solve is possible
// (see MA97 paper Section 4)
void backward_solve(Int nrhs, Real* X, ...);
```

### 5.5 Diagonal Solve (Indefinite)

```cpp
// Apply D⁻¹ (block diagonal, 1×1 and 2×2 blocks)
void diagonal_solve(Int nrhs, Real* X, Int ldx,
                    const Real* D,         // 2×n array
                    const Int*  piv_order, // signs encode 2×2
                    Int n);
```

### 5.6 Sparse Forward Solve

```cpp
// Solve PLX = SB for sparse B
// Only visit nodes reachable from B's nonzero pattern
// Uses topological ordering of the elimination tree
void sparse_fwd_solve(Int nbi, const Int* bindex,
                      const Real* b, const Int* order,
                      bool* lflag,
                      Int& nxi, Int* xindex, Real* x, ...);
```

---

## 6. Scaling

### 6.1 MC64 Analogue (Maximum Weighted Bipartite Matching)

```cpp
// scaling=1: minimise max |log|a_ij| + r_i + c_j|| 
// Result: S = diag(exp(r)), apply as A ← S A S
// Algorithm: auction algorithm or Hungarian method
void scale_mc64(Int n, const Int* ptr, const Int* row,
                const Real* val, std::vector<Real>& scale);
```

### 6.2 MC77 Analogue (Iterative Scaling)

```cpp
// scaling=2: equilibrate row/column infinity norms
// 1 iteration in ∞-norm, 3 in 1-norm
void scale_mc77(Int n, const Int* ptr, const Int* row,
                const Real* val, std::vector<Real>& scale);
```

### 6.3 MC30 Analogue (Log-sum Minimization)

```cpp
// scaling=4: minimise Σ |log|s_i a_ij s_j||
// Sinkhorn-Knopp iteration on log scale
void scale_mc30(Int n, const Int* ptr, const Int* row,
                const Real* val, std::vector<Real>& scale);
```

---

## 7. Parallelism (OpenMP)

### 7.1 Tree-Level Parallelism

```cpp
// src/parallel/task_tree.cpp
// Each subtree without a ready parent is an independent task
// Use OpenMP 3.0 tasks:

void factor_parallel(std::vector<SuperNode>& snodes,
                     const SolverControl& ctrl) {
    #pragma omp parallel
    #pragma omp single
    {
        for (auto& node : postorder_leaves(snodes)) {
            #pragma omp task depend(out: node.L_data)        \
                             depend(in: children_tasks...)
            {
                factor_node(node, ctrl);
            }
        }
    }
}
```

**Bit-compatibility**: Fix child assembly order at analysis time. Each thread processes the same sub-operations in the same sequence → identical floating-point results regardless of thread count.

### 7.2 Node-Level Parallelism

For large fronts (num_flops ≥ `factor_min`):
```cpp
// Split the dsyrk / outer-product into independent column strips
// Each strip is a separate task
// Bit-compat: per-strip partial sums summed in fixed order

#pragma omp task for schedule(static)
for (int col = 0; col < ncols; col += BLOCK_SIZE) {
    update_schur_block(F, col, col+BLOCK_SIZE, nrows);
}
// Reduction in fixed order (not omp reduction)
```

### 7.3 Thread-Safe Memory

- Use per-thread arenas for contribution blocks
- Global pool allocator with mutex for FactorKeep L_data allocation
- Avoid false sharing: align contribution blocks to cache line (64 bytes)

---

## 8. Public API (C++ Free Functions)

```cpp
// include/solver/solver.hpp
namespace solver {

// Analyse: CSC format
void analyse(bool check, Int n,
             const Int* ptr, const Int* row,
             AnalysisKeep& akeep,
             const SolverControl& ctrl,
             SolverInfo& info,
             Int* order = nullptr,       // optional user ordering
             const Real* val = nullptr); // needed for matching order

// Analyse: coordinate format
void analyse_coord(Int n, Int ne,
                   const Int* row, const Int* col,
                   AnalysisKeep& akeep,
                   const SolverControl& ctrl,
                   SolverInfo& info,
                   Int* order = nullptr,
                   const Real* val = nullptr);

// Factorize
void factorize(MatrixType mtype,
               const Real* val,
               const AnalysisKeep& akeep,
               FactorKeep& fkeep,
               const SolverControl& ctrl,
               SolverInfo& info,
               Real* scale = nullptr,
               const Int* ptr = nullptr,
               const Int* row = nullptr);

// Factorize + solve (combined)
void factorize_solve(MatrixType mtype,
                     const Real* val,
                     Int nrhs, Real* X, Int ldx,
                     const AnalysisKeep& akeep,
                     FactorKeep& fkeep,
                     const SolverControl& ctrl,
                     SolverInfo& info,
                     Real* scale = nullptr);

// Solve
void solve(Int nrhs, Real* X, Int ldx,
           const AnalysisKeep& akeep,
           const FactorKeep& fkeep,
           const SolverControl& ctrl,
           SolverInfo& info,
           SolveJob job = SolveJob::Full);

// Enquire pivots (posdef)
void enquire_posdef(const AnalysisKeep& akeep,
                    const FactorKeep& fkeep,
                    const SolverControl& ctrl,
                    SolverInfo& info,
                    Real* d);

// Enquire pivots (indef)
void enquire_indef(const AnalysisKeep& akeep,
                   const FactorKeep& fkeep,
                   const SolverControl& ctrl,
                   SolverInfo& info,
                   Int* piv_order = nullptr,
                   Real* d = nullptr);

// Alter D⁻¹
void alter(const Real* d,
           const AnalysisKeep& akeep,
           FactorKeep& fkeep,
           const SolverControl& ctrl,
           SolverInfo& info);

// Fredholm solve (singular indefinite)
void solve_fredholm(Int nrhs, bool* flag_out,
                    Real* X, Int ldx,
                    const AnalysisKeep& akeep,
                    const FactorKeep& fkeep,
                    const SolverControl& ctrl,
                    SolverInfo& info);

// L-multiply: Y = S⁻¹PLX or Y = (S⁻¹PL)ᵀX
void lmultiply(bool trans, Int k,
               const Real* X, Int ldx,
               Real* Y, Int ldy,
               const AnalysisKeep& akeep,
               const FactorKeep& fkeep,
               const SolverControl& ctrl,
               SolverInfo& info);

// Sparse forward solve
void sparse_fwd_solve(Int nbi, const Int* bindex,
                      const Real* b, const Int* order,
                      bool* lflag,
                      Int& nxi, Int* xindex, Real* x,
                      const AnalysisKeep& akeep,
                      const FactorKeep& fkeep,
                      const SolverControl& ctrl,
                      SolverInfo& info);

// Memory management
void free_akeep(AnalysisKeep& akeep);
void free_fkeep(FactorKeep& fkeep);
void finalize(AnalysisKeep& akeep, FactorKeep& fkeep);

} // namespace solver
```

---

## 9. BLAS/LAPACK Strategy

Always link against optimized BLAS (OpenBLAS, MKL, BLIS). Wrap in thin layer:

```cpp
// src/utils/blas_wrap.cpp
// Provides: solver_dpotrf, solver_dsyrk, solver_dtrsm,
//           solver_dgemm, solver_dgemv, solver_dtrsv, solver_dtrmm

#ifdef USE_MKL
#  include <mkl.h>
#elif defined USE_OPENBLAS
#  include <cblas.h>
#  include <lapacke.h>
#endif
```

**Key operations by phase**:

| Phase     | BLAS routine | Purpose                              |
|-----------|-------------|--------------------------------------|
| Factor PD | `dpotrf`    | Dense Cholesky on pivot block        |
| Factor PD | `dtrsm`     | Solve L₁₁ * L₂₁ᵀ = A₂₁             |
| Factor PD | `dsyrk`     | Schur complement update              |
| Factor ID | `dgemm`     | Off-diagonal block updates           |
| Factor ID | `dger`/`dsyr`| Rank-1/rank-2 pivot updates         |
| Solve     | `dtrsv`     | Single RHS triangular solve          |
| Solve     | `dtrsm`     | Multiple RHS triangular solve        |
| Solve     | `dgemv`/`dgemm` | Update right-hand sides          |

---

## 10. CPU & Memory Optimization

### 10.1 Cache-Friendly Storage

- Store L factor in **column-major** dense supernodal blocks (optimal for BLAS)
- Keep supernode's dense front in **contiguous memory** (single `std::vector<Real>`)
- Align all large allocations to 64-byte cache lines:
  ```cpp
  Real* alloc_aligned(size_t n) {
      void* p;
      posix_memalign(&p, 64, n * sizeof(Real));
      return static_cast<Real*>(p);
  }
  ```

### 10.2 Loop Tiling for Schur Complement

For large fronts, tile the dsyrk to maximize L2/L3 cache reuse:
```cpp
constexpr int TILE = 64; // tune per architecture
for (int i = 0; i < nrows_sub; i += TILE)
    for (int k = 0; k < ncols; k += TILE)
        dsyrk_tile(F, i, k, TILE, nrows);
```

### 10.3 NUMA Awareness (multi-socket)

- Pin threads to NUMA nodes with `numactl`
- Allocate contribution blocks on the NUMA node of the thread that will process them
- Use `libnuma` or `hwloc` for topology detection

### 10.4 Memory Pooling

```cpp
// Pre-allocate large pool, sub-allocate contribution blocks
// Avoids malloc/free overhead in hot loop
class FrontPool {
    std::vector<std::byte> pool;
    std::atomic<size_t>    cursor{0};
public:
    explicit FrontPool(size_t bytes) : pool(bytes) {}
    Real* alloc(size_t n);
    void  reset();  // O(1): just reset cursor
};
```

### 10.5 Vectorization

Mark inner loops with `#pragma omp simd` or use AVX2 intrinsics for the pivot update kernel. Example for 2×2 pivot update:
```cpp
#include <immintrin.h>
// Use __m256d for 4-wide double updates
```

---

## 11. Implementation Phases & Priority

| Phase | What | Effort |
|-------|------|--------|
| P1 | Types, control, info, CSC check | 1 week |
| P2 | AMD ordering | 1 week |
| P3 | Elimination tree + supernodes | 1 week |
| P4 | Assembly tree (symbolic) | 1 week |
| P5 | Positive-definite factorization (serial) | 2 weeks |
| P6 | Solve (supernodal, serial) | 1 week |
| P7 | Indefinite factorization + Bunch-Kaufman | 2 weeks |
| P8 | MC64/MC77 scaling | 1 week |
| P9 | OpenMP tree-level parallelism | 1 week |
| P10 | Node-level parallelism + bit-compat | 1 week |
| P11 | METIS integration | 0.5 week |
| P12 | Matching-based ordering | 1.5 weeks |
| P13 | Advanced solve (Fredholm, sparse, lmultiply) | 1 week |
| P14 | Benchmarking, tuning, docs | 1 week |

**Total: ~16 weeks for full feature parity**

---

## 12. Testing Strategy

### 12.1 Unit Tests (catch2 / googletest)

- Ordering: compare AMD output against reference (AMD library)
- Elimination tree: verify `parent` array against brute-force symbolic Cholesky
- Supernode detection: check amalgamation counts
- Factor: factorize known 5×5 examples from spec, verify L*Lᵀ = A
- Pivot: test Bunch-Kaufman selection on hand-crafted indefinite 3×3
- Solve: verify `||Ax - b||/||b|| < 1e-12` on all solve paths

### 12.2 Integration Tests

- Run all three examples from the spec PDF, match outputs exactly
- Test with SuiteSparse matrix collection (Florida Sparse Matrix Collection)
- Verify bit-compatibility: run serial vs. 4-thread, compare bit-for-bit

### 12.3 Regression Suite

- Keep a library of 50+ matrices spanning SPD/indefinite/singular
- Run with all ordering methods, check backward error after every change

---

## 13. Key References

1. Hogg & Scott, "HSL MA97: a bit-compatible multifrontal code for sparse symmetric systems", RAL-TR-2011-024
2. Duff & Reid, "The multifrontal solution of indefinite sparse symmetric linear equations", ACM TOMS 1983
3. Amestoy, Davis & Duff, "An approximate minimum degree ordering algorithm", SIAM J. Matrix Anal. 1996
4. Karypis & Kumar, "A fast and high quality multilevel scheme for partitioning irregular graphs" (METIS), 1998
5. Duff & Koster, "On algorithms for permuting large entries to the diagonal of a sparse matrix", SIAM J. 2001
6. Anderson et al., LAPACK Users' Guide, 1999
