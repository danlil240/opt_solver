#include "smf/coo_matrix.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/solver.hpp"
#include "smf/types.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace smf;

// ---------------------------------------------------------------------------
// Helper: build expected CscLower and compare
// ---------------------------------------------------------------------------
static void expect_csc_eq(const CscLower& got,
                           int             n,
                           std::vector<Int> expected_col_ptr,
                           std::vector<Int> expected_row_idx,
                           std::vector<double> expected_val,
                           double tol = 1e-14)
{
    EXPECT_EQ(got.n, n);
    ASSERT_EQ(got.col_ptr, expected_col_ptr)
        << "col_ptr mismatch";
    ASSERT_EQ(got.row_idx.size(), expected_row_idx.size())
        << "row_idx size mismatch";
    ASSERT_EQ(got.values.size(), expected_val.size())
        << "values size mismatch";
    for (std::size_t i = 0; i < expected_row_idx.size(); ++i) {
        EXPECT_EQ(got.row_idx[i], expected_row_idx[i])
            << "row_idx[" << i << "] mismatch";
        EXPECT_NEAR(got.values[i], expected_val[i], tol)
            << "values[" << i << "] mismatch";
    }
}

// ---------------------------------------------------------------------------
// 1. RoundTrip: 3x3 lower-triangle COO → CSC
// ---------------------------------------------------------------------------
TEST(CooInput, RoundTrip) {
    // 3×3 matrix (lower triangle):
    //   [1  0  0]
    //   [2  4  0]
    //   [3  5  6]
    // Lower CSC: col 0 → rows {0,1,2}, col 1 → rows {1,2}, col 2 → rows {2}
    CooMatrix coo;
    coo.n = 3;
    coo.row = {0, 1, 2, 1, 2, 2};
    coo.col = {0, 0, 0, 1, 1, 2};
    coo.val = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    CscLower csc = coo_to_lower_csc(coo);

    expect_csc_eq(csc,
                  3,
                  {0, 3, 5, 6},          // col_ptr
                  {0, 1, 2, 1, 2, 2},    // row_idx
                  {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
}

// ---------------------------------------------------------------------------
// 2. UpperDiscarded: supply only upper-triangle → CSC nnz = 0 (no off-diag)
// ---------------------------------------------------------------------------
TEST(CooInput, UpperDiscarded) {
    // Upper-triangle only (no diagonal)
    CooMatrix coo;
    coo.n = 3;
    coo.row = {0, 0, 1};
    coo.col = {1, 2, 2};
    coo.val = {7.0, 8.0, 9.0};

    CscLower csc = coo_to_lower_csc(coo);

    EXPECT_EQ(csc.n, 3);
    EXPECT_EQ(csc.nnz(), 0);
    ASSERT_EQ(static_cast<int>(csc.col_ptr.size()), 4);
    EXPECT_EQ(csc.col_ptr[0], 0);
    EXPECT_EQ(csc.col_ptr[1], 0);
    EXPECT_EQ(csc.col_ptr[2], 0);
    EXPECT_EQ(csc.col_ptr[3], 0);
}

// ---------------------------------------------------------------------------
// 3. DuplicatesSummed: two (2,1) entries → value 3.0
// ---------------------------------------------------------------------------
TEST(CooInput, DuplicatesSummed) {
    CooMatrix coo;
    coo.n = 3;
    coo.row = {2, 2};
    coo.col = {1, 1};
    coo.val = {1.0, 2.0};

    CscLower csc = coo_to_lower_csc(coo);

    EXPECT_EQ(csc.n, 3);
    EXPECT_EQ(csc.nnz(), 1);
    // col_ptr: [0, 0, 1, 1] — col 0 empty, col 1 has 1 entry, col 2 empty
    ASSERT_EQ(static_cast<int>(csc.col_ptr.size()), 4);
    EXPECT_EQ(csc.col_ptr[0], 0);
    EXPECT_EQ(csc.col_ptr[1], 0);  // col 0: empty
    EXPECT_EQ(csc.col_ptr[2], 1);  // col 1: 1 entry (merged duplicate)
    EXPECT_EQ(csc.col_ptr[3], 1);  // col 2: empty
    // The single entry in column 1: row=2, val=3.0
    EXPECT_EQ(csc.row_idx[0], 2);
    EXPECT_NEAR(csc.values[0], 3.0, 1e-14);
}

// ---------------------------------------------------------------------------
// 4. UnsortedCOO: entries in random column order → rows sorted per column
// ---------------------------------------------------------------------------
TEST(CooInput, UnsortedCOO) {
    // 4×4 lower triangle, supplied in reverse order
    //   col 0: rows {3, 2, 1, 0}
    CooMatrix coo;
    coo.n = 4;
    coo.row = {3, 2, 1, 0};
    coo.col = {0, 0, 0, 0};
    coo.val = {40.0, 30.0, 20.0, 10.0};

    CscLower csc = coo_to_lower_csc(coo);

    EXPECT_EQ(csc.n, 4);
    EXPECT_EQ(csc.nnz(), 4);
    // Rows should be sorted: 0, 1, 2, 3
    ASSERT_EQ(static_cast<int>(csc.row_idx.size()), 4);
    EXPECT_EQ(csc.row_idx[0], 0);
    EXPECT_EQ(csc.row_idx[1], 1);
    EXPECT_EQ(csc.row_idx[2], 2);
    EXPECT_EQ(csc.row_idx[3], 3);
    EXPECT_NEAR(csc.values[0], 10.0, 1e-14);
    EXPECT_NEAR(csc.values[1], 20.0, 1e-14);
    EXPECT_NEAR(csc.values[2], 30.0, 1e-14);
    EXPECT_NEAR(csc.values[3], 40.0, 1e-14);
}

// ---------------------------------------------------------------------------
// 5. OutOfRange: entries with row=-1, col=5 (n=3), row=3 (n=3) → discarded
// ---------------------------------------------------------------------------
TEST(CooInput, OutOfRange) {
    CooMatrix coo;
    coo.n = 3;
    // Bad entries: row=-1, (row=3,col=0), (row=0,col=5)
    // Plus one valid entry: (1,0)
    coo.row = {-1, 3, 0, 1};
    coo.col = { 0, 0, 5, 0};
    coo.val = {99.0, 99.0, 99.0, 5.0};

    CscLower csc = coo_to_lower_csc(coo);

    EXPECT_EQ(csc.n, 3);
    EXPECT_EQ(csc.nnz(), 1);
    EXPECT_EQ(csc.row_idx[0], 1);
    EXPECT_NEAR(csc.values[0], 5.0, 1e-14);
}

// ---------------------------------------------------------------------------
// 6. EmptyMatrix: n=0 → CscLower{0, {0}, {}, {}}
// ---------------------------------------------------------------------------
TEST(CooInput, EmptyMatrix) {
    CooMatrix coo;
    coo.n = 0;

    CscLower csc = coo_to_lower_csc(coo);

    EXPECT_EQ(csc.n, 0);
    EXPECT_EQ(csc.nnz(), 0);
    ASSERT_EQ(static_cast<int>(csc.col_ptr.size()), 1);
    EXPECT_EQ(csc.col_ptr[0], 0);
    EXPECT_TRUE(csc.row_idx.empty());
    EXPECT_TRUE(csc.values.empty());
}

// ---------------------------------------------------------------------------
// 7. OneByOne: n=1, single diagonal entry → round-trip
// ---------------------------------------------------------------------------
TEST(CooInput, OneByOne) {
    CooMatrix coo;
    coo.n = 1;
    coo.row = {0};
    coo.col = {0};
    coo.val = {42.0};

    CscLower csc = coo_to_lower_csc(coo);

    expect_csc_eq(csc,
                  1,
                  {0, 1},   // col_ptr
                  {0},      // row_idx
                  {42.0});  // values
}

// ---------------------------------------------------------------------------
// 8. DiagonalOnly: n=5, only diagonal entries → nnz=5
// ---------------------------------------------------------------------------
TEST(CooInput, DiagonalOnly) {
    CooMatrix coo;
    coo.n = 5;
    for (int i = 0; i < 5; ++i) {
        coo.row.push_back(i);
        coo.col.push_back(i);
        coo.val.push_back(static_cast<double>(i + 1) * 10.0);
    }

    CscLower csc = coo_to_lower_csc(coo);

    EXPECT_EQ(csc.n, 5);
    EXPECT_EQ(csc.nnz(), 5);
    // col_ptr: {0,1,2,3,4,5}
    for (int j = 0; j <= 5; ++j) {
        EXPECT_EQ(csc.col_ptr[static_cast<std::size_t>(j)], j)
            << "col_ptr[" << j << "] wrong";
    }
    // Each column j has exactly one entry at row j
    for (int j = 0; j < 5; ++j) {
        EXPECT_EQ(csc.row_idx[static_cast<std::size_t>(j)], j)
            << "row_idx[" << j << "] wrong";
        EXPECT_NEAR(csc.values[static_cast<std::size_t>(j)],
                    static_cast<double>(j + 1) * 10.0, 1e-14)
            << "values[" << j << "] wrong";
    }
}

// ---------------------------------------------------------------------------
// 9. MixedUpperLower: same entry provided as both (i,j) and (j,i) upper/lower;
//    only lower-triangle kept.
// ---------------------------------------------------------------------------
TEST(CooInput, MixedUpperLower) {
    // 3×3: provide (1,0)=5 and (0,1)=5 (upper), also (2,0)=3 and (0,2)=3.
    // Plus diagonal (0,0)=10, (1,1)=20, (2,2)=30.
    // Lower result: col0→{(0,10),(1,5),(2,3)}, col1→{(1,20)}, col2→{(2,30)}
    CooMatrix coo;
    coo.n = 3;
    coo.row = {0, 1, 1, 0, 2, 0, 1, 2};
    coo.col = {0, 0, 1, 1, 0, 2, 2, 2};
    coo.val = {10.0, 5.0, 20.0, 5.0, 3.0, 3.0, 3.0, 30.0};

    CscLower csc = coo_to_lower_csc(coo);

    expect_csc_eq(csc,
                  3,
                  {0, 3, 4, 5},
                  {0, 1, 2, 1, 2},
                  {10.0, 5.0, 3.0, 20.0, 30.0});
}

// ---------------------------------------------------------------------------
// 10. IntegrationWithSolver: 5×5 tridiagonal SPD, convert, analyse+factor+solve
// ---------------------------------------------------------------------------
TEST(CooInput, IntegrationWithSolver) {
    // Build 5×5 tridiagonal SPD:
    //   A(i,i) = 4, A(i,i-1) = A(i-1,i) = -1  (i=1..4)
    const int n = 5;
    CooMatrix coo;
    coo.n = n;

    for (int i = 0; i < n; ++i) {
        coo.row.push_back(i);
        coo.col.push_back(i);
        coo.val.push_back(4.0);
        if (i > 0) {
            // sub-diagonal entry (lower triangle)
            coo.row.push_back(i);
            coo.col.push_back(i - 1);
            coo.val.push_back(-1.0);
        }
    }

    CscLower csc = coo_to_lower_csc(coo);
    ASSERT_EQ(csc.n, n);
    // 5 diagonal + 4 sub-diagonal = 9 entries
    EXPECT_EQ(csc.nnz(), 9);

    // Build RHS: b = A * x_exact where x_exact = [1,2,3,4,5]
    std::vector<double> x_exact = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> b(static_cast<std::size_t>(n), 0.0);

    // A is symmetric tridiagonal: compute A*x_exact
    for (int i = 0; i < n; ++i) {
        b[static_cast<std::size_t>(i)] += 4.0 * x_exact[static_cast<std::size_t>(i)];
        if (i > 0) {
            b[static_cast<std::size_t>(i)]     += -1.0 * x_exact[static_cast<std::size_t>(i-1)];
            b[static_cast<std::size_t>(i-1)]   += -1.0 * x_exact[static_cast<std::size_t>(i)];
        }
    }

    smf::Solver   solver;
    smf::Control  ctrl;
    smf::Info     info;
    ctrl.matrix_type = smf::MatrixType::RealSymmetricPositiveDefinite;

    auto fkeep = solver.factor_solve(csc, ctrl, info, b.data(), n, 1);
    ASSERT_NE(fkeep, nullptr) << "factor_solve returned nullptr";
    EXPECT_EQ(info.status, smf::ErrorCode::Success);

    // Check residual ||x_computed - x_exact|| < 1e-12
    double res = 0.0;
    for (int i = 0; i < n; ++i) {
        const double diff = b[static_cast<std::size_t>(i)] -
                            x_exact[static_cast<std::size_t>(i)];
        res += diff * diff;
    }
    res = std::sqrt(res);
    EXPECT_LT(res, 1e-12) << "residual too large: " << res;
}
