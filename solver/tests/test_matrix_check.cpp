#include <gtest/gtest.h>
#include "smf/check_matrix.hpp"

using namespace smf;

// ---------------------------------------------------------------------------
// Test 1: ValidLowerCSC
// 4x4 fully valid lower-CSC; clean == input, all counts zero.
// ---------------------------------------------------------------------------
TEST(MatrixCheck, ValidLowerCSC) {
    // Column 0: (0,0)=1.0, (2,0)=3.0
    // Column 1: (1,1)=2.0, (3,1)=5.0
    // Column 2: (2,2)=4.0, (3,2)=6.0
    // Column 3: (3,3)=7.0
    CscLower A;
    A.n       = 4;
    A.col_ptr = {0, 2, 4, 6, 7};
    A.row_idx = {0, 2, 1, 3, 2, 3, 3};
    A.values  = {1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 7.0};

    Info info;
    auto cp = clean_lower_csc(A, info);

    EXPECT_EQ(info.status,     ErrorCode::Success);
    EXPECT_EQ(cp.duplicates,   0);
    EXPECT_EQ(cp.out_of_range, 0);
    EXPECT_EQ(cp.missing_diag, 0);

    ASSERT_EQ(cp.clean.n, 4);
    EXPECT_EQ(cp.clean.col_ptr, A.col_ptr);
    EXPECT_EQ(cp.clean.row_idx, A.row_idx);
    EXPECT_EQ(cp.clean.values,  A.values);
}

// ---------------------------------------------------------------------------
// Test 2: UpperTriangleRejection
// Entries with row < col are discarded and counted in out_of_range.
// ---------------------------------------------------------------------------
TEST(MatrixCheck, UpperTriangleRejection) {
    // Column 0: (0,0)=1.0                  [valid]
    // Column 1: (0,1)=99.0 [row<col INVALID], (1,1)=2.0 [valid]
    // Column 2: (1,2)=88.0 [row<col INVALID], (2,2)=3.0 [valid]
    CscLower A;
    A.n       = 3;
    A.col_ptr = {0, 1, 3, 5};
    A.row_idx = {0, 0, 1, 1, 2};
    A.values  = {1.0, 99.0, 2.0, 88.0, 3.0};

    Info info;
    auto cp = clean_lower_csc(A, info);

    EXPECT_EQ(cp.out_of_range, 2);
    EXPECT_EQ(cp.duplicates,   0);
    EXPECT_EQ(cp.missing_diag, 0);
    EXPECT_EQ(cp.clean.nnz(),  3);

    // Only the diagonal entries survive
    EXPECT_EQ(cp.clean.row_idx, (std::vector<Int>{0, 1, 2}));
}

// ---------------------------------------------------------------------------
// Test 3: DuplicateSummation
// Two entries at the same (row,col) are summed; duplicates > 0.
// ---------------------------------------------------------------------------
TEST(MatrixCheck, DuplicateSummation) {
    // Column 0: (0,0)=1.0 and (0,0)=2.0  -> summed to 3.0, duplicates=1
    // Column 1: (1,1)=4.0
    // Column 2: (2,2)=5.0
    CscLower A;
    A.n       = 3;
    A.col_ptr = {0, 2, 3, 4};
    A.row_idx = {0, 0, 1, 2};
    A.values  = {1.0, 2.0, 4.0, 5.0};

    Info info;
    auto cp = clean_lower_csc(A, info);

    EXPECT_EQ(cp.duplicates,   1);
    EXPECT_GT(cp.duplicates,   0);
    EXPECT_EQ(cp.out_of_range, 0);
    EXPECT_EQ(cp.missing_diag, 0);
    EXPECT_EQ(cp.clean.nnz(),  3);
    EXPECT_DOUBLE_EQ(cp.clean.values[0], 3.0);
}

// ---------------------------------------------------------------------------
// Test 4: MissingDiagonal
// Columns without a diagonal entry are counted in missing_diag.
// ---------------------------------------------------------------------------
TEST(MatrixCheck, MissingDiagonal) {
    // 3x3:
    // Column 0: (1,0)=2.0  -- no (0,0) diagonal
    // Column 1: (1,1)=3.0  -- has diagonal
    // Column 2: empty      -- no (2,2) diagonal
    CscLower A;
    A.n       = 3;
    A.col_ptr = {0, 1, 2, 2};
    A.row_idx = {1, 1};
    A.values  = {2.0, 3.0};

    Info info;
    auto cp = clean_lower_csc(A, info);

    EXPECT_EQ(cp.missing_diag, 2);
    EXPECT_EQ(cp.out_of_range, 0);
    EXPECT_EQ(cp.duplicates,   0);

    // col 0 has one entry; its row is not 0
    EXPECT_EQ(cp.clean.col_ptr[1] - cp.clean.col_ptr[0], 1);
    EXPECT_NE(cp.clean.row_idx[0], 0);
}

// ---------------------------------------------------------------------------
// Test 5: BadColPtrHandled
// col_ptr with wrong size triggers an error path in clean_lower_csc.
// ---------------------------------------------------------------------------
TEST(MatrixCheck, BadColPtrHandled) {
    CscLower A;
    A.n       = 3;
    A.col_ptr = {0, 1, 2};  // size 3, should be n+1 = 4
    A.row_idx = {0, 1};
    A.values  = {1.0, 2.0};

    Info info;
    auto cp = clean_lower_csc(A, info);

    EXPECT_EQ(info.status, ErrorCode::IllegalValue);
}

// ---------------------------------------------------------------------------
// Test 6: EmptyMatrix
// n=0 CscLower; no crash, nnz=0, all counts zero.
// ---------------------------------------------------------------------------
TEST(MatrixCheck, EmptyMatrix) {
    CscLower A;
    A.n       = 0;
    A.col_ptr = {0};
    A.row_idx = {};
    A.values  = {};

    Info info;
    auto cp = clean_lower_csc(A, info);

    EXPECT_EQ(info.status,     ErrorCode::Success);
    EXPECT_EQ(cp.clean.n,      0);
    EXPECT_EQ(cp.clean.nnz(),  0);
    EXPECT_EQ(cp.duplicates,   0);
    EXPECT_EQ(cp.out_of_range, 0);
    EXPECT_EQ(cp.missing_diag, 0);
}

// ---------------------------------------------------------------------------
// Test 7: OneByOne
// 1x1 matrix with a single diagonal entry.
// ---------------------------------------------------------------------------
TEST(MatrixCheck, OneByOne) {
    CscLower A;
    A.n       = 1;
    A.col_ptr = {0, 1};
    A.row_idx = {0};
    A.values  = {5.0};

    Info info;
    auto cp = clean_lower_csc(A, info);

    EXPECT_EQ(info.status,     ErrorCode::Success);
    EXPECT_EQ(cp.clean.nnz(),  1);
    EXPECT_EQ(cp.missing_diag, 0);
    EXPECT_EQ(cp.duplicates,   0);
    EXPECT_EQ(cp.out_of_range, 0);
    EXPECT_DOUBLE_EQ(cp.clean.values[0], 5.0);
}

// ---------------------------------------------------------------------------
// Test 8: TwoByTwo
// 2x2 lower CSC with a sub-diagonal entry.
// ---------------------------------------------------------------------------
TEST(MatrixCheck, TwoByTwo) {
    // Column 0: (0,0)=1.0, (1,0)=2.0
    // Column 1: (1,1)=3.0
    CscLower A;
    A.n       = 2;
    A.col_ptr = {0, 2, 3};
    A.row_idx = {0, 1, 1};
    A.values  = {1.0, 2.0, 3.0};

    Info info;
    auto cp = clean_lower_csc(A, info);

    EXPECT_EQ(info.status,     ErrorCode::Success);
    EXPECT_EQ(cp.clean.nnz(),  3);
    EXPECT_EQ(cp.missing_diag, 0);
    EXPECT_EQ(cp.duplicates,   0);
    EXPECT_EQ(cp.out_of_range, 0);
    EXPECT_EQ(cp.clean.col_ptr, A.col_ptr);
    EXPECT_EQ(cp.clean.row_idx, A.row_idx);
}

// ---------------------------------------------------------------------------
// Test 9: NegativeIndices
// Manually crafted CscLower with row_idx[0]=-1 (bypasses validate_shape).
// The negative entry must be counted as out_of_range (since -1 < col=0).
// ---------------------------------------------------------------------------
TEST(MatrixCheck, NegativeIndices) {
    // Column 0: row=-1 -> i=-1 < j=0 -> out_of_range
    // Column 1: row= 1 -> valid diagonal
    // Column 2: row= 2 -> valid diagonal
    CscLower A;
    A.n       = 3;
    A.col_ptr = {0, 1, 2, 3};
    A.row_idx = {-1, 1, 2};
    A.values  = {1.0, 2.0, 3.0};

    Info info;
    auto cp = clean_lower_csc(A, info);

    EXPECT_EQ(cp.out_of_range, 1);
    EXPECT_EQ(cp.clean.nnz(),  2);
    EXPECT_EQ(cp.missing_diag, 1);
    EXPECT_EQ(cp.original_to_clean[0], -1);
}

// ---------------------------------------------------------------------------
// Test 10: InfoCounts
// Verify info.matrix_* fields mirror CleanedPattern counts.
// ---------------------------------------------------------------------------
TEST(MatrixCheck, InfoCounts) {
    // 4x4 matrix with one of each issue:
    //   duplicates=1, out_of_range=1, missing_diag=1
    //
    // Column 0: (0,0)=1.0, (0,0)=2.0  -> 1 duplicate, sum=3.0
    // Column 1: (0,1)=9.0 [upper tri]  -> 1 out_of_range; (1,1)=4.0
    // Column 2: (3,2)=7.0              -> valid entry but no diagonal (row 2)
    // Column 3: (3,3)=5.0              -> valid diagonal
    CscLower A;
    A.n       = 4;
    A.col_ptr = {0, 2, 4, 5, 6};
    A.row_idx = {0, 0, 0, 1, 3, 3};
    A.values  = {1.0, 2.0, 9.0, 4.0, 7.0, 5.0};

    Info info;
    auto cp = clean_lower_csc(A, info);

    EXPECT_EQ(info.matrix_duplicates,   1);
    EXPECT_EQ(info.matrix_out_of_range, 1);
    EXPECT_EQ(info.matrix_missing_diag, 1);

    EXPECT_EQ(cp.duplicates,   info.matrix_duplicates);
    EXPECT_EQ(cp.out_of_range, info.matrix_out_of_range);
    EXPECT_EQ(cp.missing_diag, info.matrix_missing_diag);

    // The merged duplicate in col 0 sums to 3.0
    EXPECT_DOUBLE_EQ(cp.clean.values[0], 3.0);
}
