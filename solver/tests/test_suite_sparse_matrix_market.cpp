// test_suite_sparse_matrix_market.cpp — Matrix Market reader tests
//
// Tests the smf::read_matrix_market() function.
// All tests are self-contained: synthetic .mtx files are written to /tmp and
// read back.  No network access required.

#include "smf/matrix_market.hpp"
#include "smf/control.hpp"
#include "smf/info.hpp"
#include "smf/solver.hpp"
#include "smf/types.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Write text to a file; returns true on success.
static bool write_file(const char *path, const char *content) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) return false;
    ofs << content;
    return true;
}

/// Symmetric (lower-triangle only) SpMV: y = A*x, A stored lower-CSC.
static void spmv_sym_lower(const smf::CscLower &A,
                            const double *x, double *y) {
    int n = A.n;
    for (int i = 0; i < n; ++i) y[i] = 0.0;
    for (int j = 0; j < n; ++j) {
        for (int p = A.col_ptr[static_cast<std::size_t>(j)];
             p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p)
        {
            int    i = A.row_idx[static_cast<std::size_t>(p)];
            double v = A.values [static_cast<std::size_t>(p)];
            y[i] += v * x[j];
            if (i != j) y[j] += v * x[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Test: read a small 3×3 symmetric matrix
// ---------------------------------------------------------------------------
TEST(MatrixMarket, SmallSymmetric) {
    const char *path = "/tmp/smf_test_mm_small_sym.mtx";
    // 3×3 symmetric:
    //   [ 4  1  2 ]
    //   [ 1  5  3 ]
    //   [ 2  3  6 ]
    // Lower triangle (1-based):
    //   (1,1,4) (2,1,1) (2,2,5) (3,1,2) (3,2,3) (3,3,6)
    ASSERT_TRUE(write_file(path,
        "%%MatrixMarket matrix coordinate real symmetric\n"
        "% A small 3x3 test matrix\n"
        "3 3 6\n"
        "1 1 4.0\n"
        "2 1 1.0\n"
        "2 2 5.0\n"
        "3 1 2.0\n"
        "3 2 3.0\n"
        "3 3 6.0\n"
    ));

    smf::MatrixMarketResult r = smf::read_matrix_market(path);
    ASSERT_TRUE(r.ok) << "Error: " << r.error_message;
    EXPECT_TRUE(r.is_symmetric);
    EXPECT_EQ(r.rows, 3);
    EXPECT_EQ(r.cols, 3);
    EXPECT_EQ(r.matrix.n, 3);

    // 6 lower-triangle entries
    EXPECT_EQ(static_cast<int>(r.matrix.nnz()), 6);

    // Check column pointer structure.
    // Lower-triangle storage: col j contains rows >= j.
    // col 0: rows 0, 1, 2  → 3 entries
    // col 1: rows 1, 2     → 2 entries
    // col 2: row 2         → 1 entry
    // col_ptr = {0, 3, 5, 6}
    EXPECT_EQ(r.matrix.col_ptr[0], 0);
    EXPECT_EQ(r.matrix.col_ptr[1], 3);
    EXPECT_EQ(r.matrix.col_ptr[2], 5);
    EXPECT_EQ(r.matrix.col_ptr[3], 6);

    // Row indices for col 0 (sorted): 0, 1, 2
    EXPECT_EQ(r.matrix.row_idx[0], 0); EXPECT_DOUBLE_EQ(r.matrix.values[0], 4.0);
    EXPECT_EQ(r.matrix.row_idx[1], 1); EXPECT_DOUBLE_EQ(r.matrix.values[1], 1.0);
    EXPECT_EQ(r.matrix.row_idx[2], 2); EXPECT_DOUBLE_EQ(r.matrix.values[2], 2.0);
    // Col 1: rows 1, 2
    EXPECT_EQ(r.matrix.row_idx[3], 1); EXPECT_DOUBLE_EQ(r.matrix.values[3], 5.0);
    EXPECT_EQ(r.matrix.row_idx[4], 2); EXPECT_DOUBLE_EQ(r.matrix.values[4], 3.0);
    // Col 2: row 2
    EXPECT_EQ(r.matrix.row_idx[5], 2); EXPECT_DOUBLE_EQ(r.matrix.values[5], 6.0);
}

// ---------------------------------------------------------------------------
// Test: symmetric file where some entries are given as upper-triangle (mirrored)
// ---------------------------------------------------------------------------
TEST(MatrixMarket, UpperTriangleMirrored) {
    const char *path = "/tmp/smf_test_mm_upper.mtx";
    // Same 3×3 matrix but entries given as upper triangle
    ASSERT_TRUE(write_file(path,
        "%%MatrixMarket matrix coordinate real symmetric\n"
        "3 3 6\n"
        "1 1 4.0\n"
        "1 2 1.0\n"   // upper → mirrored to (2,1)
        "2 2 5.0\n"
        "1 3 2.0\n"   // upper → mirrored to (3,1)
        "2 3 3.0\n"   // upper → mirrored to (3,2)
        "3 3 6.0\n"
    ));

    smf::MatrixMarketResult r = smf::read_matrix_market(path);
    ASSERT_TRUE(r.ok) << r.error_message;
    EXPECT_EQ(static_cast<int>(r.matrix.nnz()), 6);
    // Same content as SmallSymmetric test
    EXPECT_EQ(r.matrix.col_ptr[1], 3);
    EXPECT_EQ(r.matrix.col_ptr[2], 5);
    EXPECT_EQ(r.matrix.col_ptr[3], 6);
}

// ---------------------------------------------------------------------------
// Test: general matrix (lower triangle only)
// ---------------------------------------------------------------------------
TEST(MatrixMarket, GeneralLowerOnly) {
    const char *path = "/tmp/smf_test_mm_general.mtx";
    // 3×3 general — only lower entries stored, upper are dropped
    ASSERT_TRUE(write_file(path,
        "%%MatrixMarket matrix coordinate real general\n"
        "3 3 5\n"
        "1 1 10.0\n"
        "2 1 -1.0\n"
        "2 2 10.0\n"
        "1 2 -999.0\n"  // upper triangle → discarded
        "3 3 10.0\n"
    ));

    smf::MatrixMarketResult r = smf::read_matrix_market(path);
    ASSERT_TRUE(r.ok) << r.error_message;
    EXPECT_TRUE(r.is_general);
    EXPECT_FALSE(r.is_symmetric);
    // Should have 4 entries (upper entry discarded)
    EXPECT_EQ(static_cast<int>(r.matrix.nnz()), 4);
}

// ---------------------------------------------------------------------------
// Test: 5×5 SPD matrix — write, read, solve with smf, check residual
// ---------------------------------------------------------------------------
TEST(MatrixMarket, Solve5x5SPD) {
    const char *path = "/tmp/smf_test_mm_5x5_spd.mtx";
    // Diagonally dominant 5×5 SPD matrix (lower triangle):
    //   diag = 10
    //   sub-diag entries = -1 (adjacent)
    // Full matrix:
    //   [ 10 -1  0  0  0 ]
    //   [ -1 10 -1  0  0 ]
    //   [  0 -1 10 -1  0 ]
    //   [  0  0 -1 10 -1 ]
    //   [  0  0  0 -1 10 ]
    ASSERT_TRUE(write_file(path,
        "%%MatrixMarket matrix coordinate real symmetric\n"
        "% 5x5 tridiagonal SPD\n"
        "5 5 9\n"
        "1 1 10.0\n"
        "2 1 -1.0\n"
        "2 2 10.0\n"
        "3 2 -1.0\n"
        "3 3 10.0\n"
        "4 3 -1.0\n"
        "4 4 10.0\n"
        "5 4 -1.0\n"
        "5 5 10.0\n"
    ));

    smf::MatrixMarketResult r = smf::read_matrix_market(path);
    ASSERT_TRUE(r.ok) << r.error_message;
    ASSERT_EQ(r.matrix.n, 5);
    ASSERT_EQ(static_cast<int>(r.matrix.nnz()), 9);

    const int N = 5;
    // RHS b = [1,1,1,1,1]
    std::vector<double> b(N, 1.0);
    std::vector<double> b_orig(b);

    smf::Control ctrl;
    ctrl.matrix_type = smf::MatrixType::RealSymmetricPositiveDefinite;
    smf::Info info;
    smf::Solver solver;

    auto ak = solver.analyse(r.matrix, ctrl, info);
    ASSERT_TRUE(ak) << "analyse failed";

    smf::FactorKeep fkeep;
    smf::FactorStatus fs = solver.factor(*ak, ctrl, info, fkeep);
    ASSERT_EQ(fs, smf::FactorStatus::Success) << "factor failed";

    int rc = solver.solve(fkeep, ctrl, info, b.data(), N, 1);
    ASSERT_EQ(rc, 0) << "solve failed";

    // Compute residual ||A*x - b|| / ||b||
    std::vector<double> ax(N, 0.0);
    spmv_sym_lower(r.matrix, b.data(), ax.data());
    double num = 0.0, den = 0.0;
    for (int i = 0; i < N; ++i) {
        double res = ax[i] - b_orig[i];
        num += res * res;
        den += b_orig[i] * b_orig[i];
    }
    double residual = std::sqrt(num / den);
    EXPECT_LT(residual, 1e-10) << "residual too large: " << residual;
}

// ---------------------------------------------------------------------------
// Test: duplicate entries are summed
// ---------------------------------------------------------------------------
TEST(MatrixMarket, DuplicatesSummed) {
    const char *path = "/tmp/smf_test_mm_dup.mtx";
    ASSERT_TRUE(write_file(path,
        "%%MatrixMarket matrix coordinate real symmetric\n"
        "2 2 4\n"
        "1 1 3.0\n"
        "1 1 2.0\n"   // duplicate → sum = 5.0
        "2 1 1.0\n"
        "2 2 4.0\n"
    ));

    smf::MatrixMarketResult r = smf::read_matrix_market(path);
    ASSERT_TRUE(r.ok) << r.error_message;
    EXPECT_EQ(r.matrix.n, 2);
    // After deduplication: col 0 → rows {0:5.0, 1:1.0}, col 1 → rows {1:4.0}
    EXPECT_EQ(static_cast<int>(r.matrix.nnz()), 3);
    // col 0: values[0]=5.0 (row 0), values[1]=1.0 (row 1)
    EXPECT_DOUBLE_EQ(r.matrix.values[0], 5.0);
    EXPECT_DOUBLE_EQ(r.matrix.values[1], 1.0);
    EXPECT_DOUBLE_EQ(r.matrix.values[2], 4.0);
}

// ---------------------------------------------------------------------------
// Test: out-of-range indices are silently discarded
// ---------------------------------------------------------------------------
TEST(MatrixMarket, OutOfRangeDiscarded) {
    const char *path = "/tmp/smf_test_mm_oor.mtx";
    ASSERT_TRUE(write_file(path,
        "%%MatrixMarket matrix coordinate real symmetric\n"
        "2 2 3\n"
        "1 1 5.0\n"
        "3 1 9.9\n"   // row 3 > rows(2) → discarded (0-based: 2 >= 2)
        "2 2 7.0\n"
    ));

    smf::MatrixMarketResult r = smf::read_matrix_market(path);
    ASSERT_TRUE(r.ok) << r.error_message;
    EXPECT_EQ(static_cast<int>(r.matrix.nnz()), 2); // only (0,0)=5 and (1,1)=7
}

// ---------------------------------------------------------------------------
// Test error: missing file
// ---------------------------------------------------------------------------
TEST(MatrixMarket, ErrorMissingFile) {
    smf::MatrixMarketResult r =
        smf::read_matrix_market("/tmp/smf_test_mm_NONEXISTENT_FILE_12345.mtx");
    EXPECT_FALSE(r.ok);
    EXPECT_FALSE(r.error_message.empty());
}

// ---------------------------------------------------------------------------
// Test error: malformed header (no %%MatrixMarket banner)
// ---------------------------------------------------------------------------
TEST(MatrixMarket, ErrorMalformedHeader) {
    const char *path = "/tmp/smf_test_mm_bad_header.mtx";
    ASSERT_TRUE(write_file(path,
        "% this is just a comment\n"
        "NOT A MATRIX MARKET FILE\n"
        "3 3 3\n"
        "1 1 1.0\n"
    ));

    smf::MatrixMarketResult r = smf::read_matrix_market(path);
    EXPECT_FALSE(r.ok);
    EXPECT_FALSE(r.error_message.empty());
}

// ---------------------------------------------------------------------------
// Test error: array format (not coordinate) is rejected
// ---------------------------------------------------------------------------
TEST(MatrixMarket, ErrorArrayFormat) {
    const char *path = "/tmp/smf_test_mm_array.mtx";
    ASSERT_TRUE(write_file(path,
        "%%MatrixMarket matrix array real symmetric\n"
        "3 3\n"
        "1.0\n2.0\n3.0\n4.0\n5.0\n6.0\n"
    ));

    smf::MatrixMarketResult r = smf::read_matrix_market(path);
    EXPECT_FALSE(r.ok);
    EXPECT_FALSE(r.error_message.empty());
}

// ---------------------------------------------------------------------------
// Test error: non-square matrix
// ---------------------------------------------------------------------------
TEST(MatrixMarket, ErrorNonSquare) {
    const char *path = "/tmp/smf_test_mm_nonsquare.mtx";
    ASSERT_TRUE(write_file(path,
        "%%MatrixMarket matrix coordinate real general\n"
        "3 4 2\n"
        "1 1 1.0\n"
        "2 2 2.0\n"
    ));

    smf::MatrixMarketResult r = smf::read_matrix_market(path);
    EXPECT_FALSE(r.ok);
    EXPECT_FALSE(r.error_message.empty());
}

// ---------------------------------------------------------------------------
// Test error: empty path
// ---------------------------------------------------------------------------
TEST(MatrixMarket, ErrorEmptyPath) {
    smf::MatrixMarketResult r = smf::read_matrix_market("");
    EXPECT_FALSE(r.ok);
    EXPECT_FALSE(r.error_message.empty());
}

// ---------------------------------------------------------------------------
// Test: pattern matrix (values treated as 1.0)
// ---------------------------------------------------------------------------
TEST(MatrixMarket, PatternMatrix) {
    const char *path = "/tmp/smf_test_mm_pattern.mtx";
    ASSERT_TRUE(write_file(path,
        "%%MatrixMarket matrix coordinate pattern symmetric\n"
        "3 3 4\n"
        "1 1\n"
        "2 1\n"
        "2 2\n"
        "3 3\n"
    ));

    smf::MatrixMarketResult r = smf::read_matrix_market(path);
    ASSERT_TRUE(r.ok) << r.error_message;
    EXPECT_EQ(static_cast<int>(r.matrix.nnz()), 4);
    // All values should be 1.0
    for (double v : r.matrix.values) {
        EXPECT_DOUBLE_EQ(v, 1.0);
    }
}

// ---------------------------------------------------------------------------
// Test: comments interspersed throughout the file
// ---------------------------------------------------------------------------
TEST(MatrixMarket, CommentsHandled) {
    const char *path = "/tmp/smf_test_mm_comments.mtx";
    ASSERT_TRUE(write_file(path,
        "%%MatrixMarket matrix coordinate real symmetric\n"
        "% line 1 comment\n"
        "% line 2 comment\n"
        "2 2 2\n"
        "% data comment\n"
        "1 1 9.0\n"
        "% another comment between data\n"
        "2 2 9.0\n"
    ));

    smf::MatrixMarketResult r = smf::read_matrix_market(path);
    ASSERT_TRUE(r.ok) << r.error_message;
    EXPECT_EQ(r.matrix.n, 2);
    EXPECT_EQ(static_cast<int>(r.matrix.nnz()), 2);
}

// ---------------------------------------------------------------------------
// Test: integer field type
// ---------------------------------------------------------------------------
TEST(MatrixMarket, IntegerFieldType) {
    const char *path = "/tmp/smf_test_mm_integer.mtx";
    ASSERT_TRUE(write_file(path,
        "%%MatrixMarket matrix coordinate integer symmetric\n"
        "3 3 3\n"
        "1 1 5\n"
        "2 2 3\n"
        "3 3 7\n"
    ));

    smf::MatrixMarketResult r = smf::read_matrix_market(path);
    ASSERT_TRUE(r.ok) << r.error_message;
    EXPECT_EQ(static_cast<int>(r.matrix.nnz()), 3);
    EXPECT_DOUBLE_EQ(r.matrix.values[0], 5.0);
}
