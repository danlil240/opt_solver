#include "smf/frontal_matrix.hpp"
#include "smf/utils/memory.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

namespace {

using smf::AlignedArena;
using smf::FrontalMatrix;
using smf::Int;

// ---- Test 1: scatter_original into a 10×5 front ----------------------------

TEST(FrontalMatrix, ScatterOriginal10x5) {
    // 10×10 lower-CSC matrix with explicit nonzeros:
    //   col 0: rows {0,2,5}  vals {1,2,3}
    //   col 1: rows {1,3}    vals {4,5}
    //   col 2: rows {2,4,7}  vals {6,7,8}
    //   col 3: rows {3,6}    vals {9,10}
    //   col 4: rows {4,8,9}  vals {11,12,13}
    //   cols 5..9: diagonal only (value 1)
    const Int n = 10;
    std::vector<Int> col_ptr = {0, 3, 5, 8, 10, 13, 14, 15, 16, 17, 18};
    std::vector<Int> row_idx = {
        0, 2, 5,    // col 0
        1, 3,       // col 1
        2, 4, 7,    // col 2
        3, 6,       // col 3
        4, 8, 9,    // col 4
        5, 6, 7, 8, 9   // diagonal cols 5..9
    };
    std::vector<double> values = {
        1.0, 2.0, 3.0,
        4.0, 5.0,
        6.0, 7.0, 8.0,
        9.0, 10.0,
        11.0, 12.0, 13.0,
        1.0, 1.0, 1.0, 1.0, 1.0
    };
    ASSERT_EQ(col_ptr.size(), static_cast<std::size_t>(n + 1));
    ASSERT_EQ(row_idx.size(), static_cast<std::size_t>(col_ptr[n]));
    ASSERT_EQ(values.size(),  static_cast<std::size_t>(col_ptr[n]));

    // Front: all 10 rows, pivot cols 0..4
    const Int f = 10, p = 5;
    std::vector<Int> row_map = {0,1,2,3,4,5,6,7,8,9};
    std::vector<Int> col_map = {0,1,2,3,4};

    AlignedArena arena(4096);
    FrontalMatrix front(f, p, row_map, col_map, arena);
    front.scatter_original(col_ptr, row_idx, values);

    // col 0: orig rows {0,2,5} -> front rows {0,2,5}
    EXPECT_DOUBLE_EQ(front.at(0, 0),  1.0);
    EXPECT_DOUBLE_EQ(front.at(2, 0),  2.0);
    EXPECT_DOUBLE_EQ(front.at(5, 0),  3.0);

    // col 1: orig rows {1,3} -> front rows {1,3}
    EXPECT_DOUBLE_EQ(front.at(1, 1),  4.0);
    EXPECT_DOUBLE_EQ(front.at(3, 1),  5.0);

    // col 2: orig rows {2,4,7} -> front rows {2,4,7}
    EXPECT_DOUBLE_EQ(front.at(2, 2),  6.0);
    EXPECT_DOUBLE_EQ(front.at(4, 2),  7.0);
    EXPECT_DOUBLE_EQ(front.at(7, 2),  8.0);

    // col 3: orig rows {3,6} -> front rows {3,6}
    EXPECT_DOUBLE_EQ(front.at(3, 3),  9.0);
    EXPECT_DOUBLE_EQ(front.at(6, 3), 10.0);

    // col 4: orig rows {4,8,9} -> front rows {4,8,9}
    EXPECT_DOUBLE_EQ(front.at(4, 4), 11.0);
    EXPECT_DOUBLE_EQ(front.at(8, 4), 12.0);
    EXPECT_DOUBLE_EQ(front.at(9, 4), 13.0);

    // Entries not in the matrix should be zero
    EXPECT_DOUBLE_EQ(front.at(1, 0),  0.0);  // row 1 not in col 0
    EXPECT_DOUBLE_EQ(front.at(0, 1),  0.0);  // row 0 not in col 1
    EXPECT_DOUBLE_EQ(front.at(6, 2),  0.0);  // row 6 not in col 2
}

// ---- Test 2: scatter with permuted row/col maps ----------------------------

TEST(FrontalMatrix, ScatterPermuted) {
    // 6×6 lower-CSC; front covers original rows {1,2,3,4,5}, cols {2,4}.
    // col 2: rows {2,3,5} vals {10,11,12}
    // col 4: rows {4,5}   vals {20,21}
    // other cols: diagonal only
    const Int n = 6;
    std::vector<Int> col_ptr = {0, 1, 2, 5, 6, 8, 9};
    std::vector<Int> row_idx = {0,  1,  2,3,5,  3,  4,5,  5};
    std::vector<double> vals  = {1.0, 1.0, 10.0,11.0,12.0, 1.0, 20.0,21.0, 1.0};

    ASSERT_EQ(col_ptr.size(), static_cast<std::size_t>(n + 1));
    ASSERT_EQ(row_idx.size(), static_cast<std::size_t>(col_ptr[n]));
    ASSERT_EQ(vals.size(),    static_cast<std::size_t>(col_ptr[n]));

    const Int f = 5, p = 2;
    std::vector<Int> row_map = {1, 2, 3, 4, 5};
    std::vector<Int> col_map = {2, 4};

    AlignedArena arena(4096);
    FrontalMatrix front(f, p, row_map, col_map, arena);
    front.scatter_original(col_ptr, row_idx, vals);

    // j=0 (orig col 2): orig rows {2,3,5} -> front rows {1,2,4}
    EXPECT_DOUBLE_EQ(front.at(1, 0), 10.0);
    EXPECT_DOUBLE_EQ(front.at(2, 0), 11.0);
    EXPECT_DOUBLE_EQ(front.at(4, 0), 12.0);

    // j=1 (orig col 4): orig rows {4,5} -> front rows {3,4}
    EXPECT_DOUBLE_EQ(front.at(3, 1), 20.0);
    EXPECT_DOUBLE_EQ(front.at(4, 1), 21.0);

    // Orig row 1 is not in col 2 or col 4 → row 0 of front stays zero
    EXPECT_DOUBLE_EQ(front.at(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(front.at(0, 1), 0.0);
}

// ---- Test 3: assemble two child contributions into a parent ----------------

TEST(FrontalMatrix, AssembleContribTwoChildren) {
    // Parent: f=6, p=4. row_map={0,1,2,3,4,5}, col_map={0,1,2,3}.
    // Pivot invariant: col_map[k] = row_map[k] for k < p=4. ✓
    const Int f_p = 6, p_p = 4;
    std::vector<Int> prm = {0,1,2,3,4,5};
    std::vector<Int> pcm = {0,1,2,3};

    AlignedArena arena(8192);
    FrontalMatrix parent(f_p, p_p, prm, pcm, arena);

    // Child 1: q=2, parent_rows={0,1} (both < p=4 → pivot cols 0,1)
    // Contribution (lower-triangular, col-major):
    //   [ 10,  0 ]      col-major: col0={10,20}, col1={0,30}
    //   [ 20, 30 ]      => {10,20,0,30}
    const Int q1 = 2;
    double contrib1[] = {10.0, 20.0, 0.0, 30.0};
    std::vector<Int> pr1 = {0, 1};
    parent.assemble_contrib(contrib1, q1, pr1);

    EXPECT_DOUBLE_EQ(parent.at(0, 0), 10.0);
    EXPECT_DOUBLE_EQ(parent.at(1, 0), 20.0);
    EXPECT_DOUBLE_EQ(parent.at(0, 1),  0.0);  // upper triangle, not written
    EXPECT_DOUBLE_EQ(parent.at(1, 1), 30.0);

    // Child 2: q=2, parent_rows={2,3} (both < p=4 → pivot cols 2,3)
    // Contribution:
    //   [ 40,  0 ]      col-major: {40,50,0,60}
    //   [ 50, 60 ]
    const Int q2 = 2;
    double contrib2[] = {40.0, 50.0, 0.0, 60.0};
    std::vector<Int> pr2 = {2, 3};
    parent.assemble_contrib(contrib2, q2, pr2);

    EXPECT_DOUBLE_EQ(parent.at(2, 2), 40.0);
    EXPECT_DOUBLE_EQ(parent.at(3, 2), 50.0);
    EXPECT_DOUBLE_EQ(parent.at(3, 3), 60.0);

    // Child 1 values must be intact
    EXPECT_DOUBLE_EQ(parent.at(0, 0), 10.0);
    EXPECT_DOUBLE_EQ(parent.at(1, 0), 20.0);
    EXPECT_DOUBLE_EQ(parent.at(1, 1), 30.0);

    // Extended rows (4,5) and unset pivot entries should be zero
    for (Int j = 0; j < p_p; ++j) {
        EXPECT_DOUBLE_EQ(parent.at(4, j), 0.0) << "row=4 col=" << j;
        EXPECT_DOUBLE_EQ(parent.at(5, j), 0.0) << "row=5 col=" << j;
    }
}

// ---- Test 4: contributions to extended columns are skipped -----------------

TEST(FrontalMatrix, AssembleContribSkipsExtendedCols) {
    // Parent: f=5, p=2. Extended rows are 2,3,4.
    // Contribution q=3, parent_rows={1,3,4}.
    // Only parent_rows[0]=1 < p=2 contributes (column 1).
    // parent_rows[1]=3 >= 2 and parent_rows[2]=4 >= 2 are skipped.
    const Int f_p = 5, p_p = 2;
    std::vector<Int> prm = {0,1,2,3,4};
    std::vector<Int> pcm = {0,1};

    AlignedArena arena(4096);
    FrontalMatrix parent(f_p, p_p, prm, pcm, arena);

    const Int q = 3;
    // Contribution col-major, lower-triangle:
    //   col 0: rows {0,1,2} = {7,8,9}
    //   col 1: rows {1,2}   = {5,6} (row 0 upper-tri = 0)
    //   col 2: rows {2}     = {3}   (rows 0,1 upper-tri = 0)
    double contrib[] = {7.0, 8.0, 9.0,   0.0, 5.0, 6.0,   0.0, 0.0, 3.0};
    std::vector<Int> parent_rows_vec = {1, 3, 4};
    parent.assemble_contrib(contrib, q, parent_rows_vec);

    // j=0 (pr_j=1 < p=2, pcol=1):
    //   i=0: at(parent_rows[0]=1, 1) += 7.0
    //   i=1: at(parent_rows[1]=3, 1) += 8.0
    //   i=2: at(parent_rows[2]=4, 1) += 9.0
    EXPECT_DOUBLE_EQ(parent.at(1, 1), 7.0);
    EXPECT_DOUBLE_EQ(parent.at(3, 1), 8.0);
    EXPECT_DOUBLE_EQ(parent.at(4, 1), 9.0);

    // j=1 and j=2 are skipped → col 0 stays zero everywhere
    EXPECT_DOUBLE_EQ(parent.at(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(parent.at(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(parent.at(2, 0), 0.0);
    EXPECT_DOUBLE_EQ(parent.at(3, 0), 0.0);
    EXPECT_DOUBLE_EQ(parent.at(4, 0), 0.0);
}

// ---- Test 5: scatter then assemble (combined) ------------------------------

TEST(FrontalMatrix, ScatterThenAssemble) {
    // Front: f=4, p=2. Diagonal-only matrix for cols 0,1.
    // (n=4 used only to size CSC — omitted to avoid warning)
    std::vector<Int> col_ptr = {0,1,2,3,4};
    std::vector<Int> row_idx = {0, 1, 2, 3};
    std::vector<double> vals  = {1.0, 2.0, 3.0, 4.0};

    const Int f = 4, p = 2;
    std::vector<Int> row_map = {0,1,2,3};
    std::vector<Int> col_map = {0,1};

    AlignedArena arena(4096);
    FrontalMatrix front(f, p, row_map, col_map, arena);
    front.scatter_original(col_ptr, row_idx, vals);

    EXPECT_DOUBLE_EQ(front.at(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(front.at(1, 1), 2.0);
    EXPECT_DOUBLE_EQ(front.at(1, 0), 0.0);

    // Assemble contribution [ 5,  0 ]  -> col-major: {5,6,0,7}
    //                       [ 6,  7 ]
    double contrib[] = {5.0, 6.0, 0.0, 7.0};
    std::vector<Int> pr = {0, 1};
    front.assemble_contrib(contrib, 2, pr);

    EXPECT_DOUBLE_EQ(front.at(0, 0), 6.0);  // 1+5
    EXPECT_DOUBLE_EQ(front.at(1, 0), 6.0);  // 0+6
    EXPECT_DOUBLE_EQ(front.at(1, 1), 9.0);  // 2+7
}

// ---- Test 6: zero() clears all entries ------------------------------------

TEST(FrontalMatrix, ZeroClears) {
    const Int f = 4, p = 3;
    std::vector<Int> row_map = {0,1,2,3};
    std::vector<Int> col_map = {0,1,2};

    AlignedArena arena(4096);
    FrontalMatrix front(f, p, row_map, col_map, arena);

    front.at(0, 0) = 5.0;
    front.at(2, 1) = 6.0;
    front.at(3, 2) = 7.0;
    front.zero();

    for (Int j = 0; j < p; ++j)
        for (Int i = 0; i < f; ++i)
            EXPECT_DOUBLE_EQ(front.at(i, j), 0.0) << "at(" << i << "," << j << ")";
}

// ---- Test 7: data pointer is 64-byte aligned --------------------------------

TEST(FrontalMatrix, DataAlignment) {
    const Int f = 7, p = 3;
    std::vector<Int> row_map = {0,1,2,3,4,5,6};
    std::vector<Int> col_map = {0,1,2};

    AlignedArena arena(4096);
    FrontalMatrix front(f, p, row_map, col_map, arena);

    const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(front.data());
    EXPECT_EQ(addr % 64u, 0u) << "data() pointer not 64-byte aligned";
}

// ---- Test 8: accumulate multiple scatter calls (+=) -------------------------

TEST(FrontalMatrix, ScatterAccumulates) {
    // Scatter the same matrix twice → values should double.
    const Int f = 4, p = 2;
    std::vector<Int> col_ptr = {0,2,4,5,6};
    std::vector<Int> row_idx = {0,2, 1,3, 2, 3};
    std::vector<double> vals  = {1.0,2.0, 3.0,4.0, 5.0, 6.0};
    std::vector<Int> row_map = {0,1,2,3};
    std::vector<Int> col_map = {0,1};

    AlignedArena arena(4096);
    FrontalMatrix front(f, p, row_map, col_map, arena);

    front.scatter_original(col_ptr, row_idx, vals);
    front.scatter_original(col_ptr, row_idx, vals);

    EXPECT_DOUBLE_EQ(front.at(0, 0), 2.0);   // 1+1
    EXPECT_DOUBLE_EQ(front.at(2, 0), 4.0);   // 2+2
    EXPECT_DOUBLE_EQ(front.at(1, 1), 6.0);   // 3+3
    EXPECT_DOUBLE_EQ(front.at(3, 1), 8.0);   // 4+4
}

} // anonymous namespace
