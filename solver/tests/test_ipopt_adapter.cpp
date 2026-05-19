// test_ipopt_adapter.cpp — unit tests for SmfLinearSolver (IPOPT adapter).
//
// Synthetic 4×4 indefinite "KKT-flavored" system with inertia (2+, 2-, 0):
//
//   K = [ 4  2  0  0 ]    (block-diagonal: SPD block + negative-definite block)
//       [ 2  3  0  0 ]
//       [ 0  0 -2  1 ]
//       [ 0  0  1 -3 ]
//
// This has the same inertia signature as a KKT system (2 pos, 2 neg, 0 zero).
//
// Inverse (block-diagonal):
//   inv_top = inv([[4,2],[2,3]]) = (1/8)*[[3,-2],[-2,4]]
//   inv_bot = inv([[-2,1],[1,-3]]) = (1/5)*[[-3,-1],[-1,-2]]
//
//   K^{-1} = [ 3/8   -1/4    0     0   ]
//             [-1/4   1/2    0     0   ]
//             [ 0      0   -3/5  -1/5  ]
//             [ 0      0   -1/5  -2/5  ]

#include "smf/ipopt_adapter.hpp"

#include <cmath>
#include <gtest/gtest.h>

using namespace smf;

// ---------------------------------------------------------------------------
// Matrix description (lower triangle, 1-based COO)
//   Entries: (1,1)=4, (2,1)=2, (2,2)=3, (3,3)=-2, (4,3)=1, (4,4)=-3
// ---------------------------------------------------------------------------
static constexpr int KN   = 4;
static constexpr int KNNZ = 6;

static const int    K_irn [KNNZ] = {1, 2, 2, 3, 4, 4};
static const int    K_jcn [KNNZ] = {1, 1, 2, 3, 3, 4};
static const double K_vals[KNNZ] = {4.0, 2.0, 3.0, -2.0, 1.0, -3.0};

// ---------------------------------------------------------------------------
// Expected inverse columns (column-major, 4×4)
//   Col 0: [ 3/8, -1/4,   0,   0 ]
//   Col 1: [-1/4,  1/2,   0,   0 ]
//   Col 2: [  0,    0,  -3/5, -1/5]
//   Col 3: [  0,    0,  -1/5, -2/5]
// ---------------------------------------------------------------------------
static const double K_inv[KN * KN] = {
    // col 0        col 1        col 2        col 3
     3.0/8.0,    -1.0/4.0,     0.0,          0.0,      // row 0
    -1.0/4.0,     1.0/2.0,     0.0,          0.0,      // row 1
     0.0,          0.0,       -3.0/5.0,    -1.0/5.0,   // row 2
     0.0,          0.0,       -1.0/5.0,    -2.0/5.0    // row 3
};

static constexpr double TOL = 1e-10;

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------
class IpoptAdapterTest : public ::testing::Test {
protected:
    SmfLinearSolver solver_;

    void SetUp() override {
        Control ctrl;
        ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
        solver_ = SmfLinearSolver(ctrl);
    }
};

// ---------------------------------------------------------------------------
// Test 1: InitializeStructure succeeds.
// ---------------------------------------------------------------------------
TEST_F(IpoptAdapterTest, InitializeStructureSucceeds) {
    const auto status =
        solver_.InitializeStructure(KN, KNNZ, K_irn, K_jcn);
    EXPECT_EQ(status, Ipopt::SYMSOLVER_SUCCESS);
}

// ---------------------------------------------------------------------------
// Test 2: MultiSolve with identity RHS returns correct solution columns.
// ---------------------------------------------------------------------------
TEST_F(IpoptAdapterTest, MultiSolveIdentityRHS) {
    ASSERT_EQ(solver_.InitializeStructure(KN, KNNZ, K_irn, K_jcn),
              Ipopt::SYMSOLVER_SUCCESS);

    // Build identity RHS (column-major, 4×4).
    double rhs[KN * KN] = {};
    for (int i = 0; i < KN; ++i) rhs[i * KN + i] = 1.0;

    const auto status = solver_.MultiSolve(
        /*new_matrix=*/true, K_irn, K_jcn, K_vals,
        /*nrhs=*/KN, rhs,
        /*check_NegEVals=*/false, /*numberOfNegEVals=*/0);

    ASSERT_EQ(status, Ipopt::SYMSOLVER_SUCCESS);

    for (int j = 0; j < KN; ++j) {
        for (int i = 0; i < KN; ++i) {
            const double expected = K_inv[j * KN + i];
            const double got      = rhs[j * KN + i];
            EXPECT_NEAR(got, expected, TOL)
                << "  mismatch at row=" << i << " col=" << j;
        }
    }
}

// ---------------------------------------------------------------------------
// Test 3: NumberOfNegEVals() == 2.
// ---------------------------------------------------------------------------
TEST_F(IpoptAdapterTest, InertiaNegativeEvals) {
    ASSERT_EQ(solver_.InitializeStructure(KN, KNNZ, K_irn, K_jcn),
              Ipopt::SYMSOLVER_SUCCESS);

    double rhs[KN] = {1.0, 0.0, 0.0, 0.0};
    ASSERT_EQ(solver_.MultiSolve(
                  /*new_matrix=*/true, K_irn, K_jcn, K_vals,
                  /*nrhs=*/1, rhs,
                  /*check_NegEVals=*/false, 0),
              Ipopt::SYMSOLVER_SUCCESS);

    EXPECT_EQ(solver_.NumberOfNegEVals(), 2);
}

// ---------------------------------------------------------------------------
// Test 4: check_NegEVals=true with correct count → SYMSOLVER_SUCCESS.
// ---------------------------------------------------------------------------
TEST_F(IpoptAdapterTest, InertiaCheckCorrect) {
    ASSERT_EQ(solver_.InitializeStructure(KN, KNNZ, K_irn, K_jcn),
              Ipopt::SYMSOLVER_SUCCESS);

    double rhs[KN] = {1.0, 0.0, 0.0, 0.0};
    const auto status = solver_.MultiSolve(
        /*new_matrix=*/true, K_irn, K_jcn, K_vals,
        /*nrhs=*/1, rhs,
        /*check_NegEVals=*/true, /*numberOfNegEVals=*/2);

    EXPECT_EQ(status, Ipopt::SYMSOLVER_SUCCESS);
}

// ---------------------------------------------------------------------------
// Test 5: check_NegEVals=true with wrong count → SYMSOLVER_WRONG_INERTIA.
// ---------------------------------------------------------------------------
TEST_F(IpoptAdapterTest, InertiaCheckWrong) {
    ASSERT_EQ(solver_.InitializeStructure(KN, KNNZ, K_irn, K_jcn),
              Ipopt::SYMSOLVER_SUCCESS);

    // First call to factorize.
    double rhs1[KN] = {1.0, 0.0, 0.0, 0.0};
    ASSERT_EQ(solver_.MultiSolve(
                  /*new_matrix=*/true, K_irn, K_jcn, K_vals,
                  /*nrhs=*/1, rhs1,
                  /*check_NegEVals=*/false, 0),
              Ipopt::SYMSOLVER_SUCCESS);

    // Second call: same matrix (new_matrix=false), wrong expected inertia.
    double rhs2[KN] = {1.0, 0.0, 0.0, 0.0};
    const auto status = solver_.MultiSolve(
        /*new_matrix=*/false, K_irn, K_jcn, K_vals,
        /*nrhs=*/1, rhs2,
        /*check_NegEVals=*/true, /*numberOfNegEVals=*/1);  // wrong!

    EXPECT_EQ(status, Ipopt::SYMSOLVER_WRONG_INERTIA);
}

// ---------------------------------------------------------------------------
// Test 6: ProvidesInertia() returns true.
// ---------------------------------------------------------------------------
TEST_F(IpoptAdapterTest, ProvidesInertia) {
    EXPECT_TRUE(solver_.ProvidesInertia());
}
