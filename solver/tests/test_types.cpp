#include <gtest/gtest.h>
#include "smf/types.hpp"
#include "smf/error.hpp"
#include "smf/control.hpp"
#include "smf/info.hpp"
#include "smf/csc_matrix.hpp"

// ---- Enum value checks ---------------------------------------------------

TEST(Types, EnumsCompile) {
    // SolveJob values (must match spec: Full=0, Forward=1, DiagOnly=2, Backward=3, DiagBack=4)
    EXPECT_EQ(static_cast<int>(smf::SolveJob::Full),     0);
    EXPECT_EQ(static_cast<int>(smf::SolveJob::Forward),  1);
    EXPECT_EQ(static_cast<int>(smf::SolveJob::DiagOnly), 2);
    EXPECT_EQ(static_cast<int>(smf::SolveJob::Backward), 3);
    EXPECT_EQ(static_cast<int>(smf::SolveJob::DiagBack), 4);

    // MatrixType values
    EXPECT_EQ(static_cast<int>(smf::MatrixType::RealSymmetricPositiveDefinite), 3);
    EXPECT_EQ(static_cast<int>(smf::MatrixType::RealSymmetricIndefinite),       4);

    // OrderingMethod values
    EXPECT_EQ(static_cast<int>(smf::OrderingMethod::AutoSerial),   0);
    EXPECT_EQ(static_cast<int>(smf::OrderingMethod::AutoParallel), 1);
    EXPECT_EQ(static_cast<int>(smf::OrderingMethod::AMD),          2);
    EXPECT_EQ(static_cast<int>(smf::OrderingMethod::METIS),        3);
    EXPECT_EQ(static_cast<int>(smf::OrderingMethod::UserSupplied), 4);

    // ScalingMethod values
    EXPECT_EQ(static_cast<int>(smf::ScalingMethod::None),         0);
    EXPECT_EQ(static_cast<int>(smf::ScalingMethod::Equilibration), 1);
    EXPECT_EQ(static_cast<int>(smf::ScalingMethod::Matching),     2);
    EXPECT_EQ(static_cast<int>(smf::ScalingMethod::UserSupplied), 3);

    // FactorStatus values
    EXPECT_EQ(static_cast<int>(smf::FactorStatus::Success),             0);
    EXPECT_EQ(static_cast<int>(smf::FactorStatus::NotPositiveDefinite), 1);
    EXPECT_EQ(static_cast<int>(smf::FactorStatus::Singular),            2);
    EXPECT_EQ(static_cast<int>(smf::FactorStatus::MaxPivotDelays),      3);

    // ErrorCode values
    EXPECT_EQ(static_cast<int>(smf::ErrorCode::Success),             0);
    EXPECT_EQ(static_cast<int>(smf::ErrorCode::NotInitialised),     -1);
    EXPECT_EQ(static_cast<int>(smf::ErrorCode::IllegalValue),       -3);
    EXPECT_EQ(static_cast<int>(smf::ErrorCode::InternalError),     -99);

    // Identity check
    EXPECT_EQ(smf::ErrorCode::Success, smf::ErrorCode::Success);

    // SolveJob enum round-trip
    EXPECT_EQ(smf::SolveJob::Full,     static_cast<smf::SolveJob>(0));
    EXPECT_EQ(smf::SolveJob::Forward,  static_cast<smf::SolveJob>(1));
    EXPECT_EQ(smf::SolveJob::DiagOnly, static_cast<smf::SolveJob>(2));
}

// ---- Control default values ---------------------------------------------

TEST(Types, ControlDefaults) {
    smf::Control c;
    EXPECT_EQ(c.matrix_type,   smf::MatrixType::RealSymmetricIndefinite);
    EXPECT_EQ(c.ordering,      smf::OrderingMethod::AutoSerial);
    EXPECT_EQ(c.scaling,       smf::ScalingMethod::None);
    EXPECT_EQ(c.nemin,         8);
    EXPECT_DOUBLE_EQ(c.pivot_u,      0.01);
    EXPECT_DOUBLE_EQ(c.small_pivot,  1e-20);
    EXPECT_FALSE(c.continue_on_singular);
    EXPECT_EQ(c.num_threads,   1);
    EXPECT_FALSE(c.deterministic);
    EXPECT_EQ(c.print_level,   0);
}

// ---- Info default values ------------------------------------------------

TEST(Types, InfoDefaults) {
    smf::Info info;
    EXPECT_EQ(info.status,        smf::ErrorCode::Success);
    EXPECT_EQ(info.factor_status, smf::FactorStatus::Success);
    EXPECT_EQ(info.numerical_rank,   0);
    EXPECT_EQ(info.num_negative,     0);
    EXPECT_EQ(info.num_zero,         0);
    EXPECT_EQ(info.num_positive,     0);
    EXPECT_EQ(info.delayed_pivots,   0);
    EXPECT_EQ(info.actual_factor_entries,   0);
    EXPECT_EQ(info.predicted_factor_entries, 0);
    EXPECT_DOUBLE_EQ(info.predicted_flops, 0.0);
    EXPECT_DOUBLE_EQ(info.actual_flops,    0.0);
    EXPECT_DOUBLE_EQ(info.analyse_seconds, 0.0);
    EXPECT_DOUBLE_EQ(info.factor_seconds,  0.0);
    EXPECT_DOUBLE_EQ(info.solve_seconds,   0.0);
    EXPECT_EQ(info.arena_peak_bytes, 0);
    EXPECT_EQ(info.arena_growths,    0);
}

// ---- CscLower validate_shape --------------------------------------------

TEST(Types, CscLowerEmpty) {
    // Default-constructed: n==0, all vectors empty — valid.
    smf::CscLower m;
    m.col_ptr = {0};  // n+1 = 1 entry
    EXPECT_EQ(m.nnz(), 0);
    EXPECT_EQ(m.validate_shape(), smf::ErrorCode::Success);
}

TEST(Types, CscLowerValidTriangle) {
    // 3x3 lower-triangular: entries (0,0), (1,0), (2,0), (1,1), (2,1), (2,2)
    smf::CscLower m;
    m.n       = 3;
    m.col_ptr = {0, 3, 5, 6};
    m.row_idx = {0, 1, 2,  1, 2,  2};
    m.values  = {1, 2, 3,  4, 5,  6};
    EXPECT_EQ(m.nnz(), 6);
    EXPECT_EQ(m.validate_shape(), smf::ErrorCode::Success);
}

TEST(Types, CscLowerBadColPtrSize) {
    smf::CscLower m;
    m.n       = 3;
    m.col_ptr = {0, 1, 3};  // only 3 entries, needs 4
    m.row_idx = {0, 1, 2};
    m.values  = {1, 2, 3};
    EXPECT_EQ(m.validate_shape(), smf::ErrorCode::IllegalValue);
}

TEST(Types, CscLowerColPtrNotZero) {
    smf::CscLower m;
    m.n       = 2;
    m.col_ptr = {1, 2, 3};  // col_ptr[0] != 0
    m.row_idx = {0, 1};
    m.values  = {1, 2};
    EXPECT_EQ(m.validate_shape(), smf::ErrorCode::IllegalValue);
}

TEST(Types, CscLowerNonMonotoneColPtr) {
    smf::CscLower m;
    m.n       = 2;
    m.col_ptr = {0, 2, 1};  // not monotone
    m.row_idx = {0, 1};
    m.values  = {1, 2};
    EXPECT_EQ(m.validate_shape(), smf::ErrorCode::IllegalValue);
}

TEST(Types, CscLowerUpperTriangleRejected) {
    smf::CscLower m;
    m.n       = 2;
    m.col_ptr = {0, 1, 2};
    m.row_idx = {0, 0};  // row_idx[1]=0 < col=1 → upper triangle
    m.values  = {1, 2};
    EXPECT_EQ(m.validate_shape(), smf::ErrorCode::IllegalValue);
}

TEST(Types, CscLowerRowIndexOutOfRange) {
    smf::CscLower m;
    m.n       = 2;
    m.col_ptr = {0, 1, 2};
    m.row_idx = {0, 5};  // row 5 >= n=2
    m.values  = {1, 2};
    EXPECT_EQ(m.validate_shape(), smf::ErrorCode::IllegalValue);
}

TEST(Types, CscLowerRowIdxValueSizeMismatch) {
    smf::CscLower m;
    m.n       = 2;
    m.col_ptr = {0, 1, 2};
    m.row_idx = {0, 1};
    m.values  = {1.0};  // wrong size
    EXPECT_EQ(m.validate_shape(), smf::ErrorCode::IllegalValue);
}
