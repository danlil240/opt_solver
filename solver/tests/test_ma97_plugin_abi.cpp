#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

struct ma97_control_d {
    int f_arrays;
    int action;
    int nemin;
    double multiplier;
    int ordering;
    int print_level;
    int scaling;
    double small;
    double u;
    int unit_diagnostics;
    int unit_error;
    int unit_warning;
    long factor_min;
    int solve_blas3;
    int solve_min;
    int solve_mf;
    double consist_tol;
    int ispare[5];
    double rspare[10];
};

struct ma97_info_d {
    int flag;
    int flag68;
    int flag77;
    int matrix_dup;
    int matrix_rank;
    int matrix_outrange;
    int matrix_missing_diag;
    int maxdepth;
    int maxfront;
    int num_delay;
    long num_factor;
    long num_flops;
    int num_neg;
    int num_sup;
    int num_two;
    int ordering;
    int stat;
    int maxsupernode;
    int ispare[4];
    double rspare[10];
};

TEST(Ma97PluginAbi, ControlAndInfoLayoutAndOffsets) {
    static_assert(std::is_standard_layout_v<ma97_control_d>);
    static_assert(std::is_standard_layout_v<ma97_info_d>);

    EXPECT_EQ(alignof(ma97_control_d), alignof(double));
    EXPECT_EQ(alignof(ma97_info_d), alignof(double));

    // Critical field offsets invariant across LP64 and LLP64
    EXPECT_EQ(offsetof(ma97_control_d, f_arrays), 0u);
    EXPECT_EQ(offsetof(ma97_control_d, action), 4u);
    EXPECT_EQ(offsetof(ma97_control_d, nemin), 8u);
    EXPECT_EQ(offsetof(ma97_control_d, multiplier), 16u);
    EXPECT_EQ(offsetof(ma97_control_d, ordering), 24u);
    EXPECT_EQ(offsetof(ma97_control_d, print_level), 28u);
    EXPECT_EQ(offsetof(ma97_control_d, scaling), 32u);
    EXPECT_EQ(offsetof(ma97_control_d, small), 40u);
    EXPECT_EQ(offsetof(ma97_control_d, u), 48u);

    EXPECT_EQ(offsetof(ma97_info_d, flag), 0u);
    EXPECT_EQ(offsetof(ma97_info_d, matrix_dup), 12u);
    EXPECT_EQ(offsetof(ma97_info_d, matrix_rank), 16u);
    EXPECT_EQ(offsetof(ma97_info_d, maxfront), 32u);
    EXPECT_EQ(offsetof(ma97_info_d, num_delay), 36u);

    // Explicit platform ABI check: never assume `long` has Linux LP64 width on Windows LLP64
#if defined(_WIN32)
    static_assert(sizeof(long) == 4, "Windows LLP64 has 32-bit long");
    EXPECT_EQ(offsetof(ma97_info_d, num_neg), 48u);
#else
    static_assert(sizeof(long) == 8, "Linux LP64 has 64-bit long");
    EXPECT_EQ(offsetof(ma97_info_d, num_neg), 56u);
#endif
}
