#include "smf/dense_indef.hpp"
#include <gtest/gtest.h>
#include <vector>
using namespace smf;

TEST(Inertia, AddInertia1x1) {
    InertiaCounts c{};
    add_inertia_1x1(+2.0, 1e-15, c); EXPECT_EQ(c.positive,1);
    add_inertia_1x1(-3.0, 1e-15, c); EXPECT_EQ(c.negative,1);
    add_inertia_1x1(0.0,  1e-15, c); EXPECT_EQ(c.zero,1);
}
TEST(Inertia, AddInertia2x2_Saddle) {
    InertiaCounts c{};
    // [[0,1],[1,0]] det=-1<0 -> pos+neg
    add_inertia_2x2(0.0,1.0,0.0,1e-15,c);
    EXPECT_EQ(c.positive,1); EXPECT_EQ(c.negative,1);
}
TEST(Inertia, AddInertia2x2_BothPos) {
    InertiaCounts c{};
    add_inertia_2x2(2.0,0.0,3.0,1e-15,c);
    EXPECT_EQ(c.positive,2); EXPECT_EQ(c.negative,0);
}
TEST(Inertia, AddInertia2x2_BothNeg) {
    InertiaCounts c{};
    add_inertia_2x2(-2.0,0.0,-3.0,1e-15,c);
    EXPECT_EQ(c.negative,2); EXPECT_EQ(c.positive,0);
}
TEST(Inertia, SPD3x3_Inertia) {
    double A[9]={4,2,1, 2,5,2, 1,2,6};
    std::vector<Pivot> piv; InertiaCounts in;
    dense_ldlt_indef(A,3,3,0.01,1e-20,piv,in);
    EXPECT_EQ(in.positive,3); EXPECT_EQ(in.negative,0);
}
int main(int argc,char**argv){testing::InitGoogleTest(&argc,argv);return RUN_ALL_TESTS();}
