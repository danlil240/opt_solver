#include "smf/dense_indef.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
using namespace smf;

TEST(LDLT, SPD2x2) {
    // A = [[4,2],[2,3]], SPD, expect 1x1 pivots, inertia (2,0,0)
    double A[4]={4,2, 2,3}; // col-major
    std::vector<Pivot> piv; InertiaCounts in;
    auto s=dense_ldlt_indef(A,2,2,0.01,1e-20,piv,in);
    EXPECT_EQ(s,FactorStatus::Success);
    EXPECT_EQ(in.positive,2); EXPECT_EQ(in.negative,0); EXPECT_EQ(in.zero,0);
}
TEST(LDLT, Indefinite3x3) {
    // A = [[2,1,0],[1,0,1],[0,1,2]] col-major lower: A[0]=2,A[1]=1,A[2]=0,A[3]=0,A[4]=0,A[5]=1,A[6]=0,A[7]=0,A[8]=2
    // lower col-major: col0=[2,1,0], col1=[_,0,1], col2=[_,_,2]
    double A[9]={2,1,0, 0,0,1, 0,0,2};
    std::vector<Pivot> piv; InertiaCounts in;
    dense_ldlt_indef(A,3,3,0.01,1e-20,piv,in);
    // Should produce pivots without crashing; total inertia sums to 3
    EXPECT_EQ(in.positive+in.negative+in.zero, 3);
    // Matrix det = 2*(0*2-1)-1*(1*2-0) = -2-2 = -4 < 0, so 1 neg 2 pos or vice versa
    // Actual: eigenvalues of [[2,1,0],[1,0,1],[0,1,2]]: sum=4>0, so at least 1 pos
    EXPECT_GE(in.positive,1);
}
TEST(LDLT, NearSingular) {
    double A[4]={1e-25,0,0,1.0};
    std::vector<Pivot> piv; InertiaCounts in;
    // Should not crash; singular status expected since |A[0,0]| < small=1e-20
    EXPECT_NO_THROW(dense_ldlt_indef(A,2,2,0.1,1e-20,piv,in));
}
TEST(LDLT, AllNegative2x2) {
    double A[4]={-4,-2, -2,-3};
    std::vector<Pivot> piv; InertiaCounts in;
    dense_ldlt_indef(A,2,2,0.01,1e-20,piv,in);
    EXPECT_EQ(in.negative,2); EXPECT_EQ(in.positive,0);
}
int main(int argc,char**argv){testing::InitGoogleTest(&argc,argv);return RUN_ALL_TESTS();}
