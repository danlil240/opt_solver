#include "smf/dense_indef.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
using namespace smf;

// 3x3 SPD: A = [[4,2,1],[2,5,2],[1,2,6]], L = [[2,0,0],[1,2,0],[0.5,0.75,sqrt(6-0.5^2-0.75^2)]]
TEST(Cholesky, SPD3x3) {
    // col-major: A[i+j*3]
    double A[9] = {4,2,1, 2,5,2, 1,2,6};
    double orig[9]; for(int i=0;i<9;i++) orig[i]=A[i];
    auto s = dense_cholesky_lower(A, 3, 3);
    EXPECT_EQ(s, FactorStatus::Success);
    // reconstruct L*Lt (col-major) and compare to orig (lower only)
    for(int j=0;j<3;j++) for(int i=j;i<3;i++){
        double v=0; for(int k=0;k<=j;k++) v+=A[i+k*3]*A[j+k*3];
        EXPECT_NEAR(v, orig[i+j*3], 1e-10) << "i="<<i<<" j="<<j;
    }
}
TEST(Cholesky, Indefinite) {
    double A[4] = {-1,0, 0,1}; // col-major 2x2
    EXPECT_EQ(dense_cholesky_lower(A,2,2), FactorStatus::NotPositiveDefinite);
}
TEST(Cholesky, Identity2x2) {
    double A[4]={1,0,0,1};
    EXPECT_EQ(dense_cholesky_lower(A,2,2), FactorStatus::Success);
    EXPECT_NEAR(A[0],1.0,1e-15); EXPECT_NEAR(A[3],1.0,1e-15);
}
TEST(Cholesky, LargerLda) {
    // 2x2 matrix in a 4-row buffer (lda=4)
    double A[8]={4,2,99,99, 2,5,99,99}; // col 0: [4,2,99,99], col 1: [2,5,99,99]
    auto s = dense_cholesky_lower(A,2,4);
    EXPECT_EQ(s, FactorStatus::Success);
    EXPECT_NEAR(A[0], 2.0, 1e-12); // L[0,0]=sqrt(4)
}
int main(int argc,char**argv){testing::InitGoogleTest(&argc,argv);return RUN_ALL_TESTS();}
