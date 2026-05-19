#include "smf/factor_indef.hpp"
#include <gtest/gtest.h>

using namespace smf;

TEST(FactorIndef, HeaderCompiles) {
  FactorKeep fk;
  EXPECT_TRUE(fk.factor_values.empty());
  EXPECT_NE(FactorStatus::Singular, FactorStatus::Success);
}

TEST(FactorIndef, InertiaEnums) {
  InertiaCounts ic;
  add_inertia_1x1(1.0, 1e-15, ic);
  EXPECT_EQ(ic.positive, 1);
  add_inertia_1x1(-1.0, 1e-15, ic);
  EXPECT_EQ(ic.negative, 1);
}
