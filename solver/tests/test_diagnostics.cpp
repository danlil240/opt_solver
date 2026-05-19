#include "smf/diagnostics.hpp"
#include <gtest/gtest.h>
#include <string>
using namespace smf;

TEST(Diagnostics, EmptyFactorKeepCount) {
  FactorKeep fk;
  EXPECT_EQ(count_factor_entries(fk), 0LL);
}

TEST(Diagnostics, FormatInfoSmoke) {
  Info info;
  std::string s = format_info(info);
  EXPECT_NE(s.find("status"), std::string::npos);
  EXPECT_NE(s.find("factor_seconds"), std::string::npos);
}

TEST(Diagnostics, CountFlopsZero) {
  // Empty analysis keep → 0 flops
  AnalysisKeep ak;
  EXPECT_DOUBLE_EQ(count_flops(ak), 0.0);
}
