#pragma once
#include "smf/types.hpp"
#include <cstdint>
#include <vector>

namespace smf {

enum class PivotType : uint8_t { OneByOne = 1, TwoByTwo = 2 };

struct Pivot {
    PivotType type;
    int       col0;
    int       col1;
    double    d00;
    double    d10;
    double    d11;
};

struct InertiaCounts {
    int positive = 0;
    int negative = 0;
    int zero     = 0;
};

FactorStatus dense_cholesky_lower(double* A, int n, int lda) noexcept;

FactorStatus dense_ldlt_indef(double* A, int n, int lda, double u, double small,
    std::vector<Pivot>& pivots, InertiaCounts& inertia) noexcept;

void add_inertia_1x1(double d, double small, InertiaCounts& out) noexcept;

void add_inertia_2x2(double d00, double d10, double d11, double small,
    InertiaCounts& out) noexcept;

} // namespace smf
