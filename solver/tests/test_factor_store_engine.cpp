// Control::factor_in_place: the tree-parallel factorization's threads write the
// factor straight into per-thread slices of the FactorKeep's store, with no
// gather. Deterministic mode makes runs bit-for-bit comparable (its results do
// not depend on scheduling or the thread count).

#include "kkt_test_matrices.hpp"
#include "smf/control.hpp"
#include "smf/info.hpp"
#include "smf/solver.hpp"
#include "smf/workspace.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

using namespace smf;

namespace {

void set_env(const char *name, const char *value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    if (*value) setenv(name, value, 1);
    else unsetenv(name);
#endif
}

CscLower grid(int m) {  // 5-point stencil, sign-alternating diagonal (indefinite)
    CscLower A;
    A.n = m * m;
    A.col_ptr.push_back(0);
    for (int j = 0; j < A.n; ++j) {
        A.row_idx.push_back(j);
        A.values.push_back((j % 3 == 0) ? -4.5 : 4.0 + 0.001 * (j % 11));
        if ((j % m) + 1 < m) { A.row_idx.push_back(j + 1); A.values.push_back(-1.0); }
        if (j + m < A.n) { A.row_idx.push_back(j + m); A.values.push_back(-1.0); }
        A.col_ptr.push_back(static_cast<Int>(A.row_idx.size()));
    }
    return A;
}

Control base(int threads, bool in_place) {
    Control c;
    c.matrix_type = MatrixType::RealSymmetricIndefinite;
    c.num_threads = threads;
    c.auto_threads = false;
    c.deterministic = true;
    c.factor_in_place = in_place;
    return c;
}

struct Out {
    FactorStatus status = FactorStatus::Success;
    Info info;
    std::vector<double> x;
};

std::vector<double> rhs(Int n) {
    std::vector<double> b(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = std::sin(0.017 * static_cast<double>(i)) + 0.5;
    return b;
}

Out factor_solve(const Solver &s, const AnalysisKeep &ak, const CscLower &A, const Control &c,
                 FactorKeep &fk, NumericWorkspace &ws) {
    Out o;
    o.status = s.factor(ak, A, c, o.info, fk, ws);
    if (o.status != FactorStatus::Success) return o;
    o.x = rhs(A.n);
    SolveWorkspace sws;
    EXPECT_EQ(s.solve(fk, c, o.info, o.x.data(), static_cast<int>(A.n), 1, sws), 0);
    return o;
}

Out fresh(const CscLower &A, const Control &c) {
    Solver s;
    Info i;
    auto ak = s.analyse(A, c, i);
    EXPECT_NE(ak.get(), nullptr);
    FactorKeep fk;
    NumericWorkspace ws;
    return factor_solve(s, *ak, A, c, fk, ws);
}

bool same_bits(const std::vector<double> &a, const std::vector<double> &b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * 8) == 0);
}

}  // namespace

TEST(FactorStoreEngine, InPlaceMatchesGather) {
    const CscLower mats[] = {grid(200), opf_like_kkt(20000, 8000, 1)};
    for (const CscLower &A : mats) {
        const Out off = fresh(A, base(8, false));
        const Out on = fresh(A, base(8, true));
        ASSERT_EQ(off.status, FactorStatus::Success);
        ASSERT_EQ(on.status, FactorStatus::Success);
        ASSERT_GT(on.info.threads_used, 1) << "premise: tree-parallel at 8 threads";
        EXPECT_TRUE(same_bits(off.x, on.x)) << "n " << A.n;
        EXPECT_EQ(off.info.num_negative, on.info.num_negative);
        EXPECT_EQ(on.info.gather_seconds, 0.0);
        EXPECT_EQ(on.info.factor_store_regathers, 0);
        EXPECT_GT(off.info.gather_seconds, 0.0);
    }
}

TEST(FactorStoreEngine, OverflowRegathers) {
    const CscLower A = grid(200);
    const Out off = fresh(A, base(8, false));
    set_env("SMF_TEST_STORE_SLICE_CAP", "4096");
    const Out on = fresh(A, base(8, true));
    set_env("SMF_TEST_STORE_SLICE_CAP", "");
    ASSERT_EQ(on.status, FactorStatus::Success);
    EXPECT_GE(on.info.factor_store_regathers, 1);
    EXPECT_TRUE(same_bits(off.x, on.x));
}

TEST(FactorStoreEngine, ReservationFailureRegathers) {
    const CscLower A = grid(200);
    const Out off = fresh(A, base(8, false));
    set_env("SMF_TEST_STORE_FAIL_RESERVE", "1");
    const Out on = fresh(A, base(8, true));
    set_env("SMF_TEST_STORE_FAIL_RESERVE", "");
    ASSERT_EQ(on.status, FactorStatus::Success);
    EXPECT_EQ(on.info.factor_store_regathers, 1);
    EXPECT_TRUE(same_bits(off.x, on.x));
}

TEST(FactorStoreEngine, RefactorAcrossThreadCounts) {
    const CscLower A = opf_like_kkt(20000, 8000, 2);
    Solver s;
    Info i;
    auto ak = s.analyse(A, base(8, true), i);
    ASSERT_NE(ak.get(), nullptr);
    FactorKeep fk;
    NumericWorkspace ws;
    for (const int t : {8, 2, 1, 8}) {
        const Out got = factor_solve(s, *ak, A, base(t, true), fk, ws);
        ASSERT_EQ(got.status, FactorStatus::Success) << t;
        if (t > 1) EXPECT_GT(got.info.threads_used, 1) << "premise: tree-parallel at " << t;
        FactorKeep fk2;
        NumericWorkspace ws2;
        const Out ref = factor_solve(s, *ak, A, base(t, true), fk2, ws2);
        EXPECT_TRUE(same_bits(got.x, ref.x)) << "threads " << t;
    }
}

TEST(FactorStoreEngine, TinyAndEmptyMatrices) {
    CscLower one;
    one.n = 1;
    one.col_ptr = {0, 1};
    one.row_idx = {0};
    one.values = {-2.0};
    const Out o1 = fresh(one, base(8, true));
    ASSERT_EQ(o1.status, FactorStatus::Success);
    EXPECT_DOUBLE_EQ(o1.x[0], rhs(1)[0] / -2.0);

    CscLower dense;  // one supernode: a 5x5 KKT block
    dense.n = 5;
    dense.col_ptr.push_back(0);
    for (int j = 0; j < 5; ++j) {
        for (int i = j; i < 5; ++i) {
            dense.row_idx.push_back(i);
            dense.values.push_back(i == j ? (j < 3 ? 4.0 : 0.0) : (i >= 3 && j < 3 ? 1.0 : 0.5));
        }
        dense.col_ptr.push_back(static_cast<Int>(dense.row_idx.size()));
    }
    const Out o5 = fresh(dense, base(8, true));
    const Out o5off = fresh(dense, base(8, false));
    ASSERT_EQ(o5.status, o5off.status);
    EXPECT_TRUE(same_bits(o5.x, o5off.x));
}

TEST(FactorStoreEngine, MovedFactorKeepSolves) {
    const CscLower A = grid(150);
    Solver s;
    Info i;
    const Control c = base(8, true);
    auto ak = s.analyse(A, c, i);
    ASSERT_NE(ak.get(), nullptr);
    FactorKeep fk;
    NumericWorkspace ws;
    const Out ref = factor_solve(s, *ak, A, c, fk, ws);
    ASSERT_EQ(ref.status, FactorStatus::Success);
    const auto solve_with = [&](const FactorKeep &k) {
        std::vector<double> x = rhs(A.n);
        SolveWorkspace sws;
        Info si;
        EXPECT_EQ(s.solve(k, c, si, x.data(), static_cast<int>(A.n), 1, sws), 0);
        return x;
    };
    const FactorKeep copied(fk);
    EXPECT_TRUE(same_bits(solve_with(copied), ref.x));
    const FactorKeep moved(std::move(fk));
    EXPECT_TRUE(same_bits(solve_with(moved), ref.x));
}
