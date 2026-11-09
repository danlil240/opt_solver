#include "smf/analysis.hpp"
#include "smf/solve_diag.hpp"
#include <gtest/gtest.h>

using namespace smf;

// ---------------------------------------------------------------------------
// TEST: PermRoundtrip
// apply_perm then apply_inv_perm must be the identity.
// ---------------------------------------------------------------------------
TEST(SolveDiag, PermRoundtrip) {
  constexpr int n = 4;
  FactorKeep fk;
  fk.perm = {2, 0, 3, 1};  // perm[new] = old
  fk.iperm = {1, 3, 0, 2}; // iperm[old] = new

  double x[n] = {10.0, 20.0, 30.0, 40.0};
  double y[n] = {};
  double z[n] = {};

  apply_perm(fk, x, y, n);
  apply_inv_perm(fk, y, z, n);

  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(z[i], x[i], 1e-15) << "mismatch at index " << i;
  }
}

// ---------------------------------------------------------------------------
// TEST: Identity1x1
// With is_posdef=true, solve_diag must be a no-op.
// ---------------------------------------------------------------------------
TEST(SolveDiag, Identity1x1) {
  constexpr int n = 3;
  FactorKeep fk;
  fk.is_posdef = true;

  double x[n] = {1.0, 2.0, 3.0};
  const double x_orig[n] = {1.0, 2.0, 3.0};

  solve_diag(fk, x, n);

  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(x[i], x_orig[i], 1e-15) << "SPD no-op failed at index " << i;
  }
}

// ---------------------------------------------------------------------------
// TEST: BlockD2x2
// D = [[2, 1], [1, 2]] (d00=2, d10=1, d11=2).
// D^{-1} = [[2, -1], [-1, 2]] / 3.
// x = [3, 5] → D^{-1} x = [(2*3 - 1*5)/3, (-1*3 + 2*5)/3] = [1/3, 7/3].
// ---------------------------------------------------------------------------
TEST(SolveDiag, BlockD2x2) {
  // Build a minimal AnalysisKeep: 1 supernode, width=2, front_size=2.
  AnalysisKeep ak;
  ak.n = 2;

  Supernode sn;
  sn.col_start = 0;
  sn.col_end = 2;
  sn.parent = -1;
  ak.supernodes.push_back(std::move(sn));

  ak.child_ptr = {0, 0};
  ak.front_row_ptr = {0, 2};
  ak.front_row_idx = {0, 1};

  // factor_values layout: f=2 (front_size), p=2 (width), col-major.
  //   (row, col) → col*f + row
  //   (0,0)=0: d00 = 2.0
  //   (1,0)=1: d10 = 1.0   ← lower-triangle off-diagonal of 2×2 block
  //   (0,1)=2: unused (upper triangle) = 0.0
  //   (1,1)=3: d11 = 2.0
  FactorKeep fk;
  fk.is_posdef = false;
  fk.analysis = &ak;
  fk.factor_col_ptr = {0}; // supernode 0 starts at index 0
  fk.factor_values = {2.0, 1.0, 0.0, 2.0};
  fk.perm = {0, 1};
  fk.iperm = {0, 1};
  fk.pivot_types = {{2, -1}}; // col 0: first of 2×2; col 1: second of 2×2

  double x[2] = {3.0, 5.0};
  solve_diag(fk, x, 2);

  EXPECT_NEAR(x[0], 1.0 / 3.0, 1e-14) << "x[0] wrong after 2×2 D solve";
  EXPECT_NEAR(x[1], 7.0 / 3.0, 1e-14) << "x[1] wrong after 2×2 D solve";
}

// Control::parallel_dinv: the solve's D^-1 table (FactorKeep::dinv,
// dinv_kind, dinv_offset) is built over ranges of solve steps on the
// factorization's pool; it is bit-for-bit the serial build. Deterministic mode
// makes the two factorizations identical, so the tables can be compared.
#include "kkt_test_matrices.hpp"
#include "smf/solver.hpp"
#include "smf/workspace.hpp"

#include <cstring>
#include <vector>

TEST(SolveDiag, ParallelDinvEqualsSerialBuild) {
    using namespace smf;
    const CscLower mats[] = {opf_like_kkt(20000, 8000, 1), cascade_kkt(90, 300, 2.5, 1.5)};
    for (const CscLower &A : mats) {
        std::vector<double> dinv[2], x[2];
        std::vector<std::int8_t> kind[2];
        std::vector<LongInt> off[2];
        for (int v = 0; v < 2; ++v) {
            Control c;
            c.matrix_type = MatrixType::RealSymmetricIndefinite;
            c.num_threads = 8;
            c.auto_threads = false;
            c.deterministic = true;
            c.parallel_dinv = v == 1;
            Solver s;
            Info info;
            auto ak = s.analyse(A, c, info);
            ASSERT_NE(ak.get(), nullptr);
            FactorKeep fk;
            NumericWorkspace ws;
            const FactorStatus st = s.factor(*ak, A, c, info, fk, ws);
            ASSERT_TRUE(st == FactorStatus::Success || st == FactorStatus::Singular);
            ASSERT_GT(info.threads_used, 1) << "premise: tree-parallel";
            ASSERT_FALSE(fk.dinv_offset.empty()) << "premise: a D^-1 table";
            dinv[v] = fk.dinv;
            kind[v] = fk.dinv_kind;
            off[v] = fk.dinv_offset;
            x[v].assign(static_cast<std::size_t>(A.n), 1.0);
            SolveWorkspace sws;
            ASSERT_EQ(s.solve(fk, c, info, x[v].data(), static_cast<int>(A.n), 1, sws), 0);
        }
        EXPECT_EQ(off[0], off[1]);
        EXPECT_EQ(kind[0], kind[1]);
        ASSERT_EQ(dinv[0].size(), dinv[1].size());
        EXPECT_EQ(std::memcmp(dinv[0].data(), dinv[1].data(), dinv[0].size() * 8), 0);
        EXPECT_EQ(std::memcmp(x[0].data(), x[1].data(), x[0].size() * 8), 0);
    }
}

namespace {
struct DiagRun {
    std::vector<smf::DiagSolveEntry> diag;
    int two_by_two = 0;
    int scanned_two = 0;
};
DiagRun diag_run(const smf::CscLower &A, bool lazy) {
    using namespace smf;
    Control c;
    c.matrix_type = MatrixType::RealSymmetricIndefinite;
    c.num_threads = 8;
    c.auto_threads = false;
    c.deterministic = true;
    c.lazy_diag_entries = lazy;
    Solver s;
    Info info;
    auto ak = s.analyse(A, c, info);
    EXPECT_NE(ak.get(), nullptr);
    FactorKeep fk;
    NumericWorkspace ws;
    s.factor(*ak, A, c, info, fk, ws);
    EXPECT_GT(info.threads_used, 1) << "premise: tree-parallel";
    DiagRun r;
    r.diag = fk.diag();
    r.two_by_two = info.two_by_two_pivots;
    for (const auto &e : r.diag) r.scanned_two += e.tag == 2;
    const FactorKeep copy(fk);  // a copy builds the same entries
    EXPECT_EQ(copy.diag().size(), r.diag.size());
    return r;
}
bool same_entries(const std::vector<smf::DiagSolveEntry> &a, const std::vector<smf::DiagSolveEntry> &b) {
    if (a.size() != b.size()) return false;
    const auto bits = [](double x, double y) { return std::memcmp(&x, &y, sizeof x) == 0; };
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i].tag != b[i].tag || a[i].was_zero != b[i].was_zero || a[i].row0 != b[i].row0 ||
            a[i].row1 != b[i].row1 || !bits(a[i].d00, b[i].d00) || !bits(a[i].d10, b[i].d10) ||
            !bits(a[i].d11, b[i].d11))
            return false;
    return true;
}
}  // namespace

// Control::lazy_diag_entries: a tree-parallel factorization leaves the D
// entries in its threads' lists and FactorKeep::diag() merges them into
// elimination order on first use: the same entries as the eager merge.
TEST(SolveDiag, LazyDiagEqualsEager) {
    const smf::CscLower A = smf::cascade_kkt(90, 300, 2.5, 1.5);
    const DiagRun eager = diag_run(A, false), lazy = diag_run(A, true);
    ASSERT_FALSE(eager.diag.empty());
    EXPECT_TRUE(same_entries(eager.diag, lazy.diag));
}

// Info::two_by_two_pivots counts the 2x2 pivot blocks as the factorization
// makes them: the IPOPT plugin's count, without a scan of the D entries.
TEST(SolveDiag, TwoByTwoCounterMatchesScan) {
    const smf::CscLower A = smf::cascade_kkt(90, 300, 2.5, 1.5);
    for (const bool lazy : {false, true}) {
        const DiagRun r = diag_run(A, lazy);
        ASSERT_GT(r.scanned_two, 0) << "premise: 2x2 pivots";
        EXPECT_EQ(r.two_by_two, r.scanned_two) << lazy;
    }
}
