// Refactorization into the same FactorKeep reuses factor_values without
// clearing it (factor_indef): what the last factorization left there,
// including a delay cascade's differently laid-out factor, must not reach
// the next one. Every factorization of a V1, V2, V1 sequence on one analysis,
// FactorKeep and workspace is bitwise-identical (each supernode's factor
// and D, inertia, solution) to a fresh factorization of the same values,
// also with the buffer poisoned with NaN in between, under threshold and
// static pivoting, at 1 thread and at 8 threads in deterministic mode (tiled
// top fronts).
#include "kkt_test_matrices.hpp"
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/info.hpp"
#include "smf/solver.hpp"
#include "smf/workspace.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <span>
#include <vector>

using namespace smf;

namespace {

// The factor of each supernode: its rows (in pivot order) and its stored L
// columns with D. At 8 threads the subtrees' factors sit in factor_values in
// the order the lanes finished, so they are compared per supernode
// (factor_col_ptr), not as one array.
struct NodeFactor {
    std::vector<Int> rows;
    std::vector<double> values;
    std::vector<int8_t> pivots;
};

struct Result {
    FactorStatus status = FactorStatus::Success;
    int pos = 0, neg = 0, zero = 0, delays = 0, tiled = 0;
    std::vector<NodeFactor> nodes;
    std::vector<double> x;
};

Result factor_and_solve(const Solver &s, const AnalysisKeep &ak, const CscLower &A,
                        const Control &ctrl, FactorKeep &fk, NumericWorkspace &ws) {
    Result r;
    Info i;
    r.status = s.factor(ak, std::span<const double>(A.values), ctrl, i, fk, ws);
    r.pos = i.num_positive;
    r.neg = i.num_negative;
    r.zero = i.num_zero;
    r.delays = i.delayed_pivots;
    r.tiled = i.top_tile_fronts;
    // Each step with pivots starts at its own factor offset.
    std::map<LongInt, const SolveStep *> by_offset;
    for (const SolveStep &st : fk.solve_steps)
        if (st.p > 0) by_offset[st.factor_offset] = &st;
    const std::size_t ns = fk.pivot_types.size();
    r.nodes.resize(ns);
    for (std::size_t sn = 0; sn < ns; ++sn) {
        NodeFactor &nf = r.nodes[sn];
        nf.pivots = fk.pivot_types[sn];
        const auto it = by_offset.find(fk.factor_col_ptr[sn]);
        if (nf.pivots.empty() || it == by_offset.end()) continue;
        const SolveStep &st = *it->second;
        const auto rb = fk.solve_row_indices.begin() + static_cast<std::ptrdiff_t>(st.row_offset);
        nf.rows.assign(rb, rb + st.f);
        const auto vb = fk.factor_values.begin() + static_cast<std::ptrdiff_t>(st.factor_offset);
        nf.values.assign(vb, vb + static_cast<std::ptrdiff_t>(st.f) * st.p);
    }
    const int n = static_cast<int>(A.n);
    r.x.resize(static_cast<std::size_t>(n));
    for (int k = 0; k < n; ++k) r.x[static_cast<std::size_t>(k)] = 1.0 + 0.01 * (k % 97);
    if (r.status == FactorStatus::Success) {
        Info si;
        EXPECT_EQ(s.solve(fk, ctrl, si, r.x.data(), n, 1), 0);
    }
    return r;
}

bool same_bits(const std::vector<double> &a, const std::vector<double> &b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}

void expect_identical(const Result &got, const Result &fresh, const char *what) {
    EXPECT_EQ(got.status, fresh.status) << what;
    EXPECT_EQ(got.pos, fresh.pos) << what;
    EXPECT_EQ(got.neg, fresh.neg) << what;
    EXPECT_EQ(got.zero, fresh.zero) << what;
    EXPECT_EQ(got.delays, fresh.delays) << what;
    ASSERT_EQ(got.nodes.size(), fresh.nodes.size()) << what;
    int bad = 0;
    for (std::size_t sn = 0; sn < got.nodes.size(); ++sn) {
        const NodeFactor &g = got.nodes[sn], &f = fresh.nodes[sn];
        if (g.rows != f.rows || g.pivots != f.pivots || !same_bits(g.values, f.values)) {
            if (++bad <= 3) ADD_FAILURE() << what << ": supernode " << sn << "'s factor (L and D) differs";
        }
    }
    EXPECT_EQ(bad, 0) << what << ": supernodes whose factor differs from a fresh factorization";
    EXPECT_TRUE(same_bits(got.x, fresh.x)) << what << ": solution differs";
}

// V1: unscaled threshold pivoting cascades (cascade_kkt's spread); V2: the
// same pattern without the scaling, few delays.
void run_sequence(Control ctrl, const char *name) {
    const CscLower V1 = cascade_kkt(90, 300, 2.5, 1.5);
    const CscLower V2 = cascade_kkt(90, 300, 2.5, 0.0);
    ASSERT_EQ(V1.values.size(), V2.values.size());
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    ctrl.amalgamation = Amalgamation::Smf;
    ctrl.scaling = ScalingMethod::None;
    ctrl.scale_on_delay_ratio = -1.0;  // no rescue: no scale carried between factorizations
    Solver s;
    Info ia;
    auto ak = s.analyse(V1, ctrl, ia);
    ASSERT_NE(ak.get(), nullptr);

    const auto fresh = [&](const CscLower &A) {
        FactorKeep fk;
        NumericWorkspace ws;
        return factor_and_solve(s, *ak, A, ctrl, fk, ws);
    };
    const Result f1 = fresh(V1), f2 = fresh(V2);
    std::printf("[FactorValuesReuse] %s: V1 %d delays, V2 %d delays, %d / %d tiled fronts\n", name,
                f1.delays, f2.delays, f1.tiled, f2.tiled);
    if (ctrl.deterministic && ctrl.pivoting == Pivoting::Threshold) {
        EXPECT_GT(f1.tiled, 0) << "premise: tiled top fronts";
    }
    if (ctrl.pivoting == Pivoting::Threshold) {
        EXPECT_GT(f1.delays, 10 * f2.delays + 100) << "premise: V1 cascades, V2 does not";
    }

    FactorKeep fk;
    NumericWorkspace ws;
    expect_identical(factor_and_solve(s, *ak, V1, ctrl, fk, ws), f1, "V1 (first)");
    expect_identical(factor_and_solve(s, *ak, V2, ctrl, fk, ws), f2, "V2 after V1");
    expect_identical(factor_and_solve(s, *ak, V1, ctrl, fk, ws), f1, "V1 after V2");
    // Whatever a front leaves unwritten keeps the poison; none of it may be
    // read before it is written.
    fk.factor_values.assign(fk.factor_values.capacity(),
                            std::numeric_limits<double>::quiet_NaN());
    expect_identical(factor_and_solve(s, *ak, V2, ctrl, fk, ws), f2, "V2 after NaN poison");
    fk.factor_values.assign(fk.factor_values.capacity(),
                            std::numeric_limits<double>::quiet_NaN());
    expect_identical(factor_and_solve(s, *ak, V1, ctrl, fk, ws), f1, "V1 after NaN poison");
}

Control threads(int t, bool deterministic) {
    Control c;
    c.num_threads = t;
    c.deterministic = deterministic;
    // Tile the top fronts of this n = 8400 KKT (deterministic mode tiles
    // from this size at every thread count).
    if (deterministic) c.top_tile_min_rows = 128;
    return c;
}

} // namespace

TEST(FactorValuesReuse, Threshold1T) { run_sequence(threads(1, false), "threshold 1T"); }

TEST(FactorValuesReuse, Static1T) {
    Control c = threads(1, false);
    c.pivoting = Pivoting::Static;
    run_sequence(c, "static 1T");
}

TEST(FactorValuesReuse, Threshold8TDeterministic) {
    run_sequence(threads(8, true), "threshold 8T deterministic");
}

TEST(FactorValuesReuse, Static8TDeterministic) {
    Control c = threads(8, true);
    c.pivoting = Pivoting::Static;
    run_sequence(c, "static 8T deterministic");
}
