// A scaled factorization prepares its values (S A S, permuted, max |S A S|)
// in one pass (factor_indef); the result must be bitwise the factorization
// of the pre-scaled values with no scaling: every supernode's L and D, the
// inertia and the delays, on KKTs that delay (threshold pivoting) and that
// do not, through both value layouts (the matrix analysed, and new values
// of its pattern).
#include "kkt_test_matrices.hpp"
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/info.hpp"
#include "smf/solver.hpp"
#include "smf/workspace.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <map>
#include <span>
#include <vector>

using namespace smf;

namespace {

struct Factored {
    FactorStatus status = FactorStatus::Success;
    int pos = 0, neg = 0, zero = 0, delays = 0;
    std::vector<std::vector<double>> nodes;  // each supernode's stored L columns with D
};

Factored factor(const Solver &s, const AnalysisKeep &ak, const std::vector<double> &values,
                const Control &ctrl) {
    FactorKeep fk;
    NumericWorkspace ws;
    Info i;
    Factored r;
    r.status = s.factor(ak, std::span<const double>(values), ctrl, i, fk, ws);
    r.pos = i.num_positive;
    r.neg = i.num_negative;
    r.zero = i.num_zero;
    r.delays = i.delayed_pivots;
    std::map<LongInt, const SolveStep *> by_offset;
    for (const SolveStep &st : fk.solve_steps)
        if (st.p > 0) by_offset[st.factor_offset] = &st;
    r.nodes.resize(fk.pivot_types.size());
    for (std::size_t sn = 0; sn < fk.pivot_types.size(); ++sn) {
        const auto it = by_offset.find(fk.factor_col_ptr[sn]);
        if (fk.pivot_types[sn].empty() || it == by_offset.end()) continue;
        const auto vb = fk.factor_values.begin() + static_cast<std::ptrdiff_t>(it->second->factor_offset);
        r.nodes[sn].assign(vb, vb + static_cast<std::ptrdiff_t>(it->second->f) * it->second->p);
    }
    return r;
}

// S A S of the cleaned lower values, as factor_indef forms it:
// a_ij * (s_i * s_j).
std::vector<double> prescaled(const CscLower &A, const std::vector<double> &s) {
    std::vector<double> v = A.values;
    for (Int j = 0; j < A.n; ++j)
        for (Int p = A.col_ptr[static_cast<std::size_t>(j)]; p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
            const std::size_t sp = static_cast<std::size_t>(p);
            v[sp] = A.values[sp] * (s[static_cast<std::size_t>(A.row_idx[sp])] * s[static_cast<std::size_t>(j)]);
        }
    return v;
}

void expect_same(const Factored &a, const Factored &b) {
    EXPECT_EQ(a.status, b.status);
    EXPECT_EQ(a.pos, b.pos);
    EXPECT_EQ(a.neg, b.neg);
    EXPECT_EQ(a.zero, b.zero);
    EXPECT_EQ(a.delays, b.delays);
    ASSERT_EQ(a.nodes.size(), b.nodes.size());
    int bad = 0;
    for (std::size_t sn = 0; sn < a.nodes.size(); ++sn) {
        const auto &x = a.nodes[sn], &y = b.nodes[sn];
        if (x.size() != y.size() || (!x.empty() && std::memcmp(x.data(), y.data(), x.size() * sizeof(double)) != 0))
            ++bad;
    }
    EXPECT_EQ(bad, 0) << "supernodes whose factor differs";
}

void run(double spread, double pivot_u, const char *name) {
    // Pattern and values from cascade_kkt; a second value set of the same
    // pattern (another seed of the scaling) exercises new values.
    const CscLower A = cascade_kkt(40, 400, 2.5, spread);
    const CscLower B = cascade_kkt(40, 400, 2.5, spread, -0.1, 11);
    std::vector<double> s(static_cast<std::size_t>(A.n));
    for (std::size_t i = 0; i < s.size(); ++i) s[i] = std::ldexp(1.0 + 0.37 * static_cast<double>(i % 7), -static_cast<int>(i % 5));
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
    ctrl.pivot_u = pivot_u;
    ctrl.scale_on_delay_ratio = -1.0;
    Solver solver;
    Control an = ctrl;
    an.scaling = ScalingMethod::None;
    Info ia;
    auto ak = solver.analyse(A, an, ia);
    ASSERT_NE(ak.get(), nullptr);
    Control scaled = ctrl;
    scaled.scaling = ScalingMethod::UserSupplied;
    scaled.user_scale = s.data();
    Control plain = ctrl;
    plain.scaling = ScalingMethod::None;
    for (const CscLower *M : {&A, &B}) {
        SCOPED_TRACE(M == &A ? "analysed values" : "new values");
        const Factored fs = factor(solver, *ak, M->values, scaled);
        const Factored fp = factor(solver, *ak, prescaled(*M, s), plain);
        std::printf("[ScaledValuePrep] %s: %d delays, inertia %d/%d/%d\n", name, fs.delays, fs.pos, fs.neg, fs.zero);
        expect_same(fs, fp);
    }
}

} // namespace

TEST(ScaledValuePrep, ThresholdDelaying) { run(1.5, 0.01, "threshold, delaying"); }
TEST(ScaledValuePrep, IpoptPivotTolerance) { run(0.0, 1e-8, "u = 1e-8, no delays"); }
