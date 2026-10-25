#include "../benchmarks/adapters/adapter.hpp"
#include "../benchmarks/common/validation.hpp"

#include <gtest/gtest.h>

#include <vector>

using namespace smf;
using namespace smf::bench;

namespace {

CscLower make_identity_spd_3x3() {
    CscLower A;
    A.n = 3;
    A.col_ptr = {0, 1, 2, 3};
    A.row_idx = {0, 1, 2};
    A.values = {1.0, 1.0, 1.0};
    return A;
}

CscLower make_exchange_2x2_indef() {
    CscLower A;
    A.n = 2;
    A.col_ptr = {0, 2, 3};
    A.row_idx = {0, 1, 1};
    A.values = {0.0, 1.0, 0.0};
    return A;
}

CscLower make_singular_2x2() {
    CscLower A;
    A.n = 2;
    A.col_ptr = {0, 1, 2};
    A.row_idx = {0, 1};
    A.values = {2.0, 0.0};
    return A;
}

} // namespace

TEST(BenchmarkAdapters, SmfAdapterAndAllRegisteredAdapters) {
    auto ids = all_registered_adapter_ids();
    ASSERT_FALSE(ids.empty());

    for (const auto &id : ids) {
        auto adapter = make_solver_adapter(id);
        ASSERT_NE(adapter.get(), nullptr) << "Null adapter for " << id;
        AdapterIdentity ident = adapter->identity();
        EXPECT_EQ(ident.solver_id, id);
        EXPECT_FALSE(ident.integer_abi.empty());
        // Never label libsmf_ma97 as genuine HSL ma97!
        if (id == "ma97") {
            EXPECT_EQ(ident.library_path.find("smf_ma97"), std::string::npos);
        }

        if (!ident.available) {
            CscLower I3 = make_identity_spd_3x3();
            EXPECT_EQ(adapter->analyse(I3), Outcome::Unavailable);
            EXPECT_EQ(adapter->factor(I3.values), Outcome::Unavailable);
            std::vector<double> b = {1.0, 2.0, 3.0};
            EXPECT_EQ(adapter->solve(b, 1), Outcome::Unavailable);
            continue;
        }

        // Test 1: Identity SPD
        if (ident.supports_spd) {
            Control ctrl;
            ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
            adapter->configure(ctrl, "recommended");
            CscLower I3 = make_identity_spd_3x3();
            ASSERT_EQ(adapter->analyse(I3), Outcome::Success);
            ASSERT_EQ(adapter->factor(I3.values), Outcome::Success);
            std::vector<double> b = {1.0, 2.0, 3.0};
            std::vector<double> x = b;
            ASSERT_EQ(adapter->solve(x, 1), Outcome::Success);
            auto v = validate(I3, b, x, 1);
            EXPECT_TRUE(is_qualified_validation(v));

            // Changed-value replay on the same pattern: diag(2, 4, 8)
            std::vector<double> updated_vals = {2.0, 4.0, 8.0};
            ASSERT_EQ(adapter->factor(updated_vals), Outcome::Success);
            // Multi-RHS (nrhs = 2)
            std::vector<double> b2 = {2.0, 8.0, 24.0, 4.0, 12.0, 16.0};
            std::vector<double> x2 = b2;
            ASSERT_EQ(adapter->solve(x2, 2), Outcome::Success);
            CscLower A_upd = I3;
            A_upd.values = updated_vals;
            auto v2 = validate(A_upd, b2, x2, 2);
            EXPECT_TRUE(is_qualified_validation(v2));
        }

        // Test 2: 2x2 exchange matrix where indefinite is supported
        if (ident.supports_indefinite) {
            Control ctrl;
            ctrl.matrix_type = MatrixType::RealSymmetricIndefinite;
            adapter->configure(ctrl, "recommended");
            CscLower E2 = make_exchange_2x2_indef();
            ASSERT_EQ(adapter->analyse(E2), Outcome::Success);
            ASSERT_EQ(adapter->factor(E2.values), Outcome::Success);
            std::vector<double> b = {1.0, 2.0};
            std::vector<double> x = b;
            ASSERT_EQ(adapter->solve(x, 1), Outcome::Success);
            EXPECT_NEAR(x[0], 2.0, 1e-12);
            EXPECT_NEAR(x[1], 1.0, 1e-12);
            AdapterStats st = adapter->stats();
            EXPECT_EQ(st.inertia_pos, 1);
            EXPECT_EQ(st.inertia_neg, 1);
            EXPECT_EQ(st.inertia_zero, 0);
        }

        // Test 3: Singularity detection
        {
            Control ctrl;
            ctrl.matrix_type = ident.supports_indefinite
                                   ? MatrixType::RealSymmetricIndefinite
                                   : MatrixType::RealSymmetricPositiveDefinite;
            adapter->configure(ctrl, "recommended");
            CscLower S2 = make_singular_2x2();
            ASSERT_EQ(adapter->analyse(S2), Outcome::Success);
            Outcome f_out = adapter->factor(S2.values);
            EXPECT_NE(f_out, Outcome::Success);
        }
    }
}
