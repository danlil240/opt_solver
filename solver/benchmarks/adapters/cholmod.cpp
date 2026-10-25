#include "adapter.hpp"

#if defined(SMF_HAS_CHOLMOD_IMPL)
#include <cholmod.h>
#endif

#include <chrono>
#include <cstring>
#include <vector>

namespace smf::bench {

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0, Clock::time_point t1) noexcept {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

#if defined(SMF_HAS_CHOLMOD_IMPL)

class CholmodSolverAdapter final : public SolverAdapter {
public:
    CholmodSolverAdapter() {
        cholmod_start(&c_);
        c_.print = 0;  // silence benchmark output
    }

    ~CholmodSolverAdapter() override {
        if (L_) cholmod_free_factor(&L_, &c_);
        if (A_) cholmod_free_sparse(&A_, &c_);
        cholmod_finish(&c_);
    }

    AdapterIdentity identity() const override {
        AdapterIdentity id;
        id.solver_id = "cholmod";
        id.version = "5.4.0";
        id.library_path = "cholmod";
        id.integer_abi = "int32";
        id.available = true;
        id.supports_spd = true;
        id.supports_indefinite = false;
        return id;
    }

    void configure(const Control &ctrl, const std::string &) override {
        indefinite_ =
            (ctrl.matrix_type == MatrixType::RealSymmetricIndefinite);
    }

    Outcome analyse(const CscLower &A) override {
        stats_ = AdapterStats{};
        if (L_) {
            cholmod_free_factor(&L_, &c_);
            L_ = nullptr;
        }
        if (A_) {
            cholmod_free_sparse(&A_, &c_);
            A_ = nullptr;
        }
        n_ = A.n;
        const std::size_t nz = A.values.size();
        if (n_ <= 0 || A.col_ptr.size() != static_cast<std::size_t>(n_) + 1) {
            stats_.outcome = Outcome::InvalidInput;
            stats_.reason = "cholmod: invalid CSC shape";
            return stats_.outcome;
        }

        A_ = cholmod_allocate_sparse(
            static_cast<std::size_t>(n_), static_cast<std::size_t>(n_), nz,
            /*sorted=*/1, /*packed=*/1, /*stype=*/-1, CHOLMOD_REAL, &c_);
        if (!A_) {
            stats_.outcome = Outcome::OutOfMemory;
            stats_.reason = "cholmod_allocate_sparse failed";
            return stats_.outcome;
        }
        int *p = static_cast<int *>(A_->p);
        int *i = static_cast<int *>(A_->i);
        double *x = static_cast<double *>(A_->x);
        for (std::size_t k = 0; k <= static_cast<std::size_t>(n_); ++k) {
            p[k] = static_cast<int>(A.col_ptr[k]);
        }
        for (std::size_t k = 0; k < nz; ++k) {
            i[k] = static_cast<int>(A.row_idx[k]);
            x[k] = A.values[k];
        }

        const auto t0 = Clock::now();
        L_ = cholmod_analyze(A_, &c_);
        const auto t1 = Clock::now();
        stats_.analyse_ms = ms_since(t0, t1);

        if (!L_ || c_.status != CHOLMOD_OK) {
            stats_.outcome = (c_.status == CHOLMOD_OUT_OF_MEMORY)
                                 ? Outcome::OutOfMemory
                                 : Outcome::Failed;
            stats_.reason = "cholmod_analyze failed (status " +
                            std::to_string(c_.status) + ")";
            return stats_.outcome;
        }
        stats_.outcome = Outcome::Success;
        return Outcome::Success;
    }

    Outcome factor(std::span<const double> values) override {
        if (!A_ || !L_ || values.size() != A_->nzmax) {
            stats_.outcome = Outcome::InvalidInput;
            stats_.reason = "cholmod: factor before analyse or size mismatch";
            return stats_.outcome;
        }
        std::memcpy(A_->x, values.data(), values.size() * sizeof(double));

        const auto t0 = Clock::now();
        const int ok = cholmod_factorize(A_, L_, &c_);
        const auto t1 = Clock::now();
        stats_.factor_ms = ms_since(t0, t1);
        stats_.factor_entries = L_ ? static_cast<int64_t>(c_.lnz) : 0;

        if (!ok || c_.status != CHOLMOD_OK ||
            L_->minor < static_cast<std::size_t>(n_)) {
            if (c_.status == CHOLMOD_OUT_OF_MEMORY) {
                stats_.outcome = Outcome::OutOfMemory;
                stats_.reason = "cholmod_factorize out of memory";
            } else {
                stats_.outcome = Outcome::Singular;
                stats_.reason =
                    "cholmod: matrix not positive definite at column " +
                    std::to_string(static_cast<long long>(L_->minor));
            }
            return stats_.outcome;
        }
        stats_.inertia_pos = static_cast<int>(n_);
        stats_.inertia_neg = 0;
        stats_.inertia_zero = 0;
        stats_.outcome = Outcome::Success;
        return Outcome::Success;
    }

    Outcome solve(std::span<double> b_and_x, int nrhs) override {
        if (!L_ || nrhs <= 0 ||
            b_and_x.size() < static_cast<std::size_t>(n_) *
                                 static_cast<std::size_t>(nrhs)) {
            stats_.outcome = Outcome::InvalidInput;
            stats_.reason = "cholmod: solve without valid factor";
            return stats_.outcome;
        }

        cholmod_dense *B = cholmod_allocate_dense(
            static_cast<std::size_t>(n_), static_cast<std::size_t>(nrhs),
            static_cast<std::size_t>(n_), CHOLMOD_REAL, &c_);
        if (!B) {
            stats_.outcome = Outcome::OutOfMemory;
            stats_.reason = "cholmod_allocate_dense failed";
            return stats_.outcome;
        }
        std::memcpy(B->x, b_and_x.data(),
                    static_cast<std::size_t>(n_) * nrhs * sizeof(double));

        const auto t0 = Clock::now();
        cholmod_dense *X = cholmod_solve(CHOLMOD_A, L_, B, &c_);
        const auto t1 = Clock::now();
        stats_.solve_ms = ms_since(t0, t1);
        cholmod_free_dense(&B, &c_);

        if (!X || c_.status != CHOLMOD_OK) {
            if (X) cholmod_free_dense(&X, &c_);
            stats_.outcome = Outcome::Failed;
            stats_.reason = "cholmod_solve failed (status " +
                            std::to_string(c_.status) + ")";
            return stats_.outcome;
        }
        std::memcpy(b_and_x.data(), X->x,
                    static_cast<std::size_t>(n_) * nrhs * sizeof(double));
        cholmod_free_dense(&X, &c_);
        stats_.outcome = Outcome::Success;
        return Outcome::Success;
    }

    AdapterStats stats() const override { return stats_; }

private:
    cholmod_common c_;
    cholmod_sparse *A_ = nullptr;
    cholmod_factor *L_ = nullptr;
    Int n_ = 0;
    bool indefinite_ = false;
    AdapterStats stats_{};
};

#endif // SMF_HAS_CHOLMOD_IMPL

class UnavailableCholmodAdapter final : public SolverAdapter {
public:
    AdapterIdentity identity() const override {
        AdapterIdentity id;
        id.solver_id = "cholmod";
        id.version = "not-linked";
        id.library_path = "unavailable";
        id.integer_abi = "int32";
        id.available = false;
        id.supports_spd = true;
        id.supports_indefinite = false;
        return id;
    }
    void configure(const Control &, const std::string &) override {}
    Outcome analyse(const CscLower &) override { return Outcome::Unavailable; }
    Outcome factor(std::span<const double>) override { return Outcome::Unavailable; }
    Outcome solve(std::span<double>, int) override { return Outcome::Unavailable; }
    AdapterStats stats() const override {
        return AdapterStats{Outcome::Unavailable, "CHOLMOD not installed"};
    }
};

} // namespace

std::unique_ptr<SolverAdapter> make_cholmod_adapter() {
#if defined(SMF_HAS_CHOLMOD_IMPL)
    return std::make_unique<CholmodSolverAdapter>();
#else
    return std::make_unique<UnavailableCholmodAdapter>();
#endif
}

} // namespace smf::bench
