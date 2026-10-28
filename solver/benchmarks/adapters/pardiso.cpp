#include "adapter.hpp"

#if defined(SMF_HAS_MKL_PARDISO)
#include <mkl.h>
#include <mkl_pardiso.h>
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

#if defined(SMF_HAS_MKL_PARDISO)

Outcome map_pardiso_error(MKL_INT error, bool is_factor_phase) {
    switch (error) {
        case 0:
            return Outcome::Success;
        case -1:
            return Outcome::InvalidInput;
        case -2:
        case -9:
        case -10:
            return Outcome::OutOfMemory;
        case -4:
            return Outcome::Singular;
        default:
            return is_factor_phase ? Outcome::Singular : Outcome::Failed;
    }
}

// MKL PARDISO baseline adapter. Note on singularity semantics: MKL's symmetric
// indefinite path (mtype=-2) performs Bunch-Kaufman factorization with pivot
// perturbation as a non-optional internal step and does not expose a reliable
// zero-pivot count through iparm, so exact singularity cannot be reported at
// factor time. A regularized factor+solve is caught downstream by the shared
// backward-error qualification (Inaccurate), which is the spec's mechanism for
// "no falsely successful singular solve".
class MklPardisoSolverAdapter final : public SolverAdapter {
public:
    MklPardisoSolverAdapter() {
        std::memset(pt_, 0, sizeof(pt_));
        std::memset(iparm_, 0, sizeof(iparm_));
    }

    ~MklPardisoSolverAdapter() override { release_handle(); }

    AdapterIdentity identity() const override {
        AdapterIdentity id;
        id.solver_id = "mkl_pardiso";
        char ver_buf[256] = {0};
        MKL_Get_Version_String(ver_buf, sizeof(ver_buf));
        id.version = ver_buf;
        id.library_path = "mkl_rt";
        id.integer_abi = "int32_lp64";
        id.available = true;
        id.supports_spd = true;
        id.supports_indefinite = true;
        return id;
    }

    void configure(const Control &ctrl, const std::string &track) override {
        indefinite_ = (ctrl.matrix_type == MatrixType::RealSymmetricIndefinite);
        track_ = track;
    }

    // iparm is session state: it is only (re)built when a new factor session is
    // created in analyse(), never reconfigured mid-session, so release_handle()
    // always sees the settings the handle was created with.
    void build_iparm() {
        std::memset(iparm_, 0, sizeof(iparm_));
        iparm_[0] = 1;   // use non-default iparm values
        iparm_[1] = 2;   // METIS nested dissection ordering
        iparm_[5] = 0;   // write solution to x (not b)
        iparm_[23] = 1;  // parallel numerical factorization
        iparm_[26] = (track_ == "common") ? 1 : 0;
        if (indefinite_) {
            iparm_[9] = 13;   // perturb pivots below 10^-13 (recommended for indef)
            iparm_[10] = 1;   // scaling vectors
            iparm_[12] = 1;   // weighted matching
        }
        iparm_[34] = 1;  // 0-based indexing
    }

    Outcome analyse(const CscLower &A) override {
        release_handle();
        stats_ = AdapterStats{};
        build_iparm();
        session_mtype_ = indefinite_ ? -2 : 2;
        n_ = static_cast<MKL_INT>(A.n);
        if (n_ <= 0 || A.col_ptr.size() != static_cast<std::size_t>(n_) + 1) {
            stats_.outcome = Outcome::InvalidInput;
            stats_.reason = "pardiso: invalid CSC shape";
            return stats_.outcome;
        }

        ia_.assign(A.col_ptr.begin(), A.col_ptr.end());
        ja_.assign(A.row_idx.begin(), A.row_idx.end());
        a_.assign(A.values.begin(), A.values.end());
        perm_.assign(static_cast<std::size_t>(n_), 0);
        x_.assign(static_cast<std::size_t>(n_), 0.0);

        MKL_INT phase = 11, error = 0, maxfct = 1, mnum = 1, nrhs = 1;
        const auto t0 = Clock::now();
        pardiso(pt_, &maxfct, &mnum, &session_mtype_, &phase, &n_,
                a_.data(), ia_.data(), ja_.data(), perm_.data(), &nrhs,
                iparm_, &msglvl_, nullptr, nullptr, &error);
        const auto t1 = Clock::now();
        stats_.analyse_ms = ms_since(t0, t1);

        stats_.outcome = map_pardiso_error(error, false);
        if (error != 0) {
            stats_.reason = "pardiso phase 11 error " + std::to_string(error);
        }
        return stats_.outcome;
    }

    Outcome factor(std::span<const double> values) override {
        if (n_ <= 0 || values.size() != a_.size()) {
            stats_.outcome = Outcome::InvalidInput;
            stats_.reason = "pardiso: factor before analyse or values size mismatch";
            return stats_.outcome;
        }
        a_.assign(values.begin(), values.end());

        MKL_INT phase = 22, error = 0, maxfct = 1, mnum = 1, nrhs = 1;
        const auto t0 = Clock::now();
        pardiso(pt_, &maxfct, &mnum, &session_mtype_, &phase, &n_, a_.data(),
                ia_.data(), ja_.data(), perm_.data(), &nrhs, iparm_,
                &msglvl_, nullptr, nullptr, &error);
        const auto t1 = Clock::now();
        stats_.factor_ms = ms_since(t0, t1);

        if (indefinite_ && error == 0) {
            stats_.inertia_pos = static_cast<int>(iparm_[21]);
            stats_.inertia_neg = static_cast<int>(iparm_[22]);
            stats_.inertia_zero =
                static_cast<int>(n_) - stats_.inertia_pos - stats_.inertia_neg;
            if (stats_.inertia_zero < 0) stats_.inertia_zero = 0;
        } else if (!indefinite_) {
            stats_.inertia_pos = (error == 0) ? static_cast<int>(n_) : 0;
            stats_.inertia_neg = 0;
            stats_.inertia_zero = 0;
        }

        stats_.outcome = map_pardiso_error(error, true);
        if (error != 0) {
            stats_.reason = "pardiso phase 22 error " + std::to_string(error);
        }
        return stats_.outcome;
    }

    Outcome solve(std::span<double> b_and_x, int nrhs) override {
        if (n_ <= 0 || nrhs <= 0 ||
            b_and_x.size() < static_cast<std::size_t>(n_) *
                                 static_cast<std::size_t>(nrhs)) {
            stats_.outcome = Outcome::InvalidInput;
            stats_.reason = "pardiso: solve buffer too small";
            return stats_.outcome;
        }
        x_.assign(static_cast<std::size_t>(n_) *
                      static_cast<std::size_t>(nrhs),
                  0.0);

        MKL_INT phase = 33, error = 0, maxfct = 1, mnum = 1;
        MKL_INT nrhs_k = static_cast<MKL_INT>(nrhs);
        const auto t0 = Clock::now();
        pardiso(pt_, &maxfct, &mnum, &session_mtype_, &phase, &n_, a_.data(),
                ia_.data(), ja_.data(), perm_.data(), &nrhs_k, iparm_,
                &msglvl_, b_and_x.data(), x_.data(), &error);
        const auto t1 = Clock::now();
        stats_.solve_ms = ms_since(t0, t1);

        if (error == 0) {
            std::memcpy(b_and_x.data(), x_.data(),
                        static_cast<std::size_t>(n_) *
                            static_cast<std::size_t>(nrhs) * sizeof(double));
        }
        stats_.outcome = map_pardiso_error(error, false);
        if (error != 0) {
            stats_.reason = "pardiso phase 33 error " + std::to_string(error);
        }
        return stats_.outcome;
    }

    AdapterStats stats() const override { return stats_; }

private:
    void release_handle() noexcept {
        if (n_ > 0) {
            MKL_INT phase = -1, error = 0, maxfct = 1, mnum = 1, nrhs = 1;
            pardiso(pt_, &maxfct, &mnum, &session_mtype_, &phase, &n_, a_.data(),
                    ia_.data(), ja_.data(), perm_.data(), &nrhs, iparm_,
                    &msglvl_, nullptr, nullptr, &error);
        }
        std::memset(pt_, 0, sizeof(pt_));
        n_ = 0;
    }

    void *pt_[64];
    MKL_INT iparm_[64];
    MKL_INT session_mtype_ = -2;
    MKL_INT msglvl_ = 0;
    MKL_INT n_ = 0;
    bool indefinite_ = true;
    std::string track_ = "recommended";
    std::vector<MKL_INT> ia_, ja_, perm_;
    std::vector<double> a_, x_;
    AdapterStats stats_{};
};

#endif // SMF_HAS_MKL_PARDISO

class UnavailablePardisoAdapter final : public SolverAdapter {
public:
    explicit UnavailablePardisoAdapter(std::string variant)
        : variant_(std::move(variant)) {}

    AdapterIdentity identity() const override {
        AdapterIdentity id;
        id.solver_id = variant_;
        id.version = "not-linked";
        id.library_path = "unavailable";
        id.integer_abi = "int32";
        id.available = false;
        id.supports_spd = true;
        id.supports_indefinite = true;
        return id;
    }
    void configure(const Control &, const std::string &) override {}
    Outcome analyse(const CscLower &) override { return Outcome::Unavailable; }
    Outcome factor(std::span<const double>) override { return Outcome::Unavailable; }
    Outcome solve(std::span<double>, int) override { return Outcome::Unavailable; }
    AdapterStats stats() const override {
        return AdapterStats{Outcome::Unavailable, variant_ + " not installed"};
    }

private:
    std::string variant_;
};

} // namespace

std::unique_ptr<SolverAdapter> make_pardiso_adapter(const std::string &variant) {
#if defined(SMF_HAS_MKL_PARDISO)
    if (variant == "mkl_pardiso") {
        return std::make_unique<MklPardisoSolverAdapter>();
    }
#endif
    return std::make_unique<UnavailablePardisoAdapter>(variant);
}

} // namespace smf::bench
