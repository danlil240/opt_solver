#include "adapter.hpp"
#include "smf/solver.hpp"
#include "smf/workspace.hpp"

#include <chrono>
#include <cmath>
#include <algorithm>

#if defined(SMF_HAS_MA27_IMPL) && SMF_HAS_MA27_IMPL
// MA27 double-precision Fortran interface (CoinHSL / libcoinhsl).
// gfortran ABI: all args by address, 32-bit ints, no hidden string lengths.
extern "C" {
void ma27id_(int* icntl, double* cntl);
void ma27ad_(int* n, int* nz, int* irn, int* icn, int* iw, int* liw,
             int* ikeep, int* iw1, int* nsteps, int* iflag, int* icntl,
             double* cntl, int* info, double* ops);
void ma27bd_(int* n, int* nz, int* irn, int* icn, double* a, int* la,
             int* iw, int* liw, int* ikeep, int* nsteps, int* maxfrt,
             int* iw1, int* icntl, double* cntl, int* info);
void ma27cd_(int* n, double* a, int* la, int* iw, int* liw, double* w,
             int* maxfrt, double* rhs, int* iw1, int* nsteps, int* icntl,
             int* info);
}
#endif

namespace smf::bench {

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0, Clock::time_point t1) noexcept {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

#if defined(SMF_HAS_MA27_IMPL) && SMF_HAS_MA27_IMPL
/// Real CoinHSL MA27 baseline adapter.
///
/// MA27 consumes 1-indexed COO triplets for the lower triangle, runs
/// ma27ad_ once per pattern in analyse() (its symbolic phase fills IKEEP /
/// the IW control array / NSTEPS), then ma27bd_ per numeric factorisation
/// (it overwrites A[] in place, so values are re-scattered into a_fac_
/// from the caller's span each call) and ma27cd_ per right-hand side.
///
/// Numerical qualification remains end-to-end: if the backend reports
/// success but produces a backward-error-failing solution, the harness
/// marks it Inaccurate rather than trusting factor success.
class Ma27Adapter final : public SolverAdapter {
public:
    AdapterIdentity identity() const override {
        AdapterIdentity id;
        id.solver_id = "ma27";
        id.version = "coinhsl-ma27";
        id.library_path = "libcoinhsl.dll";
        id.integer_abi = "int32";
        id.available = true;
        id.supports_spd = true;
        id.supports_indefinite = true;
        id.is_external_memory = false;
        return id;
    }

    void configure(const Control &ctrl, const std::string &track) override {
        ctrl_ = ctrl;
        track_ = track;
    }

    Outcome analyse(const CscLower &A) override {
        stats_ = AdapterStats{};
        analysed_ = false;
        factored_ = false;
        n_ = static_cast<int>(A.n);
        nz_ = static_cast<int>(A.nnz());

        // CSC lower -> 1-indexed COO, recording the CSC index of each COO
        // entry so factor() can scatter updated values in O(nnz).
        irn_.assign(static_cast<std::size_t>(nz_), 0);
        icn_.assign(static_cast<std::size_t>(nz_), 0);
        coo_src_.assign(static_cast<std::size_t>(nz_), 0);
        int p = 0;
        for (int j = 0; j < n_; ++j) {
            for (Int k = A.col_ptr[static_cast<std::size_t>(j)];
                 k < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++k, ++p) {
                irn_[static_cast<std::size_t>(p)] =
                    static_cast<int>(A.row_idx[static_cast<std::size_t>(k)]) + 1;
                icn_[static_cast<std::size_t>(p)] = j + 1;
                coo_src_[static_cast<std::size_t>(p)] = static_cast<int>(k);
            }
        }

        ma27id_(icntl_, cntl_);
        icntl_[0] = 0; // error output unit (0 = suppress)
        icntl_[1] = 0; // warning output unit (0 = suppress)
        // CNTL(1) = pivot threshold; 0 -> pure Cholesky on SPD input.
        cntl_[0] = (ctrl_.matrix_type ==
                    MatrixType::RealSymmetricPositiveDefinite)
                       ? 0.0
                       : ctrl_.pivot_u;

        liw_ = 2 * nz_ + 3 * n_ + 1;
        iw_.assign(static_cast<std::size_t>(liw_), 0);
        ikeep_.assign(static_cast<std::size_t>(3 * n_), 0);
        std::vector<int> iw1(static_cast<std::size_t>(2 * n_), 0);
        nsteps_ = 0;
        int iflag = 0;
        info_[0] = 0;
        double ops = 0.0;

        const auto t0 = Clock::now();
        ma27ad_(&n_, &nz_, irn_.data(), icn_.data(), iw_.data(), &liw_,
                ikeep_.data(), iw1.data(), &nsteps_, &iflag, icntl_, cntl_,
                info_, &ops);
        const auto t1 = Clock::now();
        stats_.analyse_ms = ms_since(t0, t1);
        stats_.flops = ops;

        if (info_[0] != 0) {
            stats_.outcome = Outcome::InvalidInput;
            stats_.reason = "ma27ad_ info(1)=" + std::to_string(info_[0]);
            return stats_.outcome;
        }

        // INFO(4)/INFO(5) report the minimum real/int workspace the numeric
        // phase needs; scale generously for pivoting fill (same heuristic
        // as bench_compare's proven call sequence).
        la_ = std::max(4 * nz_, static_cast<int>(3.0 * info_[3]))
              + 2 * n_ + 200;
        liw2_ = std::max(4 * nz_, static_cast<int>(3.0 * info_[4]))
                + 2 * n_ + 200;
        if (liw2_ > liw_) iw_.resize(static_cast<std::size_t>(liw2_));
        a_fac_.assign(static_cast<std::size_t>(la_), 0.0);
        // MA27BD uses IW1 as a work stack of up to ~2n+nsteps entries.
        iw1_.assign(static_cast<std::size_t>(2 * n_ + nsteps_ + 100), 0);

        analysed_ = true;
        stats_.outcome = Outcome::Success;
        return Outcome::Success;
    }

    Outcome factor(std::span<const double> values) override {
        if (!analysed_) {
            stats_.outcome = Outcome::InvalidInput;
            stats_.reason = "factor called before analyse";
            return stats_.outcome;
        }
        for (int p = 0; p < nz_; ++p) {
            a_fac_[static_cast<std::size_t>(p)] =
                values[static_cast<std::size_t>(
                    coo_src_[static_cast<std::size_t>(p)])];
        }

        int maxfrt = 0;
        const auto t0 = Clock::now();
        ma27bd_(&n_, &nz_, irn_.data(), icn_.data(), a_fac_.data(), &la_,
                iw_.data(), &liw2_, ikeep_.data(), &nsteps_, &maxfrt,
                iw1_.data(), icntl_, cntl_, info_);
        const auto t1 = Clock::now();
        stats_.factor_ms = ms_since(t0, t1);

        // MA27 info(1): 0 success; negative = argument/workspace errors;
        // positive warnings — notably 3 = matrix found singular. Verified
        // empirically against the CoinHSL 2024.5.15 MinGW build.
        if (info_[0] < 0) {
            stats_.outcome = Outcome::Failed;
            stats_.reason = "ma27bd_ info(1)=" + std::to_string(info_[0]);
            return stats_.outcome;
        }
        if (info_[0] == 3) {
            stats_.outcome = Outcome::Singular;
            stats_.reason = "ma27bd_ reported singular matrix";
            return stats_.outcome;
        }
        if (info_[0] != 0) {
            stats_.outcome = Outcome::Failed;
            stats_.reason = "ma27bd_ info(1)=" + std::to_string(info_[0]);
            return stats_.outcome;
        }
        maxfrt_ = maxfrt;
        factored_ = true;
        // Verified empirically: INFO(15) = #negative eigenvalues; on a
        // successful (nonsingular) factorisation pos = n - neg, zero = 0.
        stats_.inertia_neg = info_[14];
        stats_.inertia_pos = n_ - info_[14];
        stats_.inertia_zero = 0;
        stats_.factor_entries = info_[3];
        stats_.outcome = Outcome::Success;
        return Outcome::Success;
    }

    Outcome solve(std::span<double> b_and_x, int nrhs) override {
        if (!factored_) {
            stats_.outcome = Outcome::Failed;
            stats_.reason = "solve called without valid factor";
            return stats_.outcome;
        }
        if (nrhs <= 0 ||
            b_and_x.size() < static_cast<std::size_t>(n_) *
                             static_cast<std::size_t>(nrhs)) {
            stats_.outcome = Outcome::InvalidInput;
            stats_.reason = "solve buffer too small";
            return stats_.outcome;
        }
        std::vector<double> w(
            static_cast<std::size_t>(std::max(maxfrt_, 1)), 0.0);
        const auto t0 = Clock::now();
        for (int r = 0; r < nrhs; ++r) {
            ma27cd_(&n_, a_fac_.data(), &la_, iw_.data(), &liw2_, w.data(),
                    &maxfrt_, b_and_x.data() + static_cast<std::size_t>(r) * n_,
                    iw1_.data(), &nsteps_, icntl_, info_);
            if (info_[0] != 0) break;
        }
        const auto t1 = Clock::now();
        stats_.solve_ms = ms_since(t0, t1);
        if (info_[0] != 0) {
            stats_.outcome = Outcome::Failed;
            stats_.reason = "ma27cd_ info(1)=" + std::to_string(info_[0]);
            return stats_.outcome;
        }
        stats_.outcome = Outcome::Success;
        return Outcome::Success;
    }

    AdapterStats stats() const override { return stats_; }

private:
    Control ctrl_{};
    std::string track_ = "recommended";
    bool analysed_ = false;
    bool factored_ = false;
    int n_ = 0;
    int nz_ = 0;
    int nsteps_ = 0;
    int la_ = 0;
    int liw_ = 0;
    int liw2_ = 0;
    int maxfrt_ = 0;
    int icntl_[30] = {};
    double cntl_[5] = {};
    int info_[20] = {};
    std::vector<int> irn_, icn_, coo_src_;
    std::vector<int> iw_, ikeep_, iw1_;
    std::vector<double> a_fac_;
    AdapterStats stats_{};
};
#endif // SMF_HAS_MA27_IMPL

class SmfSolverAdapter final : public SolverAdapter {
public:
    AdapterIdentity identity() const override {
        AdapterIdentity id;
        id.solver_id = "smf";
        id.version = "0.1.0";
        id.library_path = "in-tree:libsmf";
        id.integer_abi = "int32_offsets64";
        id.available = true;
        id.supports_spd = true;
        id.supports_indefinite = true;
        id.is_external_memory = false;
        return id;
    }

    void configure(const Control &ctrl, const std::string &track) override {
        ctrl_ = ctrl;
        track_ = track;
        if (track_ == "recommended" &&
            ctrl_.matrix_type == MatrixType::RealSymmetricPositiveDefinite) {
            ctrl_.ordering = OrderingMethod::AutoParallel;
        }
    }

    Outcome analyse(const CscLower &A) override {
        stats_ = AdapterStats{};
        base_matrix_ = A;
        Info info;
        const auto t0 = Clock::now();
        ak_ = solver_.analyse(A, ctrl_, info);
        const auto t1 = Clock::now();
        stats_.analyse_ms = ms_since(t0, t1);
        if (!ak_ || info.status != ErrorCode::Success) {
            stats_.outcome = Outcome::InvalidInput;
            stats_.reason = "smf::Solver::analyse failed";
            return stats_.outcome;
        }
        stats_.outcome = Outcome::Success;
        return Outcome::Success;
    }

    Outcome factor(std::span<const double> values) override {
        if (!ak_) {
            stats_.outcome = Outcome::InvalidInput;
            stats_.reason = "factor called before analyse";
            return stats_.outcome;
        }
        if (values.size() != base_matrix_.values.size()) {
            stats_.outcome = Outcome::InvalidInput;
            stats_.reason = "values size mismatch";
            return stats_.outcome;
        }
        base_matrix_.values.assign(values.begin(), values.end());

        Info info;
        const auto t0 = Clock::now();
        const FactorStatus fs = solver_.factor(*ak_, base_matrix_, ctrl_, info, fk_, nws_);
        const auto t1 = Clock::now();
        stats_.factor_ms = ms_since(t0, t1);
        stats_.inertia_pos = info.num_positive;
        stats_.inertia_neg = info.num_negative;
        stats_.inertia_zero = info.num_zero;
        stats_.delayed_pivots = info.delayed_pivots;
        stats_.factor_entries = info.actual_factor_entries;
        stats_.flops = info.actual_flops;

        if (fs == FactorStatus::Success) {
            if (ctrl_.matrix_type == MatrixType::RealSymmetricPositiveDefinite) {
                stats_.inertia_pos = static_cast<int>(base_matrix_.n);
                stats_.inertia_neg = 0;
                stats_.inertia_zero = 0;
            }
            stats_.outcome = Outcome::Success;
            return Outcome::Success;
        }
        if (info.status == ErrorCode::AllocationError) {
            stats_.outcome = Outcome::OutOfMemory;
            stats_.reason = "allocation / storage overflow";
            return stats_.outcome;
        }
        if (info.status == ErrorCode::IllegalValue) {
            stats_.outcome = Outcome::InvalidInput;
            stats_.reason = "invalid input values";
            return stats_.outcome;
        }
        stats_.outcome = Outcome::Singular;
        stats_.reason = "singular or not positive definite";
        return stats_.outcome;
    }

    Outcome solve(std::span<double> b_and_x, int nrhs) override {
        if (!ak_ || fk_.status != FactorStatus::Success) {
            stats_.outcome = Outcome::Failed;
            stats_.reason = "solve called without valid factor";
            return stats_.outcome;
        }
        const int n = static_cast<int>(base_matrix_.n);
        if (nrhs <= 0 || b_and_x.size() < static_cast<std::size_t>(n) * static_cast<std::size_t>(nrhs)) {
            stats_.outcome = Outcome::InvalidInput;
            stats_.reason = "solve buffer too small";
            return stats_.outcome;
        }
        Info info;
        const auto t0 = Clock::now();
        const int rc = solver_.solve(fk_, ctrl_, info, b_and_x.data(), n, nrhs, sws_);
        const auto t1 = Clock::now();
        stats_.solve_ms = ms_since(t0, t1);
        if (rc != 0) {
            stats_.outcome = Outcome::Failed;
            stats_.reason = "smf::Solver::solve failed";
            return stats_.outcome;
        }
        stats_.outcome = Outcome::Success;
        return Outcome::Success;
    }

    AdapterStats stats() const override { return stats_; }

private:
    Control ctrl_{};
    std::string track_ = "recommended";
    Solver solver_;
    CscLower base_matrix_;
    std::unique_ptr<AnalysisKeep> ak_;
    FactorKeep fk_;
    NumericWorkspace nws_;
    SolveWorkspace sws_;
    AdapterStats stats_{};
};

class UnavailableOrExternalHslAdapter final : public SolverAdapter {
public:
    explicit UnavailableOrExternalHslAdapter(std::string hsl_id)
        : hsl_id_(std::move(hsl_id)) {}

    AdapterIdentity identity() const override {
        AdapterIdentity id;
        id.solver_id = hsl_id_;
        id.version = "not-linked";
        id.library_path = "unavailable";
        id.integer_abi = "int32";
        id.available = false;
        id.supports_spd = true;
        id.supports_indefinite = (hsl_id_ != "ma87");
        id.is_external_memory = (hsl_id_ == "ma77");
        return id;
    }

    void configure(const Control &, const std::string &) override {}

    Outcome analyse(const CscLower &) override {
        stats_.outcome = Outcome::Unavailable;
        stats_.reason = "Genuine HSL " + hsl_id_ + " library is not installed";
        return Outcome::Unavailable;
    }

    Outcome factor(std::span<const double>) override {
        stats_.outcome = Outcome::Unavailable;
        stats_.reason = "Genuine HSL " + hsl_id_ + " library is not installed";
        return Outcome::Unavailable;
    }

    Outcome solve(std::span<double>, int) override {
        stats_.outcome = Outcome::Unavailable;
        stats_.reason = "Genuine HSL " + hsl_id_ + " library is not installed";
        return Outcome::Unavailable;
    }

    AdapterStats stats() const override { return stats_; }

private:
    std::string hsl_id_;
    AdapterStats stats_{Outcome::Unavailable, "not installed"};
};

} // namespace

std::unique_ptr<SolverAdapter> make_smf_adapter() {
    return std::make_unique<SmfSolverAdapter>();
}

std::unique_ptr<SolverAdapter> make_hsl_adapter(const std::string &hsl_id) {
#if defined(SMF_HAS_MA27_IMPL) && SMF_HAS_MA27_IMPL
    if (hsl_id == "ma27") return std::make_unique<Ma27Adapter>();
#endif
    return std::make_unique<UnavailableOrExternalHslAdapter>(hsl_id);
}

std::unique_ptr<SolverAdapter> make_solver_adapter(const std::string &solver_id) {
    if (solver_id == "smf") return make_smf_adapter();
    if (solver_id == "ma27" || solver_id == "ma57" || solver_id == "ma77" ||
        solver_id == "ma86" || solver_id == "ma87" || solver_id == "ma97") {
        return make_hsl_adapter(solver_id);
    }
    if (solver_id == "ssids") return make_spral_ssids_adapter();
    if (solver_id == "mumps") return make_mumps_adapter();
    if (solver_id == "mkl_pardiso" || solver_id == "panua_pardiso") {
        return make_pardiso_adapter(solver_id);
    }
    if (solver_id == "cholmod") return make_cholmod_adapter();
    return nullptr;
}

std::vector<std::string> all_registered_adapter_ids() {
    return {
        "smf",
        "ma27",
        "ma57",
        "ma77",
        "ma86",
        "ma87",
        "ma97",
        "ssids",
        "mumps",
        "mkl_pardiso",
        "panua_pardiso",
        "cholmod"
    };
}

} // namespace smf::bench
