#include "adapter.hpp"
#include "smf/solver.hpp"
#include "smf/workspace.hpp"

#include <chrono>
#include <cmath>

namespace smf::bench {

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0, Clock::time_point t1) noexcept {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

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
