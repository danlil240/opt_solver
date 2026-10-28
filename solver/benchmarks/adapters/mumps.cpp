#include "adapter.hpp"

#if defined(SMF_HAS_MUMPS_IMPL)
#include <dmumps_c.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstring>
#include <vector>

namespace smf::bench {

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0, Clock::time_point t1) noexcept {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

#if defined(SMF_HAS_MUMPS_IMPL)

// ICNTL(i) maps to icntl[i-1]; INFOG(i) maps to infog[i-1].
Outcome map_mumps_infog(MUMPS_INT infog1, MUMPS_INT infog2) {
    switch (infog1) {
        case 0:
            return Outcome::Success;
        case -6:
        case -10:
            return Outcome::Singular;
        case -8:
        case -9:
        case -19:
        case -20:
        case -21:
        case -22:
        case -23:
        case -24:
        case -26:
        case -27:
            return Outcome::OutOfMemory;
        case -1:
        case -2:
        case -3:
        case -11:
        case -16:
        case -51:
        case -52:
        case -53:
            return Outcome::InvalidInput;
        default:
            return (infog1 < 0 || infog2 < 0) ? Outcome::Failed
                                            : Outcome::Success;
    }
}

class MumpsSolverAdapter final : public SolverAdapter {
public:
    ~MumpsSolverAdapter() override { terminate(); }

    AdapterIdentity identity() const override {
        AdapterIdentity id;
        id.solver_id = "mumps";
        id.version = MUMPS_VERSION;
        id.library_path = "dmumps";
        id.integer_abi = "int32";
        id.available = true;
        id.supports_spd = true;
        id.supports_indefinite = true;
        return id;
    }

    void configure(const Control &ctrl, const std::string &track) override {
        indefinite_ =
            (ctrl.matrix_type == MatrixType::RealSymmetricIndefinite);
        track_ = track;
    }

    Outcome analyse(const CscLower &A) override {
        terminate();
        stats_ = AdapterStats{};
        n_ = A.n;
        const std::size_t nz = A.values.size();
        if (n_ <= 0 || A.col_ptr.size() != static_cast<std::size_t>(n_) + 1 ||
            A.row_idx.size() != nz) {
            stats_.outcome = Outcome::InvalidInput;
            stats_.reason = "mumps: invalid CSC shape";
            return stats_.outcome;
        }

        // Expand lower-triangular CSC into 1-based COO (irn/jcn).
        irn_.clear();
        jcn_.clear();
        irn_.reserve(nz);
        jcn_.reserve(nz);
        for (Int j = 0; j < n_; ++j) {
            for (Int p = A.col_ptr[static_cast<std::size_t>(j)];
                 p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
                irn_.push_back(
                    static_cast<MUMPS_INT>(A.row_idx[static_cast<std::size_t>(p)]) + 1);
                jcn_.push_back(static_cast<MUMPS_INT>(j) + 1);
            }
        }
        a_.assign(A.values.begin(), A.values.end());

        std::memset(&id_, 0, sizeof(id_));
        id_.comm_fortran = -9876;  // USE_COMM_WORLD for the sequential build
        id_.par = 1;
        id_.sym = indefinite_ ? 2 : 1;
        id_.job = -1;
        dmumps_c(&id_);
        if (id_.infog[0] < 0) {
            stats_.outcome = map_mumps_infog(id_.infog[0], id_.infog[1]);
            stats_.reason = "mumps init error INFOG(1)=" +
                            std::to_string(id_.infog[0]);
            initialized_ = false;
            return stats_.outcome;
        }
        initialized_ = true;

        // Silence diagnostic/error output streams.
        id_.icntl[0] = -1;  // ICNTL(1)
        id_.icntl[1] = -1;  // ICNTL(2)
        id_.icntl[2] = -1;  // ICNTL(3)
        id_.icntl[3] = 0;   // ICNTL(4): no messages
        id_.icntl[23] = 1;  // ICNTL(24): detect null pivots -> INFOG(28)
        // IPOPT's MUMPS interface default mumps_pivtol (CNTL(1)).
        if (track_ == "ipopt") id_.cntl[0] = 1e-6;

        id_.n = static_cast<MUMPS_INT>(n_);
        id_.nz = static_cast<MUMPS_INT>(nz);
        id_.irn = irn_.data();
        id_.jcn = jcn_.data();
        id_.a = a_.data();

        id_.job = 1;
        const auto t0 = Clock::now();
        dmumps_c(&id_);
        const auto t1 = Clock::now();
        stats_.analyse_ms = ms_since(t0, t1);

        stats_.outcome = map_mumps_infog(id_.infog[0], id_.infog[1]);
        if (id_.infog[0] < 0) {
            stats_.reason = "mumps analyse error INFOG(1)=" +
                            std::to_string(id_.infog[0]);
        }
        return stats_.outcome;
    }

    Outcome factor(std::span<const double> values) override {
        if (!initialized_ || values.size() != a_.size()) {
            stats_.outcome = Outcome::InvalidInput;
            stats_.reason = "mumps: factor before analyse or size mismatch";
            return stats_.outcome;
        }
        a_.assign(values.begin(), values.end());
        id_.a = a_.data();

        // INFOG(1) = -8/-9 means the analysis' workspace estimate was too
        // small (delayed pivots); the documented remedy, which IPOPT applies
        // too, is to raise ICNTL(14) (% extra workspace) and refactor.
        id_.job = 2;
        const auto t0 = Clock::now();
        dmumps_c(&id_);
        for (int retry = 0; retry < 4 && (id_.infog[0] == -8 || id_.infog[0] == -9); ++retry) {
            id_.icntl[13] = std::max<MUMPS_INT>(2 * id_.icntl[13], 100);  // ICNTL(14)
            dmumps_c(&id_);
        }
        const auto t1 = Clock::now();
        stats_.factor_ms = ms_since(t0, t1);

        stats_.outcome = map_mumps_infog(id_.infog[0], id_.infog[1]);
        if (stats_.outcome == Outcome::Success) {
            const MUMPS_INT neg = id_.infog[11];   // INFOG(12)
            const MUMPS_INT nul = id_.infog[27];   // INFOG(28)
            stats_.inertia_neg = static_cast<int>(neg < 0 ? 0 : neg);
            stats_.inertia_zero = static_cast<int>(nul < 0 ? 0 : nul);
            stats_.inertia_pos = static_cast<int>(n_) -
                                 stats_.inertia_neg - stats_.inertia_zero;
            if (stats_.inertia_pos < 0) stats_.inertia_pos = 0;
            stats_.factor_entries = static_cast<int64_t>(id_.infog[8]);
            if (indefinite_ && stats_.inertia_zero > 0) {
                stats_.outcome = Outcome::Singular;
                stats_.reason = "mumps: " +
                                std::to_string(stats_.inertia_zero) +
                                " null pivots detected";
            }
        } else {
            stats_.reason = "mumps factor error INFOG(1)=" +
                            std::to_string(id_.infog[0]) + " INFOG(2)=" +
                            std::to_string(id_.infog[1]);
        }
        return stats_.outcome;
    }

    Outcome solve(std::span<double> b_and_x, int nrhs) override {
        if (!initialized_ || nrhs <= 0 ||
            b_and_x.size() < static_cast<std::size_t>(n_) *
                                 static_cast<std::size_t>(nrhs)) {
            stats_.outcome = Outcome::InvalidInput;
            stats_.reason = "mumps: solve without valid factor";
            return stats_.outcome;
        }

        id_.nrhs = static_cast<MUMPS_INT>(nrhs);
        id_.lrhs = static_cast<MUMPS_INT>(n_);
        id_.rhs = b_and_x.data();

        id_.job = 3;
        const auto t0 = Clock::now();
        dmumps_c(&id_);
        const auto t1 = Clock::now();
        stats_.solve_ms = ms_since(t0, t1);

        stats_.outcome = map_mumps_infog(id_.infog[0], id_.infog[1]);
        if (stats_.outcome != Outcome::Success) {
            stats_.reason = "mumps solve error INFOG(1)=" +
                            std::to_string(id_.infog[0]);
        }
        return stats_.outcome;
    }

    AdapterStats stats() const override { return stats_; }

private:
    void terminate() noexcept {
        if (initialized_) {
            id_.job = -2;
            dmumps_c(&id_);
            initialized_ = false;
        }
        n_ = 0;
    }

    DMUMPS_STRUC_C id_{};
    bool initialized_ = false;
    bool indefinite_ = true;
    std::string track_ = "recommended";
    Int n_ = 0;
    std::vector<MUMPS_INT> irn_, jcn_;
    std::vector<double> a_;
    AdapterStats stats_{};
};

#endif // SMF_HAS_MUMPS_IMPL

class UnavailableMumpsAdapter final : public SolverAdapter {
public:
    AdapterIdentity identity() const override {
        AdapterIdentity id;
        id.solver_id = "mumps";
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
        return AdapterStats{Outcome::Unavailable, "MUMPS not installed"};
    }
};

} // namespace

std::unique_ptr<SolverAdapter> make_mumps_adapter() {
#if defined(SMF_HAS_MUMPS_IMPL)
    return std::make_unique<MumpsSolverAdapter>();
#else
    return std::make_unique<UnavailableMumpsAdapter>();
#endif
}

} // namespace smf::bench
