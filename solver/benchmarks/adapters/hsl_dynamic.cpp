// Runtime-loaded HSL MA97 and MA57 baselines.
//
// The CoinHSL "archive" distribution only ships MA27/MA28; MA57, MA86 and
// MA97 come with the licensed full CoinHSL (free for academic use, by
// request from STFC). These adapters load whatever library
// SMF_HSL_LIBRARY names (e.g. a full CoinHSL libcoinhsl.dll / libhsl.so)
// and report Unavailable with that instruction when it is unset or lacks
// the symbols, so licensed baselines switch on without a rebuild.
//
// Thread count: MA97 parallelizes with the HSL library's OpenMP runtime and
// MA57 only through BLAS; both follow OMP_NUM_THREADS / OPENBLAS_NUM_THREADS
// set by tools/bench/run.py.

#include "adapter.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace smf::bench {

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0, Clock::time_point t1) noexcept {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

/// Library named by SMF_HSL_LIBRARY, loaded once per process.
class HslLibrary {
public:
    static HslLibrary &get() {
        static HslLibrary lib;
        return lib;
    }
    const std::string &path() const { return path_; }
    void *symbol(const char *name) const {
        if (handle_ == nullptr) return nullptr;
#if defined(_WIN32)
        return reinterpret_cast<void *>(GetProcAddress(static_cast<HMODULE>(handle_), name));
#else
        return dlsym(handle_, name);
#endif
    }

private:
    HslLibrary() {
        const char *p = std::getenv("SMF_HSL_LIBRARY");
        if (p == nullptr || *p == '\0') return;
        path_ = p;
#if defined(_WIN32)
        // Let the library's own directory satisfy its dependencies
        // (libgfortran, libgomp, BLAS).
        handle_ = LoadLibraryExA(p, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
#else
        handle_ = dlopen(p, RTLD_NOW | RTLD_LOCAL);
#endif
    }
    void *handle_ = nullptr;
    std::string path_;
};

std::string hsl_unavailable_reason(const char *routine) {
    const auto &lib = HslLibrary::get();
    if (lib.path().empty()) {
        return std::string("set SMF_HSL_LIBRARY to a licensed HSL/CoinHSL library exporting ") +
               routine + " (the CoinHSL archive build only has MA27/MA28)";
    }
    return "SMF_HSL_LIBRARY=" + lib.path() + " does not export " + routine;
}

// ---------------------------------------------------------------- MA97 --
// C interface of HSL_MA97 (hsl_ma97d.h). `long` members are 32-bit on
// Windows, matching an LLP64 build of the library.
struct Ma97Control {
    int f_arrays;
    int action;
    int nemin;
    double multiplier;
    int ordering;
    int print_level;
    int scaling;
    double small_value; // HSL name "small" collides with a windows.h macro
    double u;
    int unit_diagnostics;
    int unit_error;
    int unit_warning;
    long factor_min;
    int solve_blas3;
    long solve_min;
    int solve_mf;
    double consist_tol;
    int ispare[5];
    double rspare[10];
};

struct Ma97Info {
    int flag;
    int flag68;
    int flag77;
    int matrix_dup;
    int matrix_rank;
    int matrix_outrange;
    int matrix_missing_diag;
    int maxdepth;
    int maxfront;
    int num_delay;
    long num_factor;
    long num_flops;
    int num_neg;
    int num_sup;
    int num_two;
    int ordering;
    int stat;
    int maxsupernode;
    int ispare[4];
    double rspare[10];
};

using Ma97DefaultControl = void (*)(Ma97Control *);
using Ma97Analyse = void (*)(int, int, const int *, const int *, double *, void **,
                             const Ma97Control *, Ma97Info *, int *);
using Ma97Factor = void (*)(int, const int *, const int *, const double *, void **, void **,
                            const Ma97Control *, Ma97Info *, double *);
using Ma97Solve = void (*)(int, int, double *, int, void **, void **, const Ma97Control *,
                           Ma97Info *);
using Ma97Finalise = void (*)(void **, void **);

class Ma97Adapter final : public SolverAdapter {
public:
    Ma97Adapter() {
        const auto &lib = HslLibrary::get();
        default_control_ = reinterpret_cast<Ma97DefaultControl>(lib.symbol("ma97_default_control_d"));
        analyse_ = reinterpret_cast<Ma97Analyse>(lib.symbol("ma97_analyse_d"));
        factor_ = reinterpret_cast<Ma97Factor>(lib.symbol("ma97_factor_d"));
        solve_ = reinterpret_cast<Ma97Solve>(lib.symbol("ma97_solve_d"));
        finalise_ = reinterpret_cast<Ma97Finalise>(lib.symbol("ma97_finalise_d"));
        available_ = default_control_ && analyse_ && factor_ && solve_ && finalise_;
        if (!available_) unavailable();
    }
    ~Ma97Adapter() override { release(); }

    AdapterIdentity identity() const override {
        AdapterIdentity id;
        id.solver_id = "ma97";
        id.version = available_ ? "hsl_ma97 (runtime-loaded)" : "not-linked";
        id.library_path = available_ ? HslLibrary::get().path() : "unavailable";
        id.integer_abi = "int32";
        id.available = available_;
        id.supports_spd = true;
        id.supports_indefinite = true;
        return id;
    }

    void configure(const Control &ctrl, const std::string &) override {
        matrix_type_ = (ctrl.matrix_type == MatrixType::RealSymmetricPositiveDefinite) ? 3 : 4;
        if (available_) {
            default_control_(&control_);
            control_.f_arrays = 0;
            control_.print_level = -1;
            control_.u = ctrl.pivot_u;
        }
    }

    Outcome analyse(const CscLower &A) override {
        stats_ = AdapterStats{};
        if (!available_) return unavailable();
        release();
        n_ = static_cast<int>(A.n);
        ptr_.assign(A.col_ptr.begin(), A.col_ptr.end());
        row_.assign(A.row_idx.begin(), A.row_idx.end());
        Ma97Info info{};
        const auto t0 = Clock::now();
        analyse_(0, n_, ptr_.data(), row_.data(), nullptr, &akeep_, &control_, &info, nullptr);
        stats_.analyse_ms = ms_since(t0, Clock::now());
        if (info.flag < 0) return fail(Outcome::Failed, "ma97_analyse_d flag=" + std::to_string(info.flag));
        stats_.outcome = Outcome::Success;
        return Outcome::Success;
    }

    Outcome factor(std::span<const double> values) override {
        if (!available_) return unavailable();
        if (akeep_ == nullptr) return fail(Outcome::InvalidInput, "factor called before analyse");
        Ma97Info info{};
        const auto t0 = Clock::now();
        factor_(matrix_type_, ptr_.data(), row_.data(), values.data(), &akeep_, &fkeep_, &control_,
                &info, nullptr);
        stats_.factor_ms = ms_since(t0, Clock::now());
        if (info.flag < 0) {
            return fail(info.flag == -6 ? Outcome::Singular : Outcome::Failed,
                        "ma97_factor_d flag=" + std::to_string(info.flag));
        }
        factored_ = true;
        stats_.inertia_neg = info.num_neg;
        stats_.inertia_zero = n_ - info.matrix_rank;
        stats_.inertia_pos = n_ - info.num_neg - stats_.inertia_zero;
        stats_.delayed_pivots = info.num_delay;
        stats_.factor_entries = info.num_factor;
        stats_.flops = static_cast<double>(info.num_flops);
        stats_.outcome = Outcome::Success;
        return Outcome::Success;
    }

    Outcome solve(std::span<double> b_and_x, int nrhs) override {
        if (!available_) return unavailable();
        if (!factored_) return fail(Outcome::Failed, "solve called without valid factor");
        Ma97Info info{};
        const auto t0 = Clock::now();
        solve_(0, nrhs, b_and_x.data(), n_, &akeep_, &fkeep_, &control_, &info);
        stats_.solve_ms = ms_since(t0, Clock::now());
        if (info.flag < 0) return fail(Outcome::Failed, "ma97_solve_d flag=" + std::to_string(info.flag));
        stats_.outcome = Outcome::Success;
        return Outcome::Success;
    }

    AdapterStats stats() const override { return stats_; }

private:
    Outcome unavailable() {
        stats_.outcome = Outcome::Unavailable;
        stats_.reason = hsl_unavailable_reason("ma97_*_d");
        return stats_.outcome;
    }
    Outcome fail(Outcome o, std::string reason) {
        stats_.outcome = o;
        stats_.reason = std::move(reason);
        return o;
    }
    void release() {
        if (finalise_ && (akeep_ || fkeep_)) finalise_(&akeep_, &fkeep_);
        akeep_ = nullptr;
        fkeep_ = nullptr;
        factored_ = false;
    }

    Ma97DefaultControl default_control_ = nullptr;
    Ma97Analyse analyse_ = nullptr;
    Ma97Factor factor_ = nullptr;
    Ma97Solve solve_ = nullptr;
    Ma97Finalise finalise_ = nullptr;
    bool available_ = false;
    bool factored_ = false;
    Ma97Control control_{};
    int matrix_type_ = 4;
    int n_ = 0;
    std::vector<int> ptr_, row_;
    void *akeep_ = nullptr;
    void *fkeep_ = nullptr;
    AdapterStats stats_{};
};

// ---------------------------------------------------------------- MA57 --
// Fortran 77 interface (gfortran ABI: everything by address).
using Ma57Id = void (*)(double *, int *);
using Ma57Ad = void (*)(int *, int *, int *, int *, int *, int *, int *, int *, int *, double *);
using Ma57Bd = void (*)(int *, int *, double *, double *, int *, int *, int *, int *, int *, int *,
                        int *, double *, int *, double *);
using Ma57Cd = void (*)(int *, int *, double *, int *, int *, int *, int *, double *, int *,
                        double *, int *, int *, int *, int *);

class Ma57Adapter final : public SolverAdapter {
public:
    Ma57Adapter() {
        const auto &lib = HslLibrary::get();
        id_ = reinterpret_cast<Ma57Id>(lib.symbol("ma57id_"));
        ad_ = reinterpret_cast<Ma57Ad>(lib.symbol("ma57ad_"));
        bd_ = reinterpret_cast<Ma57Bd>(lib.symbol("ma57bd_"));
        cd_ = reinterpret_cast<Ma57Cd>(lib.symbol("ma57cd_"));
        available_ = id_ && ad_ && bd_ && cd_;
        if (!available_) unavailable();
    }

    AdapterIdentity identity() const override {
        AdapterIdentity id;
        id.solver_id = "ma57";
        id.version = available_ ? "ma57 (runtime-loaded)" : "not-linked";
        id.library_path = available_ ? HslLibrary::get().path() : "unavailable";
        id.integer_abi = "int32";
        id.available = available_;
        id.supports_spd = true;
        id.supports_indefinite = true;
        return id;
    }

    void configure(const Control &ctrl, const std::string &) override {
        if (!available_) return;
        id_(cntl_, icntl_);
        // Silence error/warning/monitor streams and diagnostics; keep the
        // library's default ordering choice and scaling (what users get).
        icntl_[0] = -1;
        icntl_[1] = -1;
        icntl_[2] = -1;
        icntl_[3] = -1;
        icntl_[4] = 0;
        cntl_[0] = (ctrl.matrix_type == MatrixType::RealSymmetricPositiveDefinite) ? 0.0
                                                                                    : ctrl.pivot_u;
    }

    Outcome analyse(const CscLower &A) override {
        stats_ = AdapterStats{};
        factored_ = false;
        if (!available_) return unavailable();
        n_ = static_cast<int>(A.n);
        ne_ = static_cast<int>(A.nnz());
        irn_.resize(static_cast<std::size_t>(ne_));
        jcn_.resize(static_cast<std::size_t>(ne_));
        for (int j = 0, p = 0; j < n_; ++j) {
            for (Int k = A.col_ptr[static_cast<std::size_t>(j)]; k < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++k, ++p) {
                irn_[static_cast<std::size_t>(p)] = static_cast<int>(A.row_idx[static_cast<std::size_t>(k)]) + 1;
                jcn_[static_cast<std::size_t>(p)] = j + 1;
            }
        }
        lkeep_ = 5 * n_ + ne_ + std::max(n_, ne_) + 42;
        keep_.assign(static_cast<std::size_t>(lkeep_), 0);
        iwork_.assign(static_cast<std::size_t>(5 * n_), 0);
        const auto t0 = Clock::now();
        ad_(&n_, &ne_, irn_.data(), jcn_.data(), &lkeep_, keep_.data(), iwork_.data(), icntl_,
            info_, rinfo_);
        stats_.analyse_ms = ms_since(t0, Clock::now());
        if (info_[0] < 0) return fail(Outcome::Failed, "ma57ad_ info(1)=" + std::to_string(info_[0]));
        // INFO(9)/INFO(10): forecast lengths of FACT / IFACT; headroom for delays.
        lfact_ = std::max(2 * info_[8], ne_ + 1);
        lifact_ = std::max(2 * info_[9], ne_ + 1);
        fact_.assign(static_cast<std::size_t>(lfact_), 0.0);
        ifact_.assign(static_cast<std::size_t>(lifact_), 0);
        stats_.outcome = Outcome::Success;
        return Outcome::Success;
    }

    Outcome factor(std::span<const double> values) override {
        if (!available_) return unavailable();
        std::vector<double> a(values.begin(), values.end());
        const auto t0 = Clock::now();
        for (int attempt = 0; attempt < 16; ++attempt) {
            bd_(&n_, &ne_, a.data(), fact_.data(), &lfact_, ifact_.data(), &lifact_, &lkeep_,
                keep_.data(), iwork_.data(), icntl_, cntl_, info_, rinfo_);
            // -3 / -4: FACT / IFACT too small; INFO(17)/INFO(18) suggest sizes.
            if (info_[0] == -3) {
                lfact_ = std::max(2 * lfact_, info_[16]);
                fact_.assign(static_cast<std::size_t>(lfact_), 0.0);
            } else if (info_[0] == -4) {
                lifact_ = std::max(2 * lifact_, info_[17]);
                ifact_.assign(static_cast<std::size_t>(lifact_), 0);
            } else {
                break;
            }
            a.assign(values.begin(), values.end());
        }
        stats_.factor_ms = ms_since(t0, Clock::now());
        if (info_[0] < 0) return fail(Outcome::Failed, "ma57bd_ info(1)=" + std::to_string(info_[0]));
        factored_ = true;
        stats_.inertia_neg = info_[23];               // INFO(24): negative eigenvalues
        stats_.inertia_zero = n_ - info_[24];         // INFO(25): rank
        stats_.inertia_pos = n_ - stats_.inertia_neg - stats_.inertia_zero;
        stats_.factor_entries = info_[13];            // INFO(14): entries in factors
        stats_.outcome = Outcome::Success;
        return Outcome::Success;
    }

    Outcome solve(std::span<double> b_and_x, int nrhs) override {
        if (!available_) return unavailable();
        if (!factored_) return fail(Outcome::Failed, "solve called without valid factor");
        int job = 1;
        int lrhs = n_;
        int lw = n_ * nrhs;
        std::vector<double> w(static_cast<std::size_t>(lw));
        std::vector<int> iw1(static_cast<std::size_t>(n_));
        const auto t0 = Clock::now();
        cd_(&job, &n_, fact_.data(), &lfact_, ifact_.data(), &lifact_, &nrhs, b_and_x.data(), &lrhs,
            w.data(), &lw, iw1.data(), icntl_, info_);
        stats_.solve_ms = ms_since(t0, Clock::now());
        if (info_[0] < 0) return fail(Outcome::Failed, "ma57cd_ info(1)=" + std::to_string(info_[0]));
        stats_.outcome = Outcome::Success;
        return Outcome::Success;
    }

    AdapterStats stats() const override { return stats_; }

private:
    Outcome unavailable() {
        stats_.outcome = Outcome::Unavailable;
        stats_.reason = hsl_unavailable_reason("ma57ad_/ma57bd_/ma57cd_");
        return stats_.outcome;
    }
    Outcome fail(Outcome o, std::string reason) {
        stats_.outcome = o;
        stats_.reason = std::move(reason);
        return o;
    }

    Ma57Id id_ = nullptr;
    Ma57Ad ad_ = nullptr;
    Ma57Bd bd_ = nullptr;
    Ma57Cd cd_ = nullptr;
    bool available_ = false;
    bool factored_ = false;
    double cntl_[5] = {};
    int icntl_[20] = {};
    int info_[40] = {};
    double rinfo_[20] = {};
    int n_ = 0, ne_ = 0, lkeep_ = 0, lfact_ = 0, lifact_ = 0;
    std::vector<int> irn_, jcn_, keep_, iwork_, ifact_;
    std::vector<double> fact_;
    AdapterStats stats_{};
};

} // namespace

std::unique_ptr<SolverAdapter> make_hsl_ma97_adapter() { return std::make_unique<Ma97Adapter>(); }
std::unique_ptr<SolverAdapter> make_hsl_ma57_adapter() { return std::make_unique<Ma57Adapter>(); }

} // namespace smf::bench
