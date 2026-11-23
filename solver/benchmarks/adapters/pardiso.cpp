#include "adapter.hpp"

#if defined(SMF_HAS_MKL_PARDISO)
#include <mkl.h>
#include <mkl_pardiso.h>
#endif

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace smf::bench {

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0, Clock::time_point t1) noexcept {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

Outcome map_pardiso_error(int error, bool is_factor_phase) {
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

#if defined(SMF_HAS_MKL_PARDISO)

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
    //
    // iparm is 0-based here (C array); MKL's documentation numbers it 1-based,
    // so iparm_[k] is the manual's iparm(k+1). iparm_[0] = 1 means "every
    // entry below is user-supplied", so each one must match MKL's own default
    // unless there is a reason not to. Reference: pardisoinit() of the linked
    // oneMKL 2026.2 (mkl_version.h INTEL_MKL_VERSION 20260200; the 12900K kit
    // runs the same build, 20260915) returns, for mtype = -2:
    //   iparm[1]=3, iparm[7]=2, iparm[9]=8, iparm[17]=-1, iparm[18]=-1,
    //   iparm[20]=1, everything else 0;
    // for mtype = 2 the same without iparm[9] and iparm[20].
    void build_iparm() {
        std::memset(iparm_, 0, sizeof(iparm_));
        iparm_[0] = 1;   // use non-default iparm values
        // Fill-in reducing ordering: 3 = parallel (OpenMP) METIS nested
        // dissection, oneMKL 2026.2's pardisoinit default (it was 2, serial
        // METIS, in older releases). Same algorithm family; at 1 thread it
        // runs serially. SMF_BENCH_PARDISO_ORDERING=0|2|3 overrides it for
        // A/B checks.
        iparm_[1] = 3;
        if (const char *e = std::getenv("SMF_BENCH_PARDISO_ORDERING")) {
            // Only an exact 0, 2 or 3 is taken; anything else keeps the default.
            char *end = nullptr;
            const long v = std::strtol(e, &end, 10);
            if (end != e && *end == '\0' && (v == 0 || v == 2 || v == 3)) {
                iparm_[1] = static_cast<MKL_INT>(v);
            }
        }
        iparm_[5] = 0;   // write solution to x (not b)
        // iparm_[7] = 0: up to two refinement steps when pivots were perturbed
        // (MKL documents 0 as "two steps"; pardisoinit's 2 is the same cap).
        // Two-level parallel factorization: a deliberate deviation from the
        // default (0) at every thread count, kept from the original setup; MKL
        // recommends it for parallel runs and it is a no-op at one thread.
        iparm_[23] = 1;
        iparm_[26] = (track_ == "common") ? 1 : 0;
        if (indefinite_) {
            // Pivot perturbation 10^-8 (iparm(10) = 8): MKL's default for
            // symmetric indefinite matrices (13 is its nonsymmetric default).
            iparm_[9] = 8;
            // Scaling + weighted matching (iparm(11), iparm(13)): off by default
            // for mtype -2, but the MKL manual recommends both for highly
            // indefinite saddle-point / interior-point systems, which is what
            // this suite is.
            iparm_[10] = 1;
            iparm_[12] = 1;
            // Bunch-Kaufman 1x1 and 2x2 pivoting (iparm(21) = 1), MKL's
            // default for symmetric indefinite matrices. With iparm_[0] = 1 a
            // zero here meant 1x1 diagonal pivoting only, contradicting the
            // class comment above.
            iparm_[20] = 1;
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

using PanuaInitFn = void (*)(void *, int *, int *, int *, double *, int *);
using PanuaRunFn = void (*)(void *, int *, int *, int *, int *, int *,
                            void *, int *, int *, int *, int *, int *,
                            int *, void *, void *, int *, double *);

std::string panua_license_error(int error) {
    switch (error) {
        case -10:
            return "panua_pardiso: no license file found (panua.lic, error -10)";
        case -11:
            return "panua_pardiso: license is expired (error -11)";
        case -12:
            return "panua_pardiso: license wrong username or hostname (error -12)";
        default:
            return "panua_pardiso: license or initialization failed (error " +
                   std::to_string(error) + ")";
    }
}

Outcome map_panua_error(int error, bool is_factor_phase) {
    if (error == -10 || error == -11 || error == -12) {
        return Outcome::Unavailable;
    }
    return map_pardiso_error(error, is_factor_phase);
}

void *open_panua_library(const std::string &path, std::string &err_msg) {
#if defined(_WIN32)
    HMODULE mod = LoadLibraryA(path.c_str());
    if (mod == nullptr) {
        const DWORD err = GetLastError();
        err_msg = "panua_pardiso: LoadLibraryA(\"" + path +
                  "\") failed (Win32 error " + std::to_string(err) + ")";
    }
    return static_cast<void *>(mod);
#else
    void *mod = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (mod == nullptr) {
        const char *dl_err = dlerror();
        err_msg = "panua_pardiso: dlopen(\"" + path + "\") failed" +
                  (dl_err != nullptr ? std::string(": ") + dl_err : std::string());
    }
    return mod;
#endif
}

void close_panua_library(void *handle) noexcept {
    if (handle == nullptr) return;
#if defined(_WIN32)
    FreeLibrary(static_cast<HMODULE>(handle));
#else
    dlclose(handle);
#endif
}

template <typename Fn>
Fn resolve_panua_symbol(void *handle, const char *name) noexcept {
    if (handle == nullptr) return nullptr;
#if defined(_WIN32)
    FARPROC sym = GetProcAddress(static_cast<HMODULE>(handle), name);
#else
    void *sym = dlsym(handle, name);
#endif
    if (sym == nullptr) return nullptr;
    Fn fn = nullptr;
    static_assert(sizeof(Fn) == sizeof(sym));
    std::memcpy(&fn, &sym, sizeof(fn));
    return fn;
}

class PanuaPardisoSolverAdapter final : public SolverAdapter {
public:
    PanuaPardisoSolverAdapter() {
        std::memset(pt_, 0, sizeof(pt_));
        std::memset(iparm_, 0, sizeof(iparm_));
        std::memset(dparm_, 0, sizeof(dparm_));

        const char *env_path = std::getenv("SMF_PANUA_PARDISO_LIB");
        if (env_path != nullptr && *env_path != '\0') {
            lib_path_ = env_path;
        } else {
#if defined(_WIN32)
            lib_path_ = "libpardiso.dll";
#elif defined(__APPLE__)
            lib_path_ = "libpardiso.dylib";
#else
            lib_path_ = "libpardiso.so";
#endif
        }

        handle_ = open_panua_library(lib_path_, unavail_reason_);
        if (handle_ == nullptr) {
            stats_.outcome = Outcome::Unavailable;
            stats_.reason = unavail_reason_;
            return;
        }

        // Resolve Panua's double-precision entry points first; fallback to
        // unsuffixed names strictly from the loaded handle (never RTLD_DEFAULT)
        // so symbols never collide with statically linked Intel MKL PARDISO.
        init_fn_ = resolve_panua_symbol<PanuaInitFn>(handle_, "pardisoinit_d");
        run_fn_ = resolve_panua_symbol<PanuaRunFn>(handle_, "pardiso_d");
        if (init_fn_ == nullptr) {
            init_fn_ = resolve_panua_symbol<PanuaInitFn>(handle_, "pardisoinit");
        }
        if (run_fn_ == nullptr) {
            run_fn_ = resolve_panua_symbol<PanuaRunFn>(handle_, "pardiso");
        }
        if (init_fn_ == nullptr || run_fn_ == nullptr) {
            unavail_reason_ = "panua_pardiso: '" + lib_path_ +
                              "' does not export pardisoinit_d/pardiso_d or pardisoinit/pardiso";
            unload_library();
            stats_.outcome = Outcome::Unavailable;
            stats_.reason = unavail_reason_;
            return;
        }

        // One-time license probe so identity().available accurately reflects
        // missing/expired/mismatched panua.lic before any solve phase runs.
        int probe_mtype = -2;
        int probe_solver = 0;
        int probe_error = 0;
        init_fn_(pt_, &probe_mtype, &probe_solver, iparm_, dparm_, &probe_error);
        std::memset(pt_, 0, sizeof(pt_));
        std::memset(iparm_, 0, sizeof(iparm_));
        std::memset(dparm_, 0, sizeof(dparm_));
        if (probe_error != 0) {
            unavail_reason_ = panua_license_error(probe_error);
            unload_library();
            stats_.outcome = Outcome::Unavailable;
            stats_.reason = unavail_reason_;
            return;
        }
        available_ = true;
    }

    ~PanuaPardisoSolverAdapter() override {
        release_handle();
        unload_library();
    }

    PanuaPardisoSolverAdapter(const PanuaPardisoSolverAdapter &) = delete;
    PanuaPardisoSolverAdapter &operator=(const PanuaPardisoSolverAdapter &) = delete;
    PanuaPardisoSolverAdapter(PanuaPardisoSolverAdapter &&) = delete;
    PanuaPardisoSolverAdapter &operator=(PanuaPardisoSolverAdapter &&) = delete;

    AdapterIdentity identity() const override {
        AdapterIdentity id;
        id.solver_id = "panua_pardiso";
        id.version = available_ ? "panua_pardiso (runtime-loaded)" : "not-linked";
        id.library_path = available_ ? lib_path_ : "unavailable";
        id.integer_abi = "int32";
        id.available = available_;
        id.supports_spd = true;
        id.supports_indefinite = true;
        return id;
    }

    void configure(const Control &ctrl, const std::string &track) override {
        indefinite_ = (ctrl.matrix_type == MatrixType::RealSymmetricIndefinite);
        track_ = track;
        threads_ = std::max(1, ctrl.num_threads);
    }

    Outcome analyse(const CscLower &A) override {
        release_handle();
        stats_ = AdapterStats{};
        if (!available_) return unavailable();

        if (A.n <= 0 || A.n > std::numeric_limits<int>::max() ||
            A.col_ptr.size() != static_cast<std::size_t>(A.n) + 1 ||
            A.row_idx.size() != A.values.size() ||
            A.col_ptr[0] != 0 ||
            A.col_ptr.back() != static_cast<Int>(A.row_idx.size()) ||
            A.row_idx.size() >= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            stats_.outcome = Outcome::InvalidInput;
            stats_.reason = "panua_pardiso: invalid CSC shape";
            return stats_.outcome;
        }
        for (std::size_t j = 0; j < static_cast<std::size_t>(A.n); ++j) {
            if (A.col_ptr[j] > A.col_ptr[j + 1]) {
                stats_.outcome = Outcome::InvalidInput;
                stats_.reason = "panua_pardiso: non-monotone col_ptr";
                return stats_.outcome;
            }
            for (Int p = A.col_ptr[j]; p < A.col_ptr[j + 1]; ++p) {
                const Int i = A.row_idx[static_cast<std::size_t>(p)];
                if (i < 0 || i >= A.n) {
                    stats_.outcome = Outcome::InvalidInput;
                    stats_.reason = "panua_pardiso: row_idx out of range";
                    return stats_.outcome;
                }
            }
        }
        if (!build_upper_csr(A)) {
            stats_.outcome = Outcome::InvalidInput;
            stats_.reason = "panua_pardiso: matrix nnz exceeds int32 limit";
            return stats_.outcome;
        }

        session_mtype_ = indefinite_ ? -2 : 2;
        int solver = 0;
        int error = 0;
        std::memset(pt_, 0, sizeof(pt_));
        std::memset(iparm_, 0, sizeof(iparm_));
        std::memset(dparm_, 0, sizeof(dparm_));
        init_fn_(pt_, &session_mtype_, &solver, iparm_, dparm_, &error);
        if (error != 0) {
            stats_.outcome = map_panua_error(error, false);
            stats_.reason = (stats_.outcome == Outcome::Unavailable)
                                ? panua_license_error(error)
                                : ("panua_pardisoinit error " + std::to_string(error));
            return stats_.outcome;
        }

        // Inspected Panua PARDISO 8.2 (libpardiso.dll 20240630) pardisoinit_d defaults:
        //   mtype =  2 (SPD):   iparm[0]=1, iparm[1]=2 (METIS), iparm[17]=-1, iparm[18]=-1,
        //                       iparm[23]=1, iparm[24]=1; iparm[10]=0, iparm[12]=0, iparm[20]=0.
        //   mtype = -2 (indef): same plus iparm[9]=8 (pivot perturbation 1e-8), while
        //                       iparm[10]=0, iparm[12]=0, iparm[20]=0.
        // Upgrade symmetric indefinite / KKT runs from (0, 0, 0) to Panua's manual
        // KKT recommendations for parity with MKL PARDISO: scaling (iparm[10]=1),
        // weighted matching (iparm[12]=2), and 2x2 Bunch-Kaufman pivoting with
        // weighted matching (iparm[20]=3).
        iparm_[2] = threads_;
        iparm_[5] = 0;  // write solution to x (do not overwrite b)
        if (indefinite_) {
            iparm_[10] = 1;
            iparm_[12] = 2;
            iparm_[20] = 3;
        }

        n_ = static_cast<int>(A.n);
        perm_.assign(static_cast<std::size_t>(n_), 0);
        x_.assign(static_cast<std::size_t>(n_), 0.0);

        int phase = 11, maxfct = 1, mnum = 1, nrhs = 1;
        double ddum = 0.0;
        const auto t0 = Clock::now();
        run_fn_(pt_, &maxfct, &mnum, &session_mtype_, &phase, &n_,
                a_.data(), ia_.data(), ja_.data(), perm_.data(), &nrhs,
                iparm_, &msglvl_, &ddum, &ddum, &error, dparm_);
        const auto t1 = Clock::now();
        stats_.analyse_ms = ms_since(t0, t1);

        if (error == 0) {
            analysed_ = true;
            stats_.factor_entries = iparm_[17] > 0 ? static_cast<int64_t>(iparm_[17]) : 0;
            // Panua's iparm_[18] reports factorization work in GFLOPS (10^9 flops,
            // integer-truncated: verified dense 500x500 -> 0, dense 1500x1500 -> 1
            // against libpardiso.dll, matching Panua's pardiso_sym.c).
            stats_.flops = iparm_[18] > 0 ? static_cast<double>(iparm_[18]) * 1e9 : 0.0;
        } else {
            analysed_ = false;
        }

        stats_.outcome = map_panua_error(error, false);
        if (error != 0) {
            stats_.reason = (stats_.outcome == Outcome::Unavailable)
                                ? panua_license_error(error)
                                : ("panua_pardiso phase 11 error " + std::to_string(error));
        }
        return stats_.outcome;
    }

    Outcome factor(std::span<const double> values) override {
        if (!available_) return unavailable();
        if (!analysed_ || n_ <= 0 || values.size() != input_nnz_) {
            stats_.outcome = Outcome::InvalidInput;
            stats_.reason = "panua_pardiso: factor before analyse or values size mismatch";
            return stats_.outcome;
        }
        std::fill(a_.begin(), a_.end(), 0.0);
        for (std::size_t k = 0; k < input_nnz_; ++k) {
            a_[static_cast<std::size_t>(scatter_[k])] += values[k];
        }

        int phase = 22, error = 0, maxfct = 1, mnum = 1, nrhs = 1;
        double ddum = 0.0;
        const auto t0 = Clock::now();
        run_fn_(pt_, &maxfct, &mnum, &session_mtype_, &phase, &n_,
                a_.data(), ia_.data(), ja_.data(), perm_.data(), &nrhs,
                iparm_, &msglvl_, &ddum, &ddum, &error, dparm_);
        const auto t1 = Clock::now();
        stats_.factor_ms = ms_since(t0, t1);
        stats_.reason.clear();

        if (error != 0) {
            stats_.inertia_pos = 0;
            stats_.inertia_neg = 0;
            stats_.inertia_zero = 0;
            factored_ = false;
        } else {
            if (indefinite_) {
                stats_.inertia_pos = iparm_[21];
                stats_.inertia_neg = iparm_[22];
                stats_.inertia_zero =
                    std::max(0, n_ - stats_.inertia_pos - stats_.inertia_neg);
            } else {
                stats_.inertia_pos = n_;
                stats_.inertia_neg = 0;
                stats_.inertia_zero = 0;
            }
            if (iparm_[17] > 0) {
                stats_.factor_entries = static_cast<int64_t>(iparm_[17]);
            }
            if (iparm_[18] > 0) {
                stats_.flops = static_cast<double>(iparm_[18]) * 1e9;
            }
            factored_ = true;
        }

        stats_.outcome = map_panua_error(error, true);
        if (error != 0) {
            stats_.reason = (stats_.outcome == Outcome::Unavailable)
                                ? panua_license_error(error)
                                : ("panua_pardiso phase 22 error " + std::to_string(error));
        }
        return stats_.outcome;
    }

    Outcome solve(std::span<double> b_and_x, int nrhs) override {
        if (!available_) return unavailable();
        if (!analysed_ || n_ <= 0 || nrhs <= 0 ||
            b_and_x.size() < static_cast<std::size_t>(n_) *
                                 static_cast<std::size_t>(nrhs)) {
            stats_.outcome = Outcome::InvalidInput;
            stats_.reason = "panua_pardiso: solve buffer too small or not analysed";
            return stats_.outcome;
        }
        if (!factored_) {
            stats_.outcome = Outcome::Failed;
            stats_.reason = "panua_pardiso: solve called without valid factor";
            return stats_.outcome;
        }
        x_.assign(static_cast<std::size_t>(n_) *
                      static_cast<std::size_t>(nrhs),
                  0.0);

        int phase = 33, error = 0, maxfct = 1, mnum = 1;
        int nrhs_k = nrhs;
        const auto t0 = Clock::now();
        run_fn_(pt_, &maxfct, &mnum, &session_mtype_, &phase, &n_,
                a_.data(), ia_.data(), ja_.data(), perm_.data(), &nrhs_k,
                iparm_, &msglvl_, b_and_x.data(), x_.data(), &error, dparm_);
        const auto t1 = Clock::now();
        stats_.solve_ms = ms_since(t0, t1);
        stats_.reason.clear();

        if (error == 0) {
            std::memcpy(b_and_x.data(), x_.data(),
                        static_cast<std::size_t>(n_) *
                            static_cast<std::size_t>(nrhs) * sizeof(double));
        }
        stats_.outcome = map_panua_error(error, false);
        if (error != 0) {
            stats_.reason = (stats_.outcome == Outcome::Unavailable)
                                ? panua_license_error(error)
                                : ("panua_pardiso phase 33 error " + std::to_string(error));
        }
        return stats_.outcome;
    }

    AdapterStats stats() const override { return stats_; }

private:
    Outcome unavailable() {
        stats_.outcome = Outcome::Unavailable;
        stats_.reason = unavail_reason_;
        return stats_.outcome;
    }

    // Build 1-based upper-triangle CSR (ia_, ja_, a_) from lower-triangle CSC A,
    // inserting explicit 0.0 diagonal entries wherever A omits (j, j) so Panua's
    // diagonal-first upper-CSR contract holds on KKT systems, and recording a
    // scatter_ map for repeat factor() calls.
    bool build_upper_csr(const CscLower &A) {
        const std::size_t n = static_cast<std::size_t>(A.n);
        constexpr std::size_t kInsertedDiag = std::numeric_limits<std::size_t>::max();
        struct RowEntry {
            int col;
            std::size_t src_idx;
        };

        std::vector<std::size_t> row_counts(n, 0);
        std::vector<char> has_diag(n, 0);
        for (std::size_t j = 0; j < n; ++j) {
            const int jj = static_cast<int>(j);
            for (Int p = A.col_ptr[j]; p < A.col_ptr[j + 1]; ++p) {
                const int i = static_cast<int>(A.row_idx[static_cast<std::size_t>(p)]);
                const int r = std::min(i, jj);
                const int c = std::max(i, jj);
                if (r == c) has_diag[static_cast<std::size_t>(r)] = 1;
                ++row_counts[static_cast<std::size_t>(r)];
            }
        }
        for (std::size_t r = 0; r < n; ++r) {
            if (!has_diag[r]) ++row_counts[r];
        }

        std::vector<std::size_t> row_offsets(n + 1, 0);
        for (std::size_t r = 0; r < n; ++r) {
            row_offsets[r + 1] = row_offsets[r] + row_counts[r];
        }
        const std::size_t total_entries = row_offsets[n];
        std::vector<RowEntry> entries(total_entries);
        std::vector<std::size_t> cursor = row_offsets;

        for (std::size_t j = 0; j < n; ++j) {
            const int jj = static_cast<int>(j);
            for (Int p = A.col_ptr[j]; p < A.col_ptr[j + 1]; ++p) {
                const std::size_t up = static_cast<std::size_t>(p);
                const int i = static_cast<int>(A.row_idx[up]);
                const int r = std::min(i, jj);
                const int c = std::max(i, jj);
                entries[cursor[static_cast<std::size_t>(r)]++] = {c, up};
            }
        }
        for (std::size_t r = 0; r < n; ++r) {
            if (!has_diag[r]) {
                entries[cursor[r]++] = {static_cast<int>(r), kInsertedDiag};
            }
        }

        ia_.assign(n + 1, 0);
        ia_[0] = 1;
        ja_.clear();
        ja_.reserve(total_entries);
        input_nnz_ = A.values.size();
        scatter_.assign(input_nnz_, 0);
        for (std::size_t r = 0; r < n; ++r) {
            auto beg = entries.begin() + static_cast<std::ptrdiff_t>(row_offsets[r]);
            auto end = entries.begin() + static_cast<std::ptrdiff_t>(row_offsets[r + 1]);
            std::sort(beg, end, [](const RowEntry &lhs, const RowEntry &rhs) {
                return lhs.col < rhs.col || (lhs.col == rhs.col && lhs.src_idx < rhs.src_idx);
            });
            int prev = -1;
            for (auto it = beg; it != end; ++it) {
                if (it->col != prev) {
                    ja_.push_back(it->col + 1);
                    prev = it->col;
                }
                if (it->src_idx != kInsertedDiag) {
                    scatter_[it->src_idx] = static_cast<int>(ja_.size()) - 1;
                }
            }
            if (ja_.size() >= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                return false;
            }
            ia_[r + 1] = static_cast<int>(ja_.size()) + 1;
        }
        a_.assign(ja_.size(), 0.0);
        for (std::size_t k = 0; k < input_nnz_; ++k) {
            a_[static_cast<std::size_t>(scatter_[k])] += A.values[k];
        }
        return true;
    }

    void release_handle() noexcept {
        if (n_ > 0 && run_fn_ != nullptr) {
            int phase = -1, error = 0, maxfct = 1, mnum = 1, nrhs = 1;
            double ddum = 0.0;
            run_fn_(pt_, &maxfct, &mnum, &session_mtype_, &phase, &n_,
                    a_.empty() ? &ddum : a_.data(),
                    ia_.empty() ? &phase : ia_.data(),
                    ja_.empty() ? &phase : ja_.data(),
                    perm_.empty() ? &phase : perm_.data(),
                    &nrhs, iparm_, &msglvl_, &ddum, &ddum, &error, dparm_);
        }
        std::memset(pt_, 0, sizeof(pt_));
        n_ = 0;
        input_nnz_ = 0;
        analysed_ = false;
        factored_ = false;
    }

    void unload_library() noexcept {
        close_panua_library(handle_);
        handle_ = nullptr;
        init_fn_ = nullptr;
        run_fn_ = nullptr;
    }

    void *handle_ = nullptr;
    PanuaInitFn init_fn_ = nullptr;
    PanuaRunFn run_fn_ = nullptr;
    std::string lib_path_;
    std::string unavail_reason_;
    bool available_ = false;
    bool analysed_ = false;
    bool factored_ = false;

    void *pt_[64];
    int iparm_[64];
    double dparm_[64];
    int session_mtype_ = -2;
    int msglvl_ = 0;
    int n_ = 0;
    std::size_t input_nnz_ = 0;
    bool indefinite_ = true;
    std::string track_ = "recommended";
    int threads_ = 1;
    std::vector<int> ia_, ja_, perm_, scatter_;
    std::vector<double> a_, x_;
    AdapterStats stats_{};
};

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
    if (variant == "panua_pardiso") {
        return std::make_unique<PanuaPardisoSolverAdapter>();
    }
    return std::make_unique<UnavailablePardisoAdapter>(variant);
}

} // namespace smf::bench
