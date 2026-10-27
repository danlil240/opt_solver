#include "adapter.hpp"

#if defined(SMF_BENCH_CUDSS)
#include <cuda_runtime_api.h>
#include <cudss.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace smf::bench {

namespace {

#if defined(SMF_BENCH_CUDSS)

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) noexcept {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// One-time process setup, run before any timed region: creates the CUDA
// primary context (cudaFree(0)) and a throw-away cuDSS handle, which loads the
// library, its kernels' modules and the threading layer. Context creation is
// 100s of ms and is not part of any solver phase, so it is never timed; the
// later cudssCreate() for each adapter then costs well under a millisecond.
struct CudaRuntime {
    bool ok = false;
    std::string reason;
    std::string device;
    std::string version;
};

const CudaRuntime &cuda_runtime() {
    static const CudaRuntime rt = [] {
        CudaRuntime r;
        int count = 0;
        cudaError_t e = cudaGetDeviceCount(&count);
        if (e != cudaSuccess || count <= 0) {
            r.reason = std::string("cudss: no CUDA device (") + cudaGetErrorString(e) + ")";
            return r;
        }
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) r.device = prop.name;
        if (cudaFree(nullptr) != cudaSuccess) {
            r.reason = "cudss: CUDA context creation failed";
            return r;
        }
        cudssHandle_t h = nullptr;
        if (cudssCreate(&h) != CUDSS_STATUS_SUCCESS) {
            r.reason = "cudss: cudssCreate failed";
            return r;
        }
        int major = 0, minor = 0, patch = 0;
        if (cudssGetProperty(MAJOR_VERSION, &major) == CUDSS_STATUS_SUCCESS &&
            cudssGetProperty(MINOR_VERSION, &minor) == CUDSS_STATUS_SUCCESS &&
            cudssGetProperty(PATCH_LEVEL, &patch) == CUDSS_STATUS_SUCCESS) {
            r.version = "cuDSS " + std::to_string(major) + "." + std::to_string(minor) + "." +
                        std::to_string(patch);
        } else {
            r.version = "cuDSS";
        }
        cudssDestroy(h);
        r.ok = true;
        return r;
    }();
    return rt;
}

Outcome map_cudss_status(cudssStatus_t s) {
    switch (s) {
        case CUDSS_STATUS_SUCCESS:
            return Outcome::Success;
        case CUDSS_STATUS_ALLOC_FAILED:
            return Outcome::OutOfMemory;
        case CUDSS_STATUS_INVALID_VALUE:
            return Outcome::InvalidInput;
        default:
            return Outcome::Failed;
    }
}

// cuDSS runs the whole pipeline on the GPU in FP64.
//   * Indefinite: CUDSS_MTYPE_SYMMETRIC, LDL^T with the library's default
//     (diagonal / static) pivoting. cuDSS 0.8 lists CUDSS_PIVOT_BUNCH_KAUFMAN as
//     "reserved for future (not supported yet)", so Bunch-Kaufman cannot be
//     selected; the pivot strategy stays AUTO.
//   * SPD: CUDSS_MTYPE_SPD (Cholesky), as the other adapters use their SPD path.
//   * Iterative refinement inside the solve phase: CUDSS_CONFIG_IR_N_STEPS = 3,
//     the harness's refinement count.
// Timing: every phase is host wall-clock around the cuDSS calls plus a
// cudaDeviceSynchronize, and includes the host<->device copies of the matrix
// structure (analyse), the values (factor) and the right-hand sides / solution
// (solve), because the other adapters also consume and return host memory.
// One-time CUDA context / library initialisation is excluded (cuda_runtime()).
class CudssAdapter final : public SolverAdapter {
public:
    CudssAdapter() = default;
    ~CudssAdapter() override { release(); }

    AdapterIdentity identity() const override {
        const CudaRuntime &rt = cuda_runtime();
        AdapterIdentity id;
        id.solver_id = "cudss";
        id.version = rt.ok ? rt.version + " (" + rt.device + ")" : "unavailable";
        id.library_path = "cudss64";
        id.integer_abi = "int32";
        id.available = rt.ok;
        id.supports_spd = true;
        id.supports_indefinite = true;
        id.is_external_memory = true;  // GPU memory
        return id;
    }

    void configure(const Control &ctrl, const std::string &) override {
        indefinite_ = (ctrl.matrix_type == MatrixType::RealSymmetricIndefinite);
    }

    Outcome analyse(const CscLower &A) override {
        release();
        stats_ = AdapterStats{};
        const CudaRuntime &rt = cuda_runtime();
        if (!rt.ok) return fail(Outcome::Unavailable, rt.reason);
        if (A.n <= 0 || A.col_ptr.size() != static_cast<std::size_t>(A.n) + 1) {
            return fail(Outcome::InvalidInput, "cudss: invalid CSC shape");
        }

        n_ = static_cast<int>(A.n);
        build_upper_csr(A);

        const auto t0 = Clock::now();
        if (!cuda_ok(cudaMalloc(reinterpret_cast<void **>(&d_offsets_), offsets_.size() * sizeof(int)), "cudaMalloc offsets") ||
            !cuda_ok(cudaMalloc(reinterpret_cast<void **>(&d_cols_), cols_.size() * sizeof(int)), "cudaMalloc cols") ||
            !cuda_ok(cudaMalloc(reinterpret_cast<void **>(&d_vals_), cols_.size() * sizeof(double)), "cudaMalloc values")) {
            return stats_.outcome;
        }
        if (!cuda_ok(cudaMemcpy(d_offsets_, offsets_.data(), offsets_.size() * sizeof(int), cudaMemcpyHostToDevice), "copy offsets") ||
            !cuda_ok(cudaMemcpy(d_cols_, cols_.data(), cols_.size() * sizeof(int), cudaMemcpyHostToDevice), "copy cols") ||
            !cuda_ok(cudaMemcpy(d_vals_, vals_.data(), vals_.size() * sizeof(double), cudaMemcpyHostToDevice), "copy values")) {
            return stats_.outcome;
        }

        if (!cudss_ok(cudssCreate(&handle_), "cudssCreate") ||
            !cudss_ok(cudssConfigCreate(&config_), "cudssConfigCreate") ||
            !cudss_ok(cudssDataCreate(handle_, &data_), "cudssDataCreate")) {
            return stats_.outcome;
        }
        const int ir_steps = 3;
        if (!cudss_ok(cudssConfigSet(config_, CUDSS_CONFIG_IR_N_STEPS, &ir_steps, sizeof(ir_steps)), "set IR steps")) {
            return stats_.outcome;
        }

        const cudssMatrixType_t mtype = indefinite_ ? CUDSS_MTYPE_SYMMETRIC : CUDSS_MTYPE_SPD;
        // A lower-triangular CSC matrix is, byte for byte, the upper triangle
        // in CSR.
        if (!cudss_ok(cudssMatrixCreateCsr(&mat_, n_, n_, static_cast<int64_t>(cols_.size()), d_offsets_, nullptr,
                                           d_cols_, d_vals_, CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F, mtype,
                                           CUDSS_MVIEW_UPPER, CUDSS_BASE_ZERO),
                      "cudssMatrixCreateCsr")) {
            return stats_.outcome;
        }
        if (!ensure_rhs(1)) return stats_.outcome;

        cudssStatus_t st = cudssExecute(handle_, CUDSS_PHASE_ANALYSIS, config_, data_, mat_, x_dn_, b_dn_);
        cudaDeviceSynchronize();
        stats_.analyse_ms = ms_since(t0);
        stats_.outcome = map_cudss_status(st);
        if (st != CUDSS_STATUS_SUCCESS) stats_.reason = "cudss analysis status " + std::to_string(static_cast<int>(st));
        else analysed_ = true;
        return stats_.outcome;
    }

    Outcome factor(std::span<const double> values) override {
        if (!analysed_) return fail(Outcome::InvalidInput, "cudss: factor before analyse");
        if (values.size() != scatter_.size()) return fail(Outcome::InvalidInput, "cudss: values size mismatch");

        std::fill(vals_.begin(), vals_.end(), 0.0);
        for (std::size_t k = 0; k < values.size(); ++k) vals_[static_cast<std::size_t>(scatter_[k])] += values[k];

        const auto t0 = Clock::now();
        if (!cuda_ok(cudaMemcpy(d_vals_, vals_.data(), vals_.size() * sizeof(double), cudaMemcpyHostToDevice), "copy values")) {
            return stats_.outcome;
        }
        // The first numeric factorization builds the factor storage; later
        // ones on the same analysis are refactorizations.
        const cudssPhase_t phase = factored_ ? CUDSS_PHASE_REFACTORIZATION : CUDSS_PHASE_FACTORIZATION;
        cudssStatus_t st = cudssExecute(handle_, phase, config_, data_, mat_, x_dn_, b_dn_);
        cudaDeviceSynchronize();
        stats_.factor_ms = ms_since(t0);
        stats_.reason.clear();

        if (st != CUDSS_STATUS_SUCCESS) {
            // A zero / unusable pivot surfaces as an execution failure.
            stats_.outcome = (st == CUDSS_STATUS_ALLOC_FAILED) ? Outcome::OutOfMemory : Outcome::Singular;
            stats_.reason = "cudss factorization status " + std::to_string(static_cast<int>(st));
            return stats_.outcome;
        }
        factored_ = true;
        read_factor_info();
        stats_.outcome = Outcome::Success;
        return stats_.outcome;
    }

    Outcome solve(std::span<double> b_and_x, int nrhs) override {
        if (!factored_) return fail(Outcome::InvalidInput, "cudss: solve before a successful factor");
        const std::size_t total = static_cast<std::size_t>(n_) * static_cast<std::size_t>(std::max(nrhs, 1));
        if (nrhs <= 0 || b_and_x.size() < total) return fail(Outcome::InvalidInput, "cudss: solve buffer too small");

        if (!ensure_rhs(nrhs)) return stats_.outcome;
        const auto t0 = Clock::now();
        if (!cuda_ok(cudaMemcpy(d_b_, b_and_x.data(), total * sizeof(double), cudaMemcpyHostToDevice), "copy rhs")) {
            return stats_.outcome;
        }
        cudssStatus_t st = cudssExecute(handle_, CUDSS_PHASE_SOLVE, config_, data_, mat_, x_dn_, b_dn_);
        cudaDeviceSynchronize();
        if (st == CUDSS_STATUS_SUCCESS &&
            !cuda_ok(cudaMemcpy(b_and_x.data(), d_x_, total * sizeof(double), cudaMemcpyDeviceToHost), "copy solution")) {
            return stats_.outcome;
        }
        stats_.solve_ms = ms_since(t0);
        stats_.outcome = map_cudss_status(st);
        if (st != CUDSS_STATUS_SUCCESS) stats_.reason = "cudss solve status " + std::to_string(static_cast<int>(st));
        return stats_.outcome;
    }

    AdapterStats stats() const override { return stats_; }

private:
    Outcome fail(Outcome o, std::string reason) {
        stats_.outcome = o;
        stats_.reason = std::move(reason);
        return o;
    }
    bool cuda_ok(cudaError_t e, const char *what) {
        if (e == cudaSuccess) return true;
        fail(e == cudaErrorMemoryAllocation ? Outcome::OutOfMemory : Outcome::Failed,
             std::string("cudss: ") + what + ": " + cudaGetErrorString(e));
        return false;
    }
    bool cudss_ok(cudssStatus_t s, const char *what) {
        if (s == CUDSS_STATUS_SUCCESS) return true;
        fail(map_cudss_status(s), std::string("cudss: ") + what + " status " + std::to_string(static_cast<int>(s)));
        return false;
    }

    // Upper-triangle CSR from the lower-triangle CSC (identical arrays), with
    // an explicit diagonal entry in every row (cuDSS wants one; KKT blocks
    // often omit zero diagonals) and a scatter map from input entry to slot
    // (duplicates are summed, entries above the diagonal are mirrored).
    void build_upper_csr(const CscLower &A) {
        const std::size_t n = static_cast<std::size_t>(A.n);
        std::vector<std::vector<std::pair<int, std::size_t>>> rows(n);  // (col, input entry)
        std::vector<char> has_diag(n, 0);
        for (std::size_t j = 0; j < n; ++j) {
            for (Int p = A.col_ptr[j]; p < A.col_ptr[j + 1]; ++p) {
                const int i = static_cast<int>(A.row_idx[static_cast<std::size_t>(p)]);
                const int jj = static_cast<int>(j);
                const int r = std::min(i, jj), c = std::max(i, jj);
                if (r == c) has_diag[static_cast<std::size_t>(r)] = 1;
                rows[static_cast<std::size_t>(r)].emplace_back(c, static_cast<std::size_t>(p));
            }
        }
        for (std::size_t r = 0; r < n; ++r) {
            if (!has_diag[r]) rows[r].emplace_back(static_cast<int>(r), SIZE_MAX);
            std::sort(rows[r].begin(), rows[r].end());
        }
        offsets_.assign(n + 1, 0);
        cols_.clear();
        scatter_.assign(A.values.size(), 0);
        for (std::size_t r = 0; r < n; ++r) {
            int prev = -1;
            for (const auto &[c, p] : rows[r]) {
                if (c != prev) {
                    cols_.push_back(c);
                    prev = c;
                }
                if (p != SIZE_MAX) scatter_[p] = static_cast<int>(cols_.size()) - 1;
            }
            offsets_[r + 1] = static_cast<int>(cols_.size());
        }
        vals_.assign(cols_.size(), 0.0);
        for (std::size_t k = 0; k < A.values.size(); ++k) vals_[static_cast<std::size_t>(scatter_[k])] += A.values[k];
    }

    // (Re)creates the dense right-hand-side / solution objects for nrhs.
    bool ensure_rhs(int nrhs) {
        if (nrhs == nrhs_ && b_dn_ != nullptr) return true;
        free_rhs();
        const std::size_t total = static_cast<std::size_t>(n_) * static_cast<std::size_t>(nrhs);
        if (!cuda_ok(cudaMalloc(reinterpret_cast<void **>(&d_b_), total * sizeof(double)), "cudaMalloc b") ||
            !cuda_ok(cudaMalloc(reinterpret_cast<void **>(&d_x_), total * sizeof(double)), "cudaMalloc x")) {
            return false;
        }
        cudaMemset(d_b_, 0, total * sizeof(double));
        cudaMemset(d_x_, 0, total * sizeof(double));
        if (!cudss_ok(cudssMatrixCreateDn(&b_dn_, n_, nrhs, n_, d_b_, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR), "dense b") ||
            !cudss_ok(cudssMatrixCreateDn(&x_dn_, n_, nrhs, n_, d_x_, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR), "dense x")) {
            return false;
        }
        nrhs_ = nrhs;
        return true;
    }

    void read_factor_info() {
        size_t written = 0;
        int64_t lu_nnz = 0;
        if (cudssDataGet(handle_, data_, CUDSS_DATA_LU_NNZ, &lu_nnz, sizeof(lu_nnz), &written) == CUDSS_STATUS_SUCCESS) {
            stats_.factor_entries = lu_nnz;
        }
        double flops = 0.0;
        if (cudssDataGet(handle_, data_, CUDSS_DATA_FLOPS, &flops, sizeof(flops), &written) == CUDSS_STATUS_SUCCESS) {
            stats_.flops = flops;
        }
        int npiv = 0;
        if (cudssDataGet(handle_, data_, CUDSS_DATA_NPIVOTS, &npiv, sizeof(npiv), &written) == CUDSS_STATUS_SUCCESS) {
            stats_.static_pivots = npiv;
        }
        stats_.delayed_pivots = 0;
        stats_.inertia_neg = 0;
        stats_.inertia_zero = 0;
        stats_.inertia_pos = n_;
        if (indefinite_) {
            int inertia[2] = {-1, -1};
            if (cudssDataGet(handle_, data_, CUDSS_DATA_INERTIA, inertia, sizeof(inertia), &written) ==
                    CUDSS_STATUS_SUCCESS &&
                inertia[0] >= 0 && inertia[1] >= 0) {
                stats_.inertia_pos = inertia[0];
                stats_.inertia_neg = inertia[1];
                stats_.inertia_zero = std::max(0, n_ - inertia[0] - inertia[1]);
            }
        }
    }

    void free_rhs() {
        if (b_dn_) cudssMatrixDestroy(b_dn_);
        if (x_dn_) cudssMatrixDestroy(x_dn_);
        if (d_b_) cudaFree(d_b_);
        if (d_x_) cudaFree(d_x_);
        b_dn_ = x_dn_ = nullptr;
        d_b_ = d_x_ = nullptr;
        nrhs_ = 0;
    }

    void release() noexcept {
        free_rhs();
        if (mat_) cudssMatrixDestroy(mat_);
        if (handle_ && data_) cudssDataDestroy(handle_, data_);
        if (config_) cudssConfigDestroy(config_);
        if (handle_) cudssDestroy(handle_);
        if (d_offsets_) cudaFree(d_offsets_);
        if (d_cols_) cudaFree(d_cols_);
        if (d_vals_) cudaFree(d_vals_);
        mat_ = nullptr;
        data_ = nullptr;
        config_ = nullptr;
        handle_ = nullptr;
        d_offsets_ = nullptr;
        d_cols_ = nullptr;
        d_vals_ = nullptr;
        analysed_ = factored_ = false;
        n_ = 0;
    }

    bool indefinite_ = true;
    int n_ = 0;
    int nrhs_ = 0;
    bool analysed_ = false;
    bool factored_ = false;
    std::vector<int> offsets_, cols_, scatter_;
    std::vector<double> vals_;
    int *d_offsets_ = nullptr;
    int *d_cols_ = nullptr;
    double *d_vals_ = nullptr;
    double *d_b_ = nullptr;
    double *d_x_ = nullptr;
    cudssHandle_t handle_ = nullptr;
    cudssConfig_t config_ = nullptr;
    cudssData_t data_ = nullptr;
    cudssMatrix_t mat_ = nullptr;
    cudssMatrix_t b_dn_ = nullptr;
    cudssMatrix_t x_dn_ = nullptr;
    AdapterStats stats_{};
};

#else

class UnavailableCudssAdapter final : public SolverAdapter {
public:
    AdapterIdentity identity() const override {
        AdapterIdentity id;
        id.solver_id = "cudss";
        id.version = "not-linked";
        id.library_path = "unavailable";
        id.available = false;
        return id;
    }
    void configure(const Control &, const std::string &) override {}
    Outcome analyse(const CscLower &) override { return Outcome::Unavailable; }
    Outcome factor(std::span<const double>) override { return Outcome::Unavailable; }
    Outcome solve(std::span<double>, int) override { return Outcome::Unavailable; }
    AdapterStats stats() const override {
        return AdapterStats{Outcome::Unavailable, "cudss not built (configure with -DSMF_BENCH_CUDSS=ON)"};
    }
};

#endif // SMF_BENCH_CUDSS

} // namespace

std::unique_ptr<SolverAdapter> make_cudss_adapter() {
#if defined(SMF_BENCH_CUDSS)
    return std::make_unique<CudssAdapter>();
#else
    return std::make_unique<UnavailableCudssAdapter>();
#endif
}

} // namespace smf::bench
