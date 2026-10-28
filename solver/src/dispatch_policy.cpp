#include "smf/dispatch_policy.hpp"

#include <algorithm>
#include <cmath>

namespace smf {

namespace {

int compute_lower_bw(const CscLower &A) noexcept {
    int bw = 0;
    for (Int j = 0; j < A.n; ++j) {
        for (Int p = A.col_ptr[static_cast<std::size_t>(j)];
             p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
            const int d = static_cast<int>(A.row_idx[static_cast<std::size_t>(p)] - j);
            if (d > bw) bw = d;
        }
    }
    return bw;
}

bool has_full_structural_diagonal(const CscLower &A) noexcept {
    if (A.n <= 0) return false;
    for (Int j = 0; j < A.n; ++j) {
        const Int cs = A.col_ptr[static_cast<std::size_t>(j)];
        const Int ce = A.col_ptr[static_cast<std::size_t>(j) + 1];
        if (cs >= ce || A.row_idx[static_cast<std::size_t>(cs)] != j) {
            return false;
        }
    }
    return true;
}

} // namespace

const char *native_path_to_string(NativePath path) noexcept {
    switch (path) {
        case NativePath::TinyDense: return "TinyDense";
        case NativePath::BandedSPD: return "BandedSPD";
        case NativePath::SparseSPD: return "SparseSPD";
        case NativePath::GeneralIndefinite: return "GeneralIndefinite";
        case NativePath::CertifiedQuasiDefinite: return "CertifiedQuasiDefinite";
        default: return "Unknown";
    }
}

DispatchDecision select_native_path(const AnalysisKeep &ak,
                                    const Control &ctrl,
                                    int /*nrhs*/,
                                    int expected_refactors) {
    DispatchDecision dec;
    const int n = static_cast<int>(ak.n);
    const double max_lower_entries =
        (n > 0) ? (0.5 * static_cast<double>(n) * static_cast<double>(n + 1)) : 1.0;
    dec.density =
        (n > 0) ? (static_cast<double>(ak.cleaned.nnz()) / max_lower_entries) : 0.0;
    dec.lower_bandwidth = compute_lower_bw(ak.cleaned);

    if (n > 0 && n <= 16 && dec.density >= 0.5) {
        dec.path = NativePath::TinyDense;
        dec.reason = "tiny_dense_n<=16_high_density";
        return dec;
    }

    if (ctrl.matrix_type == MatrixType::RealSymmetricPositiveDefinite) {
        if (ctrl.ordering == OrderingMethod::AutoParallel &&
            n >= 100 &&
            dec.lower_bandwidth > 0 &&
            dec.lower_bandwidth <= 128) {
            dec.path = NativePath::BandedSPD;
            dec.reason = "narrow_band_spd_lapack_dpbtrf";
            return dec;
        }
        dec.path = NativePath::SparseSPD;
        dec.reason = "multifrontal_spd";
        return dec;
    }

    const int refactors = std::max(expected_refactors, ctrl.expected_refactors);
    if (refactors >= 5 &&
        n > 16 &&
        has_full_structural_diagonal(ak.cleaned)) {
        dec.path = NativePath::CertifiedQuasiDefinite;
        dec.reason = "candidate_quasi_definite_repeated_kkt";
        return dec;
    }

    dec.path = NativePath::GeneralIndefinite;
    dec.reason = "general_indefinite_multifrontal";
    return dec;
}

bool certify_quasi_definite(const CscLower &A, double min_abs_diag) noexcept {
    if (A.n <= 0 || A.validate_shape() != ErrorCode::Success) {
        return false;
    }
    int sign_transitions = 0;
    int prev_sign = 0;
    int pos_count = 0;
    int neg_count = 0;

    for (Int j = 0; j < A.n; ++j) {
        bool found_diag = false;
        double diag_val = 0.0;
        for (Int p = A.col_ptr[static_cast<std::size_t>(j)];
             p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
            if (A.row_idx[static_cast<std::size_t>(p)] == j) {
                diag_val += A.values[static_cast<std::size_t>(p)];
                found_diag = true;
            }
        }
        if (!found_diag || !std::isfinite(diag_val) || std::abs(diag_val) < min_abs_diag) {
            return false;
        }
        const int cur_sign = (diag_val > 0.0) ? 1 : -1;
        if (cur_sign > 0) ++pos_count;
        else ++neg_count;
        if (prev_sign != 0 && cur_sign != prev_sign) {
            ++sign_transitions;
        }
        prev_sign = cur_sign;
    }
    // Quasi-definite KKT matrix has a positive-definite primal block and negative-definite dual block (at most 1 sign transition)
    return pos_count > 0 && neg_count > 0 && sign_transitions <= 1;
}

} // namespace smf
