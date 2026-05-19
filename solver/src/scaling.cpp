#include "smf/scaling.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

namespace smf {

std::vector<double> compute_equilibration_scale(const CscLower& A,
                                                int num_iterations) {
    const int n = static_cast<int>(A.n);
    std::vector<double> scale(n, 1.0);

    for (int iter = 0; iter < num_iterations; ++iter) {
        // Compute row_max[i] = max |scale[i] * A_ij * scale[j]|
        // For lower-triangular CSC, entry (row_idx[k], col) with row >= col.
        // The full symmetric matrix has both (i,j) and (j,i).
        std::vector<double> row_max(n, 0.0);

        for (int col = 0; col < n; ++col) {
            for (Int k = A.col_ptr[col]; k < A.col_ptr[col + 1]; ++k) {
                int row = static_cast<int>(A.row_idx[k]);
                double val = std::abs(A.values[k]) * scale[row] * scale[col];
                // diagonal contributes to row=col only once
                if (row == col) {
                    row_max[row] = std::max(row_max[row], val);
                } else {
                    // lower triangle: (row, col) — contributes to both row and col
                    row_max[row] = std::max(row_max[row], val);
                    row_max[col] = std::max(row_max[col], val);
                }
            }
        }

        // Update scale
        for (int i = 0; i < n; ++i) {
            if (row_max[i] > 0.0) {
                scale[i] /= std::sqrt(row_max[i]);
            }
            // else leave scale[i] unchanged (zero row/col)
        }
    }

    return scale;
}

void apply_scale_to_matrix(CscLower& A, const std::vector<double>& scale) {
    const int n = static_cast<int>(A.n);
    for (int col = 0; col < n; ++col) {
        for (Int k = A.col_ptr[col]; k < A.col_ptr[col + 1]; ++k) {
            int row = static_cast<int>(A.row_idx[k]);
            A.values[k] *= scale[row] * scale[col];
        }
    }
}

void apply_scale_to_rhs(double* b, int n, int nrhs,
                         const std::vector<double>& scale) {
    // column-major: b is n x nrhs
    for (int rhs = 0; rhs < nrhs; ++rhs) {
        double* col = b + static_cast<std::ptrdiff_t>(rhs) * n;
        for (int i = 0; i < n; ++i) {
            col[i] *= scale[i];
        }
    }
}

void apply_scale_to_solution(double* x, int n, int nrhs,
                              const std::vector<double>& scale) {
    // Same operation as apply_scale_to_rhs (S is symmetric diagonal)
    apply_scale_to_rhs(x, n, nrhs, scale);
}

} // namespace smf
