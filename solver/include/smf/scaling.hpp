#pragma once
#include "csc_matrix.hpp"
#include <vector>

namespace smf {

// Compute equilibration scaling vector for symmetric matrix A.
// Implements symmetric max-norm equilibration (MC77-like):
//   scale[i] = 1 / sqrt(max(|a_ij|) for j in row/col i)
// Applied iteratively: A_scaled = S * A * S where S = diag(scale).
// num_iterations: 3 for 1-norm, 1 for infinity-norm.
std::vector<double> compute_equilibration_scale(const CscLower& A,
                                                int num_iterations = 3);

// Apply scaling in-place: A_ij <- scale[i] * A_ij * scale[j]
void apply_scale_to_matrix(CscLower& A, const std::vector<double>& scale);

// Apply scaling to RHS vector (column-major n x nrhs): b[i] <- scale[i] * b[i]
void apply_scale_to_rhs(double* b, int n, int nrhs,
                        const std::vector<double>& scale);

// Apply scaling to solution: x[i] <- scale[i] * x[i]
void apply_scale_to_solution(double* x, int n, int nrhs,
                              const std::vector<double>& scale);

} // namespace smf
