// ipopt_adapter.cpp — implementation of SmfLinearSolver.

#include "smf/ipopt_adapter.hpp"
#include "smf/analysis.hpp"    // AnalysisKeep full definition (perm/iperm/cleaned)
#include "smf/error.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>

namespace smf {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

SmfLinearSolver::SmfLinearSolver(Control ctrl)
    : ctrl_(ctrl)
{
    // Enforce indefinite mode — KKT systems are always indefinite.
    ctrl_.matrix_type = MatrixType::RealSymmetricIndefinite;
}

// ---------------------------------------------------------------------------
// build_pattern — convert 1-based COO to lower-CSC (no values)
// ---------------------------------------------------------------------------

void SmfLinearSolver::build_pattern(int n, int nnz,
                                    const int* irn,
                                    const int* jcn)
{
    // Collect lower-triangle entries: row >= col (both 0-based after shift).
    struct Entry { int row; int col; int orig; };
    std::vector<Entry> entries;
    entries.reserve(static_cast<std::size_t>(nnz));

    for (int k = 0; k < nnz; ++k) {
        const int r = irn[k] - 1;   // 1-based → 0-based
        const int c = jcn[k] - 1;
        if (r >= 0 && c >= 0 && r < n && c < n && r >= c) {
            entries.push_back({r, c, k});
        }
    }

    // Sort by column, then row (ensures CSC ordering).
    std::sort(entries.begin(), entries.end(),
              [](const Entry& a, const Entry& b) noexcept {
                  return (a.col != b.col) ? (a.col < b.col) : (a.row < b.row);
              });

    const int m = static_cast<int>(entries.size());

    // Build CSC arrays.
    mat_.n = n;
    mat_.col_ptr.assign(static_cast<std::size_t>(n + 1), 0);
    mat_.row_idx.resize(static_cast<std::size_t>(m));
    mat_.values.assign(static_cast<std::size_t>(m), 0.0);

    for (const auto& e : entries) {
        mat_.col_ptr[static_cast<std::size_t>(e.col + 1)]++;
    }
    for (int j = 0; j < n; ++j) {
        mat_.col_ptr[static_cast<std::size_t>(j + 1)] +=
            mat_.col_ptr[static_cast<std::size_t>(j)];
    }
    for (int k = 0; k < m; ++k) {
        mat_.row_idx[static_cast<std::size_t>(k)] = entries[static_cast<std::size_t>(k)].row;
    }

    // Build COO→CSC index map for later value fills.
    coo_to_csc_.resize(static_cast<std::size_t>(m));
    for (int k = 0; k < m; ++k) {
        coo_to_csc_[static_cast<std::size_t>(k)] =
            entries[static_cast<std::size_t>(k)].orig;
    }
}

// ---------------------------------------------------------------------------
// fill_values — update mat_.values from a raw COO values array
// ---------------------------------------------------------------------------

void SmfLinearSolver::fill_values(const double* vals)
{
    const int m = static_cast<int>(coo_to_csc_.size());
    for (int k = 0; k < m; ++k) {
        mat_.values[static_cast<std::size_t>(k)] =
            vals[static_cast<std::size_t>(coo_to_csc_[static_cast<std::size_t>(k)])];
    }
}

// ---------------------------------------------------------------------------
// build_csc_to_cleaned_map — map mat_ CSC positions → ak_->cleaned positions
// ---------------------------------------------------------------------------

void SmfLinearSolver::build_csc_to_cleaned_map()
{
    // ak_->cleaned is the result of permute_lower_csc(original, perm, iperm).
    // For each entry k in mat_ at (row=mat_.row_idx[k], col=j) we need to
    // find its position in ak_->cleaned after the permutation (iperm[col],
    // iperm[row]).  ak_->cleaned columns are sorted by row within each column.

    const int n = n_;
    const auto& iperm = ak_->iperm;
    const int mat_nnz = static_cast<int>(mat_.values.size());

    csc_to_cleaned_.resize(static_cast<std::size_t>(mat_nnz), -1);

    for (int old_j = 0; old_j < n; ++old_j) {
        const int new_j = static_cast<int>(iperm[static_cast<std::size_t>(old_j)]);
        for (int p = mat_.col_ptr[static_cast<std::size_t>(old_j)];
             p < mat_.col_ptr[static_cast<std::size_t>(old_j) + 1]; ++p) {
            const int old_i = mat_.row_idx[static_cast<std::size_t>(p)];
            const int new_i = static_cast<int>(iperm[static_cast<std::size_t>(old_i)]);

            // Lower triangle in permuted space: keep new_i >= new_j.
            const int col = (new_i >= new_j) ? new_j : new_i;
            const int row = (new_i >= new_j) ? new_i : new_j;

            // Binary-search for `row` in ak_->cleaned column `col`.
            const int cs = static_cast<int>(
                ak_->cleaned.col_ptr[static_cast<std::size_t>(col)]);
            const int ce = static_cast<int>(
                ak_->cleaned.col_ptr[static_cast<std::size_t>(col) + 1]);

            const auto beg = ak_->cleaned.row_idx.begin() + cs;
            const auto end = ak_->cleaned.row_idx.begin() + ce;
            const auto it  = std::lower_bound(beg, end, row);

            // Record position (lower_bound gives exact match if entry exists).
            csc_to_cleaned_[static_cast<std::size_t>(p)] =
                static_cast<int>(it - ak_->cleaned.row_idx.begin());
        }
    }
}

// ---------------------------------------------------------------------------
// push_values_to_cleaned — write mat_.values into ak_->cleaned.values
// ---------------------------------------------------------------------------

void SmfLinearSolver::push_values_to_cleaned()
{
    // Zero first to handle any entries that may have been dropped by
    // check_matrix (duplicates summed, structural zeros removed, etc.).
    std::fill(ak_->cleaned.values.begin(), ak_->cleaned.values.end(), 0.0);

    const int mat_nnz = static_cast<int>(mat_.values.size());
    for (int k = 0; k < mat_nnz; ++k) {
        const int pos = csc_to_cleaned_[static_cast<std::size_t>(k)];
        if (pos >= 0 &&
            pos < static_cast<int>(ak_->cleaned.values.size())) {
            ak_->cleaned.values[static_cast<std::size_t>(pos)] +=
                mat_.values[static_cast<std::size_t>(k)];
        }
    }
}

// ---------------------------------------------------------------------------
// InitializeStructure
// ---------------------------------------------------------------------------

Ipopt::ESymSolverStatus
SmfLinearSolver::InitializeStructure(int n, int nnz,
                                     const int* irn,
                                     const int* jcn)
{
    n_ = n;
    neg_evals_ = 0;
    info_ = Info{};

    // Build lower-CSC pattern (values zeroed).
    build_pattern(n, nnz, irn, jcn);

    // Symbolic analysis.
    ak_ = solver_.analyse(mat_, ctrl_, info_);
    if (!ak_ || info_.status != ErrorCode::Success) {
        return Ipopt::SYMSOLVER_FATAL_ERROR;
    }

    // Build the mat_ CSC → ak_->cleaned position mapping so that
    // push_values_to_cleaned() can update values efficiently in MultiSolve.
    build_csc_to_cleaned_map();

    return Ipopt::SYMSOLVER_SUCCESS;
}

// ---------------------------------------------------------------------------
// MultiSolve
// ---------------------------------------------------------------------------

Ipopt::ESymSolverStatus
SmfLinearSolver::MultiSolve(bool          new_matrix,
                             const int*    /*irn*/,
                             const int*    /*jcn*/,
                             const double* values,
                             int           nrhs,
                             double*       rhs_vals,
                             bool          check_NegEVals,
                             int           numberOfNegEVals)
{
    if (!ak_) {
        // InitializeStructure was never called successfully.
        return Ipopt::SYMSOLVER_FATAL_ERROR;
    }

    if (new_matrix) {
        // Update matrix values, push them into the analysis keep, then
        // re-factorize.
        fill_values(values);
        push_values_to_cleaned();

        const FactorStatus fs = solver_.factor(*ak_, ctrl_, info_, fk_);

        if (fs == FactorStatus::Singular) {
            return Ipopt::SYMSOLVER_SINGULAR;
        }
        if (fs != FactorStatus::Success && fs != FactorStatus::MaxPivotDelays) {
            return Ipopt::SYMSOLVER_FATAL_ERROR;
        }

        neg_evals_ = info_.num_negative;
    }

    // Inertia check (before solving, consistent with IPOPT convention).
    if (check_NegEVals && neg_evals_ != numberOfNegEVals) {
        return Ipopt::SYMSOLVER_WRONG_INERTIA;
    }

    // Solve for all nrhs right-hand sides (column-major, each column length n_).
    if (nrhs > 0) {
        const int rc = solver_.solve(fk_, ctrl_, info_,
                                     rhs_vals, n_, nrhs,
                                     SolveJob::Full);
        if (rc != 0) {
            return Ipopt::SYMSOLVER_FATAL_ERROR;
        }
    }

    return Ipopt::SYMSOLVER_SUCCESS;
}

// ---------------------------------------------------------------------------
// NumberOfNegEVals
// ---------------------------------------------------------------------------

int SmfLinearSolver::NumberOfNegEVals() const
{
    return neg_evals_;
}

} // namespace smf
