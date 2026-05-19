#pragma once
// ipopt_adapter.hpp — smf::SmfLinearSolver wraps smf::Solver to satisfy
// Ipopt::TSymLinearSolver.  Gated behind SMF_BUILD_IPOPT_ADAPTER.

#include "smf/ipopt_mock.hpp"
#include "smf/solver.hpp"
#include "smf/types.hpp"
#include "smf/info.hpp"
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"

#include <memory>
#include <vector>

namespace smf {

/// SmfLinearSolver — wraps smf::Solver to satisfy Ipopt::TSymLinearSolver.
///
/// Default matrix type: MatrixType::RealSymmetricIndefinite (KKT systems).
/// Inertia is reported via NumberOfNegEVals() using info_.num_negative.
class SmfLinearSolver : public Ipopt::TSymLinearSolver {
public:
    /// Construct with optional control override.
    /// matrix_type is forced to RealSymmetricIndefinite.
    explicit SmfLinearSolver(Control ctrl = Control{});
    ~SmfLinearSolver() override = default;

    // Non-copyable, movable
    SmfLinearSolver(const SmfLinearSolver&)            = delete;
    SmfLinearSolver& operator=(const SmfLinearSolver&) = delete;
    SmfLinearSolver(SmfLinearSolver&&)                 = default;
    SmfLinearSolver& operator=(SmfLinearSolver&&)      = default;

    // -----------------------------------------------------------------------
    // Ipopt::TSymLinearSolver interface
    // -----------------------------------------------------------------------

    /// Convert 1-based COO sparsity pattern to lower-CSC, run symbolic
    /// analysis.  Must be called before MultiSolve.
    Ipopt::ESymSolverStatus InitializeStructure(int        n,
                                                int        nnz,
                                                const int* irn,
                                                const int* jcn) override;

    /// Factorize (if new_matrix) and solve for all nrhs columns.
    /// rhs_vals is n × nrhs column-major (in-place RHS → solution).
    Ipopt::ESymSolverStatus MultiSolve(bool          new_matrix,
                                       const int*    irn,
                                       const int*    jcn,
                                       const double* values,
                                       int           nrhs,
                                       double*       rhs_vals,
                                       bool          check_NegEVals,
                                       int           numberOfNegEVals) override;

    /// Returns the number of negative eigenvalues from the last factorization.
    int  NumberOfNegEVals() const override;
    bool ProvidesInertia() const override { return true; }

    // -----------------------------------------------------------------------
    // Diagnostics
    // -----------------------------------------------------------------------
    /// Access last Info struct (inertia, timing, etc.).
    const Info& last_info() const { return info_; }

private:
    Control ctrl_;
    Solver  solver_;
    Info    info_{};

    std::unique_ptr<AnalysisKeep> ak_;
    FactorKeep                    fk_;

    int  n_{0};
    int  neg_evals_{0};

    /// Lower-CSC matrix (pattern set in InitializeStructure, values in
    /// MultiSolve).
    CscLower mat_;

    /// coo_to_csc_[k] = original COO index (0-based) that contributes to
    /// CSC position k.  Built in build_pattern(), used in fill_values().
    std::vector<int> coo_to_csc_;

    /// csc_to_cleaned_[k] = position in ak_->cleaned for mat_ CSC entry k.
    /// Built in build_csc_to_cleaned_map() after analyse(), used to push
    /// updated values into ak_->cleaned before every factor() call.
    std::vector<int> csc_to_cleaned_;

    /// Build lower-CSC sparsity pattern from 1-based COO (no values).
    /// Populates mat_.col_ptr, mat_.row_idx, mat_.values (zeroed),
    /// and coo_to_csc_.
    void build_pattern(int n, int nnz, const int* irn, const int* jcn);

    /// Fill mat_.values from a COO values array, using the coo_to_csc_
    /// permutation computed in build_pattern().
    void fill_values(const double* vals);

    /// Build csc_to_cleaned_ mapping: maps each mat_ CSC entry to its
    /// position in ak_->cleaned (the permuted matrix stored in the analysis
    /// keep).  Must be called after a successful analyse().
    void build_csc_to_cleaned_map();

    /// Push mat_.values into ak_->cleaned.values using csc_to_cleaned_.
    void push_values_to_cleaned();
};

} // namespace smf
