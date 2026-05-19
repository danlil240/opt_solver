#pragma once
// ipopt_mock.hpp — minimal stub of IPOPT's TSymLinearSolver interface.
// Used when IPOPT is NOT installed (SMF_BUILD_IPOPT_ADAPTER=ON but no IPOPT).
// IMPORTANT: wrap with include-guard; do NOT conflict with a real IPOPT install.

#ifndef SMF_IPOPT_MOCK_HPP_INCLUDED
#define SMF_IPOPT_MOCK_HPP_INCLUDED

namespace Ipopt {

// ---------------------------------------------------------------------------
// Minimal ESymSolverStatus enum
// ---------------------------------------------------------------------------
enum ESymSolverStatus {
    SYMSOLVER_SUCCESS       = 0,
    SYMSOLVER_FATAL_ERROR   = 1,
    SYMSOLVER_SINGULAR      = 2,
    SYMSOLVER_WRONG_INERTIA = 3
};

// ---------------------------------------------------------------------------
// Abstract base class that SmfLinearSolver implements.
// ---------------------------------------------------------------------------
class TSymLinearSolver {
public:
    virtual ~TSymLinearSolver() = default;

    /// InitializeStructure — called once with the sparsity pattern.
    /// @param n     matrix order
    /// @param nnz   number of lower-triangular entries in the COO input
    /// @param irn   row indices    (1-based, length nnz)
    /// @param jcn   column indices (1-based, length nnz)
    virtual ESymSolverStatus InitializeStructure(int n, int nnz,
                                                 const int* irn,
                                                 const int* jcn) = 0;

    /// MultiSolve — factorize (when new_matrix==true) and/or solve.
    /// @param new_matrix       if true, values changed: re-factorize
    /// @param irn              row indices    (1-based, same pattern as Init)
    /// @param jcn              column indices (1-based, same pattern as Init)
    /// @param values           lower-triangular matrix values (length nnz)
    /// @param nrhs             number of right-hand sides
    /// @param rhs_vals         n × nrhs column-major RHS / solution buffer
    /// @param check_NegEVals   if true, verify inertia
    /// @param numberOfNegEVals expected number of negative eigenvalues
    virtual ESymSolverStatus MultiSolve(bool          new_matrix,
                                        const int*    irn,
                                        const int*    jcn,
                                        const double* values,
                                        int           nrhs,
                                        double*       rhs_vals,
                                        bool          check_NegEVals,
                                        int           numberOfNegEVals) = 0;

    /// Return the number of negative eigenvalues from the last factorization.
    virtual int NumberOfNegEVals() const = 0;

    /// Allow the solver to request a better pivot (e.g., smaller threshold).
    virtual bool IncreaseQuality() { return false; }

    /// Returns true if this solver provides inertia information.
    virtual bool ProvidesInertia() const { return true; }
};

} // namespace Ipopt

#endif // SMF_IPOPT_MOCK_HPP_INCLUDED
