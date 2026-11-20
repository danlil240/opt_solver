// Copyright (C) 2026 smf authors
// All Rights Reserved.
// This code is published under the Eclipse Public License.

#ifndef __IPSMFSOLVERINTERFACE_HPP__
#define __IPSMFSOLVERINTERFACE_HPP__

#include "IpSparseSymLinearSolverInterface.hpp"
#include "IpLibraryLoader.hpp"
#include "IpTypes.h"

#include <stdint.h>
#include <string>

#ifdef IPOPT_HAS_SMF
extern "C"
{
#include "smf/smf_c.h"
}
#else
/* Self-contained ABI v1 definitions matching smf/smf_c.h for runtime dlopen via smflib. */
#ifdef __cplusplus
extern "C"
{
#endif

#define SMF_C_ABI_VERSION        1
#define SMF_OK                   0
#define SMF_ERR_INVALID_ARG      1
#define SMF_ERR_SINGULAR         2
#define SMF_ERR_INTERNAL         3
#define SMF_ERR_WRONG_INERTIA    4
#define SMF_ERR_INEXACT_INERTIA  5

#define SMF_POSDEF               0
#define SMF_INDEF                1

#define SMF_SOLVE_FULL           0
#define SMF_SOLVE_FORWARD        1
#define SMF_SOLVE_DIAGONAL       2
#define SMF_SOLVE_BACKWARD       3

#define SMF_PIVOTING_THRESHOLD   0
#define SMF_PIVOTING_STATIC      1

#define SMF_ORDER_AUTO           0
#define SMF_ORDER_AMD            1
#define SMF_ORDER_METIS          2
#define SMF_ORDER_MATCHING       3

#define SMF_SCALE_NONE           0
#define SMF_SCALE_MATCHING       1
#define SMF_SCALE_EQUILIBRATION  2
#define SMF_SCALE_AUTO           3

#define SMF_SLACK_OFF            0
#define SMF_SLACK_ON             1
#define SMF_SLACK_AUTO           2

typedef struct smf_ipm_session_s* smf_ipm_session_t;

typedef struct
{
   int    abi_version;
   int    struct_size;
   int    ordering;
   int    scaling;
   int    pivoting;
   int    slack_elimination;
   int    nemin;
   int    num_threads;
   int    deterministic;
   int    continue_on_singular;
   int    honest_inertia;
   int    print_level;
   double u;
   double small;
   double multiplier;
   int    reserved_i[8];
   double reserved_d[8];
} smf_ipm_control_t;

typedef struct
{
   int     struct_size;
   int     status;
   int     num_pos;
   int     num_neg;
   int     num_zero;
   int     matrix_rank;
   int     num_delay;
   int     num_two;
   int     maxfront;
   int     maxsupernode;
   int     maxdepth;
   int     num_sup;
   int     ordering_used;
   int     inertia_exact;
   int     threads_used;
   int     static_pivots;
   int64_t num_factor;
   int64_t num_flops;
   double  analyse_seconds;
   double  factor_seconds;
   double  solve_seconds;
   int     reserved_i[8];
   double  reserved_d[8];
} smf_ipm_info_t;

#ifdef __cplusplus
}
#endif
#endif /* IPOPT_HAS_SMF */

#define IPOPT_DECL_SMF_C_ABI_VERSION(x)       int (x)(void)
#define IPOPT_DECL_SMF_VERSION(x)             const char* (x)(void)
#define IPOPT_DECL_SMF_BLAS_BACKEND(x)        const char* (x)(void)
#define IPOPT_DECL_SMF_IPM_DEFAULT_CONTROL(x) void (x)(smf_ipm_control_t* ctrl)
#define IPOPT_DECL_SMF_IPM_CREATE(x)          int (x)(smf_ipm_session_t* out)
#define IPOPT_DECL_SMF_IPM_ANALYSE(x)         int (x)(smf_ipm_session_t session, int n, const int* col_ptr, const int* row_idx, const double* values, const smf_ipm_control_t* ctrl, smf_ipm_info_t* info)
#define IPOPT_DECL_SMF_IPM_FACTOR(x)          int (x)(smf_ipm_session_t session, int mtype, const double* values, const smf_ipm_control_t* ctrl, smf_ipm_info_t* info, double* scale)
#define IPOPT_DECL_SMF_IPM_SOLVE(x)           int (x)(smf_ipm_session_t session, int job, int nrhs, double* rhs, int ldx, const smf_ipm_control_t* ctrl, smf_ipm_info_t* info)
#define IPOPT_DECL_SMF_IPM_FREE(x)            void (x)(smf_ipm_session_t* session)

namespace Ipopt
{

/** Native interface to the smf sparse symmetric multifrontal linear solver.
 *
 * Uses 0-based upper-triangular CSR (CSR_Format_0_Offset), which is identical
 * to smf's 0-based lower-triangular CSC with zero index translation or sorting.
 * Supports both compile-time linking (IPOPT_HAS_SMF) and runtime loading via
 * the smflib option (independent of hsllib / CoinHSL).
 */
class SmfSolverInterface: public SparseSymLinearSolverInterface
{
private:
   enum ScalingMode
   {
      SCALE_MODE_AUTO,
      SCALE_MODE_DYNAMIC,
      SCALE_MODE_NONE,
      SCALE_MODE_MATCHING,
      SCALE_MODE_EQUILIBRATION
   };

   int               ndim_;
   Number*           val_;
   int               numneg_;
   int               numdelay_;
   smf_ipm_session_t session_;
   bool              pivtol_changed_;
   bool              dynamic_active_;
   bool              rescale_;
   double*           scaling_;

   /* Options */
   smf_ipm_control_t control_;
   double            umax_;
   ScalingMode       scaling_mode_;

   /* Library loader and resolved function pointers */
   SmartPtr<LibraryLoader> smfloader_;

   IPOPT_DECL_SMF_C_ABI_VERSION(*smf_c_abi_version_fn_);
   IPOPT_DECL_SMF_VERSION(*smf_version_fn_);
   IPOPT_DECL_SMF_BLAS_BACKEND(*smf_blas_backend_fn_);
   IPOPT_DECL_SMF_IPM_DEFAULT_CONTROL(*smf_ipm_default_control_fn_);
   IPOPT_DECL_SMF_IPM_CREATE(*smf_ipm_create_fn_);
   IPOPT_DECL_SMF_IPM_ANALYSE(*smf_ipm_analyse_fn_);
   IPOPT_DECL_SMF_IPM_FACTOR(*smf_ipm_factor_fn_);
   IPOPT_DECL_SMF_IPM_SOLVE(*smf_ipm_solve_fn_);
   IPOPT_DECL_SMF_IPM_FREE(*smf_ipm_free_fn_);

   void LoadFunctions();

public:
   explicit SmfSolverInterface(
      SmartPtr<LibraryLoader> smfloader = NULL
   );

   ~SmfSolverInterface();

   static void RegisterOptions(
      SmartPtr<RegisteredOptions> roptions
   );

   /// Name reported in IPOPT's startup banner ("Linear solver: ...").
   static std::string GetName();

   bool InitializeImpl(
      const OptionsList& options,
      const std::string& prefix
   );

   /** @name Methods for requesting solution of the linear system. */
   ///@{
   ESymSolverStatus InitializeStructure(
      Index        dim,
      Index        nonzeros,
      const Index* ia,
      const Index* ja
   );

   Number* GetValuesArrayPtr()
   {
      return val_;
   }

   ESymSolverStatus MultiSolve(
      bool         new_matrix,
      const Index* ia,
      const Index* ja,
      Index        nrhs,
      Number*      rhs_vals,
      bool         check_NegEVals,
      Index        numberOfNegEVals
   );

   Index NumberOfNegEVals() const
   {
      return numneg_;
   }
   ///@}

   /** @name Options of Linear solver */
   ///@{
   bool IncreaseQuality();

   bool ProvidesInertia() const
   {
      return true;
   }

   EMatrixFormat MatrixFormat() const
   {
      return CSR_Format_0_Offset;
   }
   ///@}

   /** @name Methods related to the detection of linearly dependent rows */
   ///@{
   bool ProvidesDegeneracyDetection() const
   {
      return false;
   }

   ESymSolverStatus DetermineDependentRows(
      const Index*      /*ia*/,
      const Index*      /*ja*/,
      std::list<Index>& /*c_deps*/
   )
   {
      return SYMSOLVER_FATAL_ERROR;
   }
   ///@}
};

} // namespace Ipopt

#endif /* __IPSMFSOLVERINTERFACE_HPP__ */
