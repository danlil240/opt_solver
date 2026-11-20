// Copyright (C) 2026 smf authors
// All Rights Reserved.
// This code is published under the Eclipse Public License.

#include "IpoptConfig.h"
#include "IpSmfSolverInterface.hpp"

#include <cmath>
#include <cstring>

namespace Ipopt
{

SmfSolverInterface::SmfSolverInterface(
   SmartPtr<LibraryLoader> smfloader
)
   : ndim_(0),
     val_(NULL),
     numneg_(0),
     numdelay_(0),
     session_(NULL),
     pivtol_changed_(false),
     dynamic_active_(false),
     rescale_(false),
     scaling_(NULL),
     umax_(1e-4),
     scaling_mode_(SCALE_MODE_AUTO),
     smfloader_(smfloader),
     smf_c_abi_version_fn_(NULL),
     smf_version_fn_(NULL),
     smf_blas_backend_fn_(NULL),
     smf_ipm_default_control_fn_(NULL),
     smf_ipm_create_fn_(NULL),
     smf_ipm_analyse_fn_(NULL),
     smf_ipm_factor_fn_(NULL),
     smf_ipm_solve_fn_(NULL),
     smf_ipm_free_fn_(NULL)
{
   std::memset(&control_, 0, sizeof(control_));
}

SmfSolverInterface::~SmfSolverInterface()
{
   if( session_ != NULL && smf_ipm_free_fn_ != NULL )
   {
      smf_ipm_free_fn_(&session_);
   }
   delete[] val_;
   delete[] scaling_;
}

std::string SmfSolverInterface::GetName()
{
#ifdef IPOPT_HAS_SMF
   std::string name = "smf";
   if( const char* ver = smf_version() )
   {
      name += " ";
      name += ver;
   }
   if( const char* blas = smf_blas_backend() )
   {
      name += " (";
      name += blas;
      name += ")";
   }
   return name;
#else
   return "smf";
#endif
}

void SmfSolverInterface::LoadFunctions()
{
   if( smf_ipm_factor_fn_ != NULL )
   {
      return;
   }

#ifdef IPOPT_HAS_SMF
   smf_c_abi_version_fn_       = &::smf_c_abi_version;
   smf_version_fn_             = &::smf_version;
   smf_blas_backend_fn_        = &::smf_blas_backend;
   smf_ipm_default_control_fn_ = &::smf_ipm_default_control;
   smf_ipm_create_fn_          = &::smf_ipm_create;
   smf_ipm_analyse_fn_         = &::smf_ipm_analyse;
   smf_ipm_factor_fn_          = &::smf_ipm_factor;
   smf_ipm_solve_fn_           = &::smf_ipm_solve;
   smf_ipm_free_fn_            = &::smf_ipm_free;
#else
   DBG_ASSERT(IsValid(smfloader_));
   smf_c_abi_version_fn_ = reinterpret_cast<IPOPT_DECL_SMF_C_ABI_VERSION(*)>(
      smfloader_->loadSymbol("smf_c_abi_version"));
   smf_version_fn_ = reinterpret_cast<IPOPT_DECL_SMF_VERSION(*)>(
      smfloader_->loadSymbol("smf_version"));
   smf_blas_backend_fn_ = reinterpret_cast<IPOPT_DECL_SMF_BLAS_BACKEND(*)>(
      smfloader_->loadSymbol("smf_blas_backend"));
   smf_ipm_default_control_fn_ = reinterpret_cast<IPOPT_DECL_SMF_IPM_DEFAULT_CONTROL(*)>(
      smfloader_->loadSymbol("smf_ipm_default_control"));
   smf_ipm_create_fn_ = reinterpret_cast<IPOPT_DECL_SMF_IPM_CREATE(*)>(
      smfloader_->loadSymbol("smf_ipm_create"));
   smf_ipm_analyse_fn_ = reinterpret_cast<IPOPT_DECL_SMF_IPM_ANALYSE(*)>(
      smfloader_->loadSymbol("smf_ipm_analyse"));
   smf_ipm_factor_fn_ = reinterpret_cast<IPOPT_DECL_SMF_IPM_FACTOR(*)>(
      smfloader_->loadSymbol("smf_ipm_factor"));
   smf_ipm_solve_fn_ = reinterpret_cast<IPOPT_DECL_SMF_IPM_SOLVE(*)>(
      smfloader_->loadSymbol("smf_ipm_solve"));
   smf_ipm_free_fn_ = reinterpret_cast<IPOPT_DECL_SMF_IPM_FREE(*)>(
      smfloader_->loadSymbol("smf_ipm_free"));
#endif

   const int abi = smf_c_abi_version_fn_();
   if( abi != SMF_C_ABI_VERSION )
   {
      THROW_EXCEPTION(DYNAMIC_LIBRARY_FAILURE,
         "smf C ABI version mismatch: loaded library reports incompatible ABI version.");
   }
}

void SmfSolverInterface::RegisterOptions(
   SmartPtr<RegisteredOptions> roptions
)
{
   roptions->AddBoundedNumberOption(
      "smf_u",
      "Pivoting threshold for smf.",
      0.0, false,
      0.5, false,
      1e-8,
      "Threshold parameter u for numerical pivoting in smf LDL^T factorization.");

   roptions->AddBoundedNumberOption(
      "smf_umax",
      "Maximum pivoting threshold for smf.",
      0.0, false,
      0.5, false,
      1e-4,
      "Upper bound on smf_u when Ipopt requests higher factorization quality.");

   roptions->AddLowerBoundedNumberOption(
      "smf_small",
      "Zero-pivot threshold for smf.",
      0.0, true,
      1e-20,
      "Pivots with absolute value smaller than smf_small are treated as zero.");

   roptions->AddLowerBoundedIntegerOption(
      "smf_nemin",
      "Supernode amalgamation threshold for smf.",
      1,
      8,
      "Two assembly-tree nodes are merged if both have fewer than smf_nemin eliminations.");

   roptions->AddStringOption4(
      "smf_ordering",
      "Fill-reducing ordering method for smf.",
      "auto",
      "auto", "Automatic AMD / METIS / Parallel ND / Matching selection",
      "amd", "SuiteSparse Approximate Minimum Degree",
      "metis", "METIS nested dissection",
      "matching", "Matching-based AMD ordering");

   roptions->AddStringOption5(
      "smf_scaling",
      "Scaling strategy for smf.",
      "auto",
      "auto", "Native smf unscaled-first with scale-on-delay rescue",
      "dynamic", "Start unscaled, switch to reused MC64 matching on high delays or quality increase",
      "none", "Disable scaling",
      "matching", "Always apply MC64-style max-product matching scaling",
      "equilibration", "Apply Ruiz equilibration scaling");

   roptions->AddLowerBoundedIntegerOption(
      "smf_num_threads",
      "Number of worker threads for smf (0 = automatic).",
      0,
      0,
      "0 uses SMF_NUM_THREADS, else OMP_NUM_THREADS, else allowed performance cores.");

   roptions->AddStringOption2(
      "smf_pivoting",
      "Pivoting mode for smf.",
      "threshold",
      "threshold", "Accurate threshold pivoting with delayed pivots and exact inertia",
      "static", "Opt-in fast static pivoting (falls back via honest-inertia guard when perturbed)");

   roptions->AddStringOption3(
      "smf_slack_elimination",
      "Inequality slack-pair 2x2 pre-elimination for smf.",
      "off",
      "off", "Disable slack-pair elimination",
      "on", "Enable slack-pair elimination when eligible",
      "auto", "Enable automatically when slack pairs exceed share threshold");

   roptions->AddStringOption2(
      "smf_deterministic",
      "Bitwise-deterministic mode across thread counts for smf.",
      "no",
      "no", "Default parallel execution",
      "yes", "Deterministic orderings, inertia, factors, and solves");

   roptions->AddIntegerOption(
      "smf_print_level",
      "Diagnostic print level for smf (<= 0 silent).",
      -1);
}

bool SmfSolverInterface::InitializeImpl(
   const OptionsList& options,
   const std::string& prefix
)
{
   LoadFunctions();
   smf_ipm_default_control_fn_(&control_);

   options.GetNumericValue("smf_u", control_.u, prefix);
   options.GetNumericValue("smf_umax", umax_, prefix);
   options.GetNumericValue("smf_small", control_.small, prefix);
   options.GetIntegerValue("smf_nemin", control_.nemin, prefix);
   options.GetIntegerValue("smf_num_threads", control_.num_threads, prefix);
   options.GetIntegerValue("smf_print_level", control_.print_level, prefix);

   std::string opt;
   options.GetStringValue("smf_ordering", opt, prefix);
   if( opt == "amd" )
   {
      control_.ordering = SMF_ORDER_AMD;
   }
   else if( opt == "metis" )
   {
      control_.ordering = SMF_ORDER_METIS;
   }
   else if( opt == "matching" )
   {
      control_.ordering = SMF_ORDER_MATCHING;
   }
   else
   {
      control_.ordering = SMF_ORDER_AUTO;
   }

   options.GetStringValue("smf_scaling", opt, prefix);
   if( opt == "dynamic" )
   {
      scaling_mode_ = SCALE_MODE_DYNAMIC;
      control_.scaling = SMF_SCALE_NONE;
   }
   else if( opt == "none" )
   {
      scaling_mode_ = SCALE_MODE_NONE;
      control_.scaling = SMF_SCALE_NONE;
   }
   else if( opt == "matching" )
   {
      scaling_mode_ = SCALE_MODE_MATCHING;
      control_.scaling = SMF_SCALE_MATCHING;
   }
   else if( opt == "equilibration" )
   {
      scaling_mode_ = SCALE_MODE_EQUILIBRATION;
      control_.scaling = SMF_SCALE_EQUILIBRATION;
   }
   else
   {
      scaling_mode_ = SCALE_MODE_AUTO;
      control_.scaling = SMF_SCALE_AUTO;
   }

   options.GetStringValue("smf_pivoting", opt, prefix);
   control_.pivoting = (opt == "static") ? SMF_PIVOTING_STATIC : SMF_PIVOTING_THRESHOLD;

   options.GetStringValue("smf_slack_elimination", opt, prefix);
   if( opt == "on" )
   {
      control_.slack_elimination = SMF_SLACK_ON;
   }
   else if( opt == "auto" )
   {
      control_.slack_elimination = SMF_SLACK_AUTO;
   }
   else
   {
      control_.slack_elimination = SMF_SLACK_OFF;
   }

   options.GetStringValue("smf_deterministic", opt, prefix);
   control_.deterministic = (opt == "yes") ? 1 : 0;

   control_.continue_on_singular = 1;
   control_.honest_inertia = 1;

   numneg_ = 0;
   numdelay_ = 0;
   pivtol_changed_ = false;
   dynamic_active_ = false;
   rescale_ = false;

   if( Jnlst().ProduceOutput(J_DETAILED, J_LINEAR_ALGEBRA) )
   {
      Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
         "smf linear solver initialized: version=%s backend=%s threads=%d u=%.2e\n",
         smf_version_fn_(), smf_blas_backend_fn_(), control_.num_threads, control_.u);
   }

   return true;
}

ESymSolverStatus SmfSolverInterface::InitializeStructure(
   Index        dim,
   Index        nonzeros,
   const Index* ia,
   const Index* ja
)
{
   ndim_ = dim;

   delete[] val_;
   val_ = new Number[nonzeros];

   delete[] scaling_;
   scaling_ = NULL;
   if( scaling_mode_ == SCALE_MODE_DYNAMIC )
   {
      scaling_ = new double[dim];
      for( int i = 0; i < dim; ++i )
      {
         scaling_[i] = 1.0;
      }
   }

   if( session_ != NULL )
   {
      smf_ipm_free_fn_(&session_);
   }
   if( smf_ipm_create_fn_(&session_) != SMF_OK || session_ == NULL )
   {
      Jnlst().Printf(J_ERROR, J_LINEAR_ALGEBRA,
         "smf: failed to create IPM session.\n");
      return SYMSOLVER_FATAL_ERROR;
   }

   smf_ipm_info_t info;
   /* Deferred analysis (values == NULL) so smf's matching trial sees real KKT values
    * on the first factorization call. */
   const int rc = smf_ipm_analyse_fn_(session_, dim, ia, ja, NULL, &control_, &info);
   if( rc != SMF_OK )
   {
      Jnlst().Printf(J_ERROR, J_LINEAR_ALGEBRA,
         "smf: symbolic analysis failed with status %d.\n", rc);
      return SYMSOLVER_FATAL_ERROR;
   }

   numneg_ = 0;
   numdelay_ = 0;
   dynamic_active_ = false;
   rescale_ = false;
   pivtol_changed_ = false;

   return SYMSOLVER_SUCCESS;
}

ESymSolverStatus SmfSolverInterface::MultiSolve(
   bool         new_matrix,
   const Index* /*ia*/,
   const Index* /*ja*/,
   Index        nrhs,
   Number*      rhs_vals,
   bool         check_NegEVals,
   Index        numberOfNegEVals
)
{
   smf_ipm_info_t info;

   if( new_matrix || pivtol_changed_ )
   {
      smf_ipm_control_t fc = control_;
      double* scale_ptr = NULL;

      if( scaling_mode_ == SCALE_MODE_DYNAMIC )
      {
         if( dynamic_active_ )
         {
            scale_ptr = scaling_;
            if( rescale_ )
            {
               fc.scaling = SMF_SCALE_MATCHING;
               rescale_ = false;
            }
            else
            {
               fc.scaling = SMF_SCALE_NONE;
            }
         }
         else
         {
            fc.scaling = SMF_SCALE_NONE;
         }
      }

      const int rc = smf_ipm_factor_fn_(session_, SMF_INDEF, val_, &fc, &info, scale_ptr);

      /* Always update numneg_ BEFORE inertia checks (immune to coin-or/Ipopt#870). */
      numneg_ = info.num_neg;
      numdelay_ = info.num_delay;
      pivtol_changed_ = false;

      if( rc == SMF_ERR_SINGULAR || rc == SMF_ERR_INEXACT_INERTIA )
      {
         Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
            "smf: matrix singular or inexact inertia (rc=%d, zero=%d, static=%d).\n",
            rc, info.num_zero, info.static_pivots);
         return SYMSOLVER_SINGULAR;
      }
      if( rc == SMF_ERR_WRONG_INERTIA )
      {
         return SYMSOLVER_WRONG_INERTIA;
      }
      if( rc != SMF_OK )
      {
         Jnlst().Printf(J_ERROR, J_LINEAR_ALGEBRA,
            "smf: factorization failed with status %d.\n", rc);
         return SYMSOLVER_FATAL_ERROR;
      }

      if( scaling_mode_ == SCALE_MODE_DYNAMIC && !dynamic_active_ &&
          info.num_delay > static_cast<int>(0.05 * ndim_) )
      {
         dynamic_active_ = true;
         rescale_ = true;
      }

      if( check_NegEVals && numberOfNegEVals != numneg_ )
      {
         Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
            "smf: wrong inertia (expected %d negative eigenvalues, got %d).\n",
            numberOfNegEVals, numneg_);
         return SYMSOLVER_WRONG_INERTIA;
      }
   }

   const int rc = smf_ipm_solve_fn_(session_, SMF_SOLVE_FULL, nrhs, rhs_vals, ndim_, &control_, &info);
   if( rc != SMF_OK )
   {
      Jnlst().Printf(J_ERROR, J_LINEAR_ALGEBRA,
         "smf: solve failed with status %d.\n", rc);
      return SYMSOLVER_FATAL_ERROR;
   }

   return SYMSOLVER_SUCCESS;
}

bool SmfSolverInterface::IncreaseQuality()
{
   if( scaling_mode_ == SCALE_MODE_DYNAMIC && !dynamic_active_ )
   {
      dynamic_active_ = true;
      rescale_ = true;
      pivtol_changed_ = true;
      Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
         "smf: activating dynamic matching scaling to increase solution quality.\n");
      return true;
   }

   if( control_.u < umax_ )
   {
      const double next_u = (control_.u <= 0.0) ? Min(umax_, 1e-8)
                                                : Min(umax_, std::pow(control_.u, 0.75));
      if( next_u > control_.u )
      {
         Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
            "smf: increasing pivot threshold smf_u from %.2e to %.2e.\n",
            control_.u, next_u);
         control_.u = next_u;
         pivtol_changed_ = true;
         return true;
      }
   }

   if( scaling_mode_ == SCALE_MODE_DYNAMIC && !rescale_ )
   {
      rescale_ = true;
      pivtol_changed_ = true;
      Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
         "smf: triggering fresh matching scaling to increase solution quality.\n");
      return true;
   }

   return false;
}

} // namespace Ipopt
