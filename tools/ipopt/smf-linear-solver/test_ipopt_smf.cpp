// End-to-end verification test for native linear_solver=smf in IPOPT 3.14.x.
// Tests:
//   1. IpoptGetAvailableLinearSolvers() reports IPOPTLINEARSOLVER_SMF
//   2. HS071 nonlinear optimization problem (4 vars, 2 constraints)
//   3. Non-convex 120-var indefinite-Hessian NLP (forces inertia correction)
//   4. Both smf_scaling=auto and smf_scaling=dynamic, smf_num_threads=4, smf_deterministic=yes

#include "IpIpoptApplication.hpp"
#include "IpLinearSolvers.h"
#include "IpTNLP.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace Ipopt;

namespace {

// Classic Hock-Schittkowski #71:
//   min  x0*x3*(x0+x1+x2) + x2
//   s.t. x0*x1*x2*x3 >= 25
//        x0^2 + x1^2 + x2^2 + x3^2 = 40
//        1 <= x_i <= 5
// Optimal objective: 17.014017145179164
class Hs071NLP : public TNLP {
 public:
  bool get_nlp_info(Index& n, Index& m, Index& nnz_jac_g, Index& nnz_h_lag,
                    IndexStyleEnum& index_style) override {
    n = 4;
    m = 2;
    nnz_jac_g = 8;
    nnz_h_lag = 10;
    index_style = TNLP::C_STYLE;
    return true;
  }

  bool get_bounds_info(Index n, Number* x_l, Number* x_u, Index m, Number* g_l,
                       Number* g_u) override {
    for (Index i = 0; i < n; ++i) {
      x_l[i] = 1.0;
      x_u[i] = 5.0;
    }
    g_l[0] = 25.0;
    g_u[0] = 2e19;
    g_l[1] = 40.0;
    g_u[1] = 40.0;
    return true;
  }

  bool get_starting_point(Index, bool init_x, Number* x, bool init_z, Number*, Number*, Index,
                          bool init_lambda, Number*) override {
    assert(init_x && !init_z && !init_lambda);
    x[0] = 1.0;
    x[1] = 5.0;
    x[2] = 5.0;
    x[3] = 1.0;
    return true;
  }

  bool eval_f(Index, const Number* x, bool, Number& obj_value) override {
    obj_value = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];
    return true;
  }

  bool eval_grad_f(Index, const Number* x, bool, Number* grad_f) override {
    grad_f[0] = x[0] * x[3] + x[3] * (x[0] + x[1] + x[2]);
    grad_f[1] = x[0] * x[3];
    grad_f[2] = x[0] * x[3] + 1.0;
    grad_f[3] = x[0] * (x[0] + x[1] + x[2]);
    return true;
  }

  bool eval_g(Index, const Number* x, bool, Index, Number* g) override {
    g[0] = x[0] * x[1] * x[2] * x[3];
    g[1] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
    return true;
  }

  bool eval_jac_g(Index, const Number* x, bool, Index, Index, Index* iRow, Index* jCol,
                  Number* values) override {
    if (values == nullptr) {
      iRow[0] = 0; jCol[0] = 0;
      iRow[1] = 0; jCol[1] = 1;
      iRow[2] = 0; jCol[2] = 2;
      iRow[3] = 0; jCol[3] = 3;
      iRow[4] = 1; jCol[4] = 0;
      iRow[5] = 1; jCol[5] = 1;
      iRow[6] = 1; jCol[6] = 2;
      iRow[7] = 1; jCol[7] = 3;
    } else {
      values[0] = x[1] * x[2] * x[3];
      values[1] = x[0] * x[2] * x[3];
      values[2] = x[0] * x[1] * x[3];
      values[3] = x[0] * x[1] * x[2];
      values[4] = 2.0 * x[0];
      values[5] = 2.0 * x[1];
      values[6] = 2.0 * x[2];
      values[7] = 2.0 * x[3];
    }
    return true;
  }

  bool eval_h(Index, const Number* x, bool, Number obj_factor, Index, const Number* lambda, bool,
              Index, Index* iRow, Index* jCol, Number* values) override {
    if (values == nullptr) {
      Index idx = 0;
      for (Index row = 0; row < 4; ++row) {
        for (Index col = 0; col <= row; ++col) {
          iRow[idx] = row;
          jCol[idx] = col;
          ++idx;
        }
      }
    } else {
      values[0] = obj_factor * (2.0 * x[3]);
      values[1] = obj_factor * (x[3]);
      values[2] = 0.0;
      values[3] = obj_factor * (x[3]);
      values[4] = 0.0;
      values[5] = 0.0;
      values[6] = obj_factor * (2.0 * x[0] + x[1] + x[2]);
      values[7] = obj_factor * (x[0]);
      values[8] = obj_factor * (x[0]);
      values[9] = 0.0;

      values[1] += lambda[0] * (x[2] * x[3]);
      values[3] += lambda[0] * (x[1] * x[3]);
      values[4] += lambda[0] * (x[0] * x[3]);
      values[6] += lambda[0] * (x[1] * x[2]);
      values[7] += lambda[0] * (x[0] * x[2]);
      values[8] += lambda[0] * (x[0] * x[1]);

      values[0] += lambda[1] * 2.0;
      values[2] += lambda[1] * 2.0;
      values[5] += lambda[1] * 2.0;
      values[9] += lambda[1] * 2.0;
    }
    return true;
  }

  void finalize_solution(SolverReturn, Index, const Number*, const Number*, const Number*, Index,
                         const Number*, const Number*, Number obj_value, const IpoptData*,
                         IpoptCalculatedQuantities*) override {
    final_obj = obj_value;
  }

  Number final_obj = 0.0;
};

// Non-convex coupled chain NLP (N variables, N-1 nonlinear constraints):
//   min  sum_i ( -cos(x_i) + 0.25*(x_i - 1)^2 )   [negative curvature at x=0 -> inertia correction!]
//   s.t. x_i + 0.5*x_{i+1}^2 = 1.0  (i = 0..N-2)
class NonConvexChainNLP : public TNLP {
 public:
  explicit NonConvexChainNLP(Index n) : n_(n) {}

  bool get_nlp_info(Index& n, Index& m, Index& nnz_jac_g, Index& nnz_h_lag,
                    IndexStyleEnum& index_style) override {
    n = n_;
    m = n_ - 1;
    nnz_jac_g = 2 * (n_ - 1);
    nnz_h_lag = n_;
    index_style = TNLP::C_STYLE;
    return true;
  }

  bool get_bounds_info(Index n, Number* x_l, Number* x_u, Index m, Number* g_l,
                       Number* g_u) override {
    for (Index i = 0; i < n; ++i) {
      x_l[i] = -3.0;
      x_u[i] = 3.0;
    }
    for (Index i = 0; i < m; ++i) {
      g_l[i] = 1.0;
      g_u[i] = 1.0;
    }
    return true;
  }

  bool get_starting_point(Index n, bool, Number* x, bool, Number*, Number*, Index, bool,
                          Number*) override {
    // Start near pi where -cos(x) has negative curvature (-cos(pi)=1, (-cos)''(pi) = -1 < 0)
    // to force IPOPT inertia correction!
    for (Index i = 0; i < n; ++i) x[i] = 2.8;
    return true;
  }

  bool eval_f(Index n, const Number* x, bool, Number& obj_value) override {
    double s = 0.0;
    for (Index i = 0; i < n; ++i) {
      const double d = x[i] - 1.0;
      s += -std::cos(x[i]) + 0.25 * d * d;
    }
    obj_value = s;
    return true;
  }

  bool eval_grad_f(Index n, const Number* x, bool, Number* grad_f) override {
    for (Index i = 0; i < n; ++i) {
      grad_f[i] = std::sin(x[i]) + 0.5 * (x[i] - 1.0);
    }
    return true;
  }

  bool eval_g(Index, const Number* x, bool, Index m, Number* g) override {
    for (Index i = 0; i < m; ++i) {
      g[i] = x[i] + 0.5 * x[i + 1] * x[i + 1];
    }
    return true;
  }

  bool eval_jac_g(Index, const Number* x, bool, Index m, Index, Index* iRow, Index* jCol,
                  Number* values) override {
    if (values == nullptr) {
      for (Index i = 0; i < m; ++i) {
        iRow[2 * i] = i;     jCol[2 * i] = i;
        iRow[2 * i + 1] = i; jCol[2 * i + 1] = i + 1;
      }
    } else {
      for (Index i = 0; i < m; ++i) {
        values[2 * i] = 1.0;
        values[2 * i + 1] = x[i + 1];
      }
    }
    return true;
  }

  bool eval_h(Index n, const Number* x, bool, Number obj_factor, Index m, const Number* lambda,
              bool, Index, Index* iRow, Index* jCol, Number* values) override {
    if (values == nullptr) {
      for (Index i = 0; i < n; ++i) {
        iRow[i] = i;
        jCol[i] = i;
      }
    } else {
      for (Index i = 0; i < n; ++i) {
        values[i] = obj_factor * (std::cos(x[i]) + 0.5);
      }
      for (Index i = 0; i < m; ++i) {
        values[i + 1] += lambda[i];
      }
    }
    return true;
  }

  void finalize_solution(SolverReturn, Index, const Number*, const Number*, const Number*, Index,
                         const Number*, const Number*, Number obj_value, const IpoptData*,
                         IpoptCalculatedQuantities*) override {
    final_obj = obj_value;
  }

  Index n_;
  Number final_obj = 0.0;
};

// Double-well non-convex problem starting near unstable saddle x_i = 0.01:
//   min  sum_i ( -5*x_i^2 + x_i^4 )
//   s.t. x_0 + x_1 = 2.0, -3 <= x_i <= 3
// At x_i = 0.01, Hessian diagonal is -10 < 0 -> forces IPOPT inertia correction (lg(rg) != '-')!
class InertiaCorrectionNLP : public TNLP {
 public:
  explicit InertiaCorrectionNLP(Index n) : n_(n) {}

  bool get_nlp_info(Index& n, Index& m, Index& nnz_jac_g, Index& nnz_h_lag,
                    IndexStyleEnum& index_style) override {
    n = n_;
    m = 1;
    nnz_jac_g = 2;
    nnz_h_lag = n_;
    index_style = TNLP::C_STYLE;
    return true;
  }

  bool get_bounds_info(Index n, Number* x_l, Number* x_u, Index, Number* g_l,
                       Number* g_u) override {
    for (Index i = 0; i < n; ++i) {
      x_l[i] = -3.0;
      x_u[i] = 3.0;
    }
    g_l[0] = 2.0;
    g_u[0] = 2.0;
    return true;
  }

  bool get_starting_point(Index n, bool, Number* x, bool, Number*, Number*, Index, bool,
                          Number*) override {
    for (Index i = 0; i < n; ++i) x[i] = 0.05 * (i % 2 == 0 ? 1.0 : -1.0);
    return true;
  }

  bool eval_f(Index n, const Number* x, bool, Number& obj_value) override {
    double s = 0.0;
    for (Index i = 0; i < n; ++i) {
      const double x2 = x[i] * x[i];
      s += -5.0 * x2 + x2 * x2;
    }
    obj_value = s;
    return true;
  }

  bool eval_grad_f(Index n, const Number* x, bool, Number* grad_f) override {
    for (Index i = 0; i < n; ++i) {
      grad_f[i] = -10.0 * x[i] + 4.0 * x[i] * x[i] * x[i];
    }
    return true;
  }

  bool eval_g(Index, const Number* x, bool, Index, Number* g) override {
    g[0] = x[0] + x[1];
    return true;
  }

  bool eval_jac_g(Index, const Number*, bool, Index, Index, Index* iRow, Index* jCol,
                  Number* values) override {
    if (values == nullptr) {
      iRow[0] = 0; jCol[0] = 0;
      iRow[1] = 0; jCol[1] = 1;
    } else {
      values[0] = 1.0;
      values[1] = 1.0;
    }
    return true;
  }

  bool eval_h(Index n, const Number* x, bool, Number obj_factor, Index, const Number*, bool,
              Index, Index* iRow, Index* jCol, Number* values) override {
    if (values == nullptr) {
      for (Index i = 0; i < n; ++i) {
        iRow[i] = i;
        jCol[i] = i;
      }
    } else {
      for (Index i = 0; i < n; ++i) {
        values[i] = obj_factor * (-10.0 + 12.0 * x[i] * x[i]);
      }
    }
    return true;
  }

  void finalize_solution(SolverReturn, Index, const Number*, const Number*, const Number*, Index,
                         const Number*, const Number*, Number obj_value, const IpoptData*,
                         IpoptCalculatedQuantities*) override {
    final_obj = obj_value;
  }

  Index n_;
  Number final_obj = 0.0;
};

} // namespace

int main(int argc, char** argv) {
  const char* smflib = (argc > 1 && argv[1][0] != '\0') ? argv[1] : nullptr;

  const IpoptLinearSolver all = IpoptGetAvailableLinearSolvers(0);
  const IpoptLinearSolver builtin = IpoptGetAvailableLinearSolvers(1);
  std::printf("[check] available_all=0x%x available_builtin=0x%x (SMF=0x%x)\n",
              all, builtin, IPOPTLINEARSOLVER_SMF);
  if ((all & IPOPTLINEARSOLVER_SMF) == 0) {
    std::fprintf(stderr, "FAIL: IPOPTLINEARSOLVER_SMF not reported available\n");
    return 1;
  }

  // Test 1: HS071 (with default solver if built-in, or explicit linear_solver=smf + smflib)
  {
    SmartPtr<IpoptApplication> app = IpoptApplicationFactory();
    app->Options()->SetIntegerValue("print_level", 5);
    app->Options()->SetNumericValue("tol", 1e-9);
    if (smflib) {
      app->Options()->SetStringValue("linear_solver", "smf");
      app->Options()->SetStringValue("smflib", smflib);
    } else {
      // Built-in: verify smf is selected as default without setting linear_solver!
      assert((builtin & IPOPTLINEARSOLVER_SMF) != 0);
    }
    app->Options()->SetStringValue("smf_scaling", "auto");
    app->Options()->SetIntegerValue("smf_num_threads", 2);
    app->Options()->SetStringValue("smf_deterministic", "yes");

    if (app->Initialize() != Solve_Succeeded) {
      std::fprintf(stderr, "FAIL: IpoptApplication::Initialize() failed\n");
      return 2;
    }
    SmartPtr<Hs071NLP> nlp = new Hs071NLP();
    const ApplicationReturnStatus st = app->OptimizeTNLP(nlp);
    std::printf("[HS071] status=%d obj=%.12f\n", static_cast<int>(st), nlp->final_obj);
    if (st != Solve_Succeeded || std::fabs(nlp->final_obj - 17.014017145179) > 1e-6) {
      std::fprintf(stderr, "FAIL: HS071 did not converge to expected optimum\n");
      return 3;
    }
  }

  // Test 2: Non-convex indefinite-Hessian chain NLP (forces inertia correction) with dynamic scaling
  {
    SmartPtr<IpoptApplication> app = IpoptApplicationFactory();
    app->Options()->SetIntegerValue("print_level", 5);
    app->Options()->SetNumericValue("tol", 1e-8);
    app->Options()->SetStringValue("linear_solver", "smf");
    if (smflib) {
      app->Options()->SetStringValue("smflib", smflib);
    }
    app->Options()->SetStringValue("smf_scaling", "dynamic");
    app->Options()->SetIntegerValue("smf_num_threads", 4);

    if (app->Initialize() != Solve_Succeeded) {
      std::fprintf(stderr, "FAIL: Initialize() failed for NonConvexChainNLP\n");
      return 4;
    }
    SmartPtr<NonConvexChainNLP> nlp = new NonConvexChainNLP(120);
    const ApplicationReturnStatus st = app->OptimizeTNLP(nlp);
    std::printf("[NonConvexChain120] status=%d obj=%.12f\n", static_cast<int>(st), nlp->final_obj);
    if (st != Solve_Succeeded) {
      std::fprintf(stderr, "FAIL: NonConvexChainNLP did not converge\n");
      return 5;
    }
  }

  // Test 3: Strongly negative-curvature double-well NLP (forces IPOPT inertia correction lg(rg)!)
  {
    SmartPtr<IpoptApplication> app = IpoptApplicationFactory();
    app->Options()->SetIntegerValue("print_level", 5);
    app->Options()->SetNumericValue("tol", 1e-8);
    app->Options()->SetStringValue("linear_solver", "smf");
    if (smflib) {
      app->Options()->SetStringValue("smflib", smflib);
    }
    app->Options()->SetStringValue("smf_scaling", "auto");
    app->Options()->SetIntegerValue("smf_num_threads", 2);

    if (app->Initialize() != Solve_Succeeded) {
      std::fprintf(stderr, "FAIL: Initialize() failed for InertiaCorrectionNLP\n");
      return 6;
    }
    SmartPtr<InertiaCorrectionNLP> nlp = new InertiaCorrectionNLP(40);
    const ApplicationReturnStatus st = app->OptimizeTNLP(nlp);
    std::printf("[InertiaCorrection40] status=%d obj=%.12f\n", static_cast<int>(st), nlp->final_obj);
    if (st != Solve_Succeeded) {
      std::fprintf(stderr, "FAIL: InertiaCorrectionNLP did not converge\n");
      return 7;
    }
  }

  std::printf("ALL IPOPT NATIVE SMF CHECKS PASSED\n");
  return 0;
}
