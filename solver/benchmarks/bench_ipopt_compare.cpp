// bench_ipopt_compare.cpp — smf vs MA27 as IPOPT linear solvers.
//
// Defines a 1-D Poisson fitting NLP:
//   min  (1/2) * sum_i (u_i - y_i)^2
//   s.t. -u_{i-1} + 2*u_i - u_{i+1} = h^2   (i = 1 .. n-2)
//        -1 <= u_i <= 1
// where y_i = sin(pi * i/(n-1)),  h = 1/(n-1).
//
// Solves the problem twice: once with IPOPT+MA27, once with IPOPT+smf.
// Compares: iterations, total wall-clock time, and final objective.
//
// Build: cmake -DSMF_BUILD_BENCHMARKS=ON -DSMF_BUILD_IPOPT_BENCHMARK=ON

#include "smf/analysis.hpp"
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/info.hpp"
#include "smf/solver.hpp"
#include "smf/types.hpp"

// Real IPOPT headers (requires IPOPT installed)
#include <IpAlgBuilder.hpp>
#include <IpIpoptApplication.hpp>
#include <IpSolveStatistics.hpp>
#include <IpSparseSymLinearSolverInterface.hpp>
#include <IpStdAugSystemSolver.hpp>
#include <IpSymLinearSolver.hpp>
#include <IpTNLP.hpp>
#include <IpTNLPAdapter.hpp>
#include <IpTSymLinearSolver.hpp>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;

static inline double elapsed_ms(TimePoint t0, TimePoint t1)
{
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ============================================================================
// SmfSparseInterface — wraps smf::Solver as Ipopt::SparseSymLinearSolverInterface
// ============================================================================
class SmfSparseInterface : public Ipopt::SparseSymLinearSolverInterface
{
public:
    SmfSparseInterface() { ctrl_.matrix_type = smf::MatrixType::RealSymmetricIndefinite; }

    // -----------------------------------------------------------------------
    // AlgorithmStrategyObject: InitializeImpl
    // -----------------------------------------------------------------------
    bool InitializeImpl(const Ipopt::OptionsList & /*options*/, const std::string & /*prefix*/) override
    {
        return true;
    }

    // -----------------------------------------------------------------------
    // InitializeStructure — build CSC pattern + symbolic analysis
    // -----------------------------------------------------------------------
    Ipopt::ESymSolverStatus InitializeStructure(Ipopt::Index dim,
                                                Ipopt::Index nonzeros,
                                                const Ipopt::Index* ia,
                                                const Ipopt::Index* ja) override
    {
        n_ = dim;
        neg_ev_ = 0;
        info_ = smf::Info{};

        // Build lower-CSC pattern from 1-based COO (lower triangle only).
        build_pattern(dim, nonzeros, ia, ja);

        // Symbolic analysis.
        ak_ = solver_.analyse(mat_, ctrl_, info_);
        if (!ak_ || info_.status != smf::ErrorCode::Success)
            return Ipopt::SYMSOLVER_FATAL_ERROR;

        build_csc_to_cleaned_map();

        // Values buffer (filled by IPOPT via GetValuesArrayPtr before each
        // MultiSolve with new_matrix=true).
        values_buf_.assign(static_cast<std::size_t>(nonzeros), 0.0);
        nnz_ = nonzeros;

        return Ipopt::SYMSOLVER_SUCCESS;
    }

    // -----------------------------------------------------------------------
    // GetValuesArrayPtr — IPOPT writes matrix values here
    // -----------------------------------------------------------------------
    Ipopt::Number* GetValuesArrayPtr() override { return values_buf_.data(); }

    // -----------------------------------------------------------------------
    // MultiSolve — factorize (if new_matrix) and solve
    // -----------------------------------------------------------------------
    Ipopt::ESymSolverStatus MultiSolve(bool new_matrix,
                                       const Ipopt::Index* /*ia*/,
                                       const Ipopt::Index* /*ja*/,
                                       Ipopt::Index nrhs,
                                       Ipopt::Number* rhs_vals,
                                       bool check_NegEVals,
                                       Ipopt::Index numberOfNegEVals) override
    {
        if (!ak_)
            return Ipopt::SYMSOLVER_FATAL_ERROR;

        if (new_matrix)
        {
            // Push COO values from IPOPT's buffer → mat_ CSC → ak_->cleaned.
            fill_values(values_buf_.data());
            push_values_to_cleaned();

            const smf::FactorStatus fs = solver_.factor(*ak_, ctrl_, info_, fk_);
            if (fs == smf::FactorStatus::Singular)
                return Ipopt::SYMSOLVER_SINGULAR;
            if (fs != smf::FactorStatus::Success && fs != smf::FactorStatus::MaxPivotDelays)
                return Ipopt::SYMSOLVER_FATAL_ERROR;

            neg_ev_ = info_.num_negative;
        }

        if (check_NegEVals && neg_ev_ != numberOfNegEVals)
            return Ipopt::SYMSOLVER_WRONG_INERTIA;

        if (nrhs > 0)
        {
            const int rc = solver_.solve(fk_, ctrl_, info_, rhs_vals, n_, nrhs, smf::SolveJob::Full);
            if (rc != 0)
                return Ipopt::SYMSOLVER_FATAL_ERROR;
        }
        return Ipopt::SYMSOLVER_SUCCESS;
    }

    Ipopt::Index NumberOfNegEVals() const override { return neg_ev_; }
    bool IncreaseQuality() override { return false; }
    bool ProvidesInertia() const override { return true; }

    Ipopt::SparseSymLinearSolverInterface::EMatrixFormat MatrixFormat() const override
    {
        return Ipopt::SparseSymLinearSolverInterface::Triplet_Format; // 1-based lower-triangle COO
    }

    // Timing diagnostics
    const smf::Info &last_info() const { return info_; }

private:
    // -----------------------------------------------------------------------
    // Helpers (logic mirrors ipopt_adapter.cpp)
    // -----------------------------------------------------------------------
    void build_pattern(int n, int nnz, const int* irn, const int* jcn)
    {
        struct Entry
        {
            int row, col, orig;
        };
        std::vector<Entry> entries;
        entries.reserve(static_cast<std::size_t>(nnz));

        for (int k = 0; k < nnz; ++k)
        {
            const int r = irn[k] - 1;
            const int c = jcn[k] - 1;
            if (r >= 0 && c >= 0 && r < n && c < n && r >= c)
                entries.push_back({r, c, k});
        }

        std::sort(entries.begin(), entries.end(), [](const Entry &a, const Entry &b) noexcept
                  { return a.col != b.col ? a.col < b.col : a.row < b.row; });

        const int m = static_cast<int>(entries.size());
        mat_.n = n;
        mat_.col_ptr.assign(static_cast<std::size_t>(n + 1), 0);
        mat_.row_idx.resize(static_cast<std::size_t>(m));
        mat_.values.assign(static_cast<std::size_t>(m), 0.0);

        for (const auto &e : entries)
            mat_.col_ptr[static_cast<std::size_t>(e.col + 1)]++;
        for (int j = 0; j < n; ++j)
            mat_.col_ptr[static_cast<std::size_t>(j + 1)] += mat_.col_ptr[static_cast<std::size_t>(j)];
        for (int k = 0; k < m; ++k)
            mat_.row_idx[static_cast<std::size_t>(k)] = entries[static_cast<std::size_t>(k)].row;

        coo_to_csc_.resize(static_cast<std::size_t>(m));
        for (int k = 0; k < m; ++k)
            coo_to_csc_[static_cast<std::size_t>(k)] = entries[static_cast<std::size_t>(k)].orig;
    }

    void fill_values(const double* vals)
    {
        const int m = static_cast<int>(coo_to_csc_.size());
        for (int k = 0; k < m; ++k)
            mat_.values[static_cast<std::size_t>(k)] =
                vals[static_cast<std::size_t>(coo_to_csc_[static_cast<std::size_t>(k)])];
    }

    void build_csc_to_cleaned_map()
    {
        // ak_->cleaned stores the ORIGINAL-ordered matrix; factor() permutes
        // it internally. Search by original (old_j, old_i) — no iperm needed.
        const int n = n_;
        const int mat_nnz = static_cast<int>(mat_.values.size());

        csc_to_cleaned_.resize(static_cast<std::size_t>(mat_nnz), -1);

        for (int old_j = 0; old_j < n; ++old_j)
        {
            for (int p = mat_.col_ptr[static_cast<std::size_t>(old_j)];
                 p < mat_.col_ptr[static_cast<std::size_t>(old_j) + 1]; ++p)
            {
                const int old_i = mat_.row_idx[static_cast<std::size_t>(p)];

                const int cs = static_cast<int>(ak_->cleaned.col_ptr[static_cast<std::size_t>(old_j)]);
                const int ce = static_cast<int>(ak_->cleaned.col_ptr[static_cast<std::size_t>(old_j) + 1]);

                const auto beg = ak_->cleaned.row_idx.begin() + cs;
                const auto end = ak_->cleaned.row_idx.begin() + ce;
                const auto it = std::lower_bound(beg, end, old_i);

                csc_to_cleaned_[static_cast<std::size_t>(p)] = static_cast<int>(it - ak_->cleaned.row_idx.begin());
            }
        }
    }

    void push_values_to_cleaned()
    {
        std::fill(ak_->cleaned.values.begin(), ak_->cleaned.values.end(), 0.0);
        const int mat_nnz = static_cast<int>(mat_.values.size());
        for (int k = 0; k < mat_nnz; ++k)
        {
            const int pos = csc_to_cleaned_[static_cast<std::size_t>(k)];
            if (pos >= 0 && pos < static_cast<int>(ak_->cleaned.values.size()))
                ak_->cleaned.values[static_cast<std::size_t>(pos)] += mat_.values[static_cast<std::size_t>(k)];
        }
    }

    smf::Control ctrl_;
    smf::Solver solver_;
    smf::Info info_{};
    std::unique_ptr<smf::AnalysisKeep> ak_;
    smf::FactorKeep fk_;
    smf::CscLower mat_;
    std::vector<int> coo_to_csc_;
    std::vector<int> csc_to_cleaned_;
    std::vector<double> values_buf_;
    int n_{0};
    int nnz_{0};
    int neg_ev_{0};
};

// ============================================================================
// PoissonFitNLP — 1-D Poisson fitting problem
//
//   min  (1/2) * sum_i (u_i - y_i)^2
//   s.t. -u_{i-1} + 2*u_i - u_{i+1} = h^2    (i = 1..n-2)
//        -1 <= u_i <= 1
//   where y_i = sin(pi * i/(n-1)),  h = 1/(n-1)
// ============================================================================
class PoissonFitNLP : public Ipopt::TNLP
{
public:
    explicit PoissonFitNLP(int n) : n_(n), m_(n - 2), h_(1.0 / (n - 1))
    {
        assert(n >= 3);
        y_.resize(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i)
            y_[static_cast<std::size_t>(i)] = std::sin(M_PI * static_cast<double>(i) / static_cast<double>(n - 1));
    }

    // ------------------------------------------------------------------
    bool get_nlp_info(Ipopt::Index &n,
                      Ipopt::Index &m,
                      Ipopt::Index &nnz_jac_g,
                      Ipopt::Index &nnz_h_lag,
                      IndexStyleEnum &index_style) override
    {
        n = n_;
        m = m_;
        // Each constraint i touches u_{i-1}, u_i, u_{i+1}: 3 entries
        // (first and last constraints touch only 2 variables each, but we
        //  still count 3 for simplicity — boundary entries remain valid).
        nnz_jac_g = 3 * m_;
        nnz_h_lag = n_; // diagonal Hessian
        index_style = TNLP::C_STYLE;
        return true;
    }

    // ------------------------------------------------------------------
    bool get_bounds_info(Ipopt::Index n,
                         Ipopt::Number* x_l,
                         Ipopt::Number* x_u,
                         Ipopt::Index m,
                         Ipopt::Number* g_l,
                         Ipopt::Number* g_u) override
    {
        const double h2 = h_ * h_;
        for (int i = 0; i < n; ++i)
        {
            x_l[i] = -1.0;
            x_u[i] = 1.0;
        }
        for (int j = 0; j < m; ++j)
        {
            g_l[j] = h2;
            g_u[j] = h2;
        }
        return true;
    }

    // ------------------------------------------------------------------
    bool get_starting_point(Ipopt::Index n,
                            bool init_x,
                            Ipopt::Number* x,
                            bool /*init_z*/,
                            Ipopt::Number* /*z_L*/,
                            Ipopt::Number* /*z_U*/,
                            Ipopt::Index /*m*/,
                            bool /*init_lambda*/,
                            Ipopt::Number* /*lambda*/) override
    {
        if (init_x)
            for (int i = 0; i < n; ++i)
                x[i] = 0.0;
        return true;
    }

    // ------------------------------------------------------------------
    bool eval_f(Ipopt::Index n, const Ipopt::Number* x, bool /*new_x*/, Ipopt::Number &obj_value) override
    {
        obj_value = 0.0;
        for (int i = 0; i < n; ++i)
        {
            const double d = x[i] - y_[static_cast<std::size_t>(i)];
            obj_value += 0.5 * d * d;
        }
        return true;
    }

    // ------------------------------------------------------------------
    bool eval_grad_f(Ipopt::Index n, const Ipopt::Number* x, bool /*new_x*/, Ipopt::Number* grad_f) override
    {
        for (int i = 0; i < n; ++i)
            grad_f[i] = x[i] - y_[static_cast<std::size_t>(i)];
        return true;
    }

    // ------------------------------------------------------------------
    // g_j = -u_{j} + 2*u_{j+1} - u_{j+2}   for j = 0..m-1
    // (constraint i=j+1: j+1-th interior node, 0-indexed)
    bool
    eval_g(Ipopt::Index /*n*/, const Ipopt::Number* x, bool /*new_x*/, Ipopt::Index /*m*/, Ipopt::Number* g) override
    {
        for (int j = 0; j < m_; ++j)
            g[j] = -x[j] + 2.0 * x[j + 1] - x[j + 2];
        return true;
    }

    // ------------------------------------------------------------------
    bool eval_jac_g(Ipopt::Index /*n*/,
                    const Ipopt::Number* /*x*/,
                    bool /*new_x*/,
                    Ipopt::Index /*m*/,
                    Ipopt::Index nnz,
                    Ipopt::Index* iRow,
                    Ipopt::Index* jCol,
                    Ipopt::Number* vals) override
    {
        if (vals == nullptr)
        {
            // Structure
            int k = 0;
            for (int j = 0; j < m_; ++j)
            {
                iRow[k] = j;
                jCol[k] = j;
                ++k; // -1 coeff
                iRow[k] = j;
                jCol[k] = j + 1;
                ++k; // +2 coeff
                iRow[k] = j;
                jCol[k] = j + 2;
                ++k; // -1 coeff
            }
            (void)nnz;
        }
        else
        {
            int k = 0;
            for (int j = 0; j < m_; ++j)
            {
                vals[k++] = -1.0;
                vals[k++] = 2.0;
                vals[k++] = -1.0;
            }
            (void)nnz;
        }
        return true;
    }

    // ------------------------------------------------------------------
    // Hessian of Lagrangian = I (objective only; constraints are linear).
    bool eval_h(Ipopt::Index n,
                const Ipopt::Number* /*x*/,
                bool /*new_x*/,
                Ipopt::Number obj_factor,
                Ipopt::Index /*m*/,
                const Ipopt::Number* /*lambda*/,
                bool /*new_lambda*/,
                Ipopt::Index /*nnz*/,
                Ipopt::Index* iRow,
                Ipopt::Index* jCol,
                Ipopt::Number* vals) override
    {
        if (vals == nullptr)
        {
            for (int i = 0; i < n; ++i)
            {
                iRow[i] = i;
                jCol[i] = i;
            }
        }
        else
        {
            for (int i = 0; i < n; ++i)
                vals[i] = obj_factor;
        }
        return true;
    }

    // ------------------------------------------------------------------
    void finalize_solution(Ipopt::SolverReturn status,
                           Ipopt::Index n,
                           const Ipopt::Number* x,
                           const Ipopt::Number* /*z_L*/,
                           const Ipopt::Number* /*z_U*/,
                           Ipopt::Index /*m*/,
                           const Ipopt::Number* /*g*/,
                           const Ipopt::Number* /*lambda*/,
                           Ipopt::Number obj_value,
                           const Ipopt::IpoptData* /*ip_data*/,
                           Ipopt::IpoptCalculatedQuantities* /*ip_cq*/) override
    {
        result_status_ = status;
        result_obj_ = obj_value;
        result_x_.assign(x, x + n);
    }

    Ipopt::SolverReturn result_status() const { return result_status_; }
    double result_obj() const { return result_obj_; }

private:
    int n_;
    int m_;
    double h_;
    std::vector<double> y_;
    Ipopt::SolverReturn result_status_{Ipopt::UNASSIGNED};
    double result_obj_{0.0};
    std::vector<double> result_x_;
};

// ============================================================================
// RunResult — statistics from one IPOPT run
// ============================================================================
struct RunResult
{
    std::string solver_name;
    Ipopt::SolverReturn status{Ipopt::UNASSIGNED};
    int iterations{-1};
    double objective{0.0};
    double wall_ms{0.0};
    bool ok() const { return status == Ipopt::SUCCESS || status == Ipopt::STOP_AT_ACCEPTABLE_POINT; }
};

// ============================================================================
// make_quiet_app — build and silently initialize an IpoptApplication
// ============================================================================
static Ipopt::SmartPtr<Ipopt::IpoptApplication> make_quiet_app()
{
    auto app = IpoptApplicationFactory();
    app->Options()->SetIntegerValue("print_level", 0);
    app->Options()->SetStringValue("sb", "yes"); // suppress banner
    return app;
}

// ============================================================================
// run_ma27 — solve with standard IPOPT+MA27
// ============================================================================
static RunResult run_ma27(int n)
{
    RunResult r;
    r.solver_name = "MA27";

    auto app = make_quiet_app();
    app->Options()->SetStringValue("linear_solver", "ma27");

    if (app->Initialize() != Ipopt::Solve_Succeeded)
    {
        r.solver_name = "MA27 [init failed]";
        return r;
    }

    Ipopt::SmartPtr<PoissonFitNLP> tnlp = new PoissonFitNLP(n);

    auto t0 = Clock::now();
    Ipopt::ApplicationReturnStatus ret = app->OptimizeTNLP(tnlp);
    auto t1 = Clock::now();

    r.wall_ms = elapsed_ms(t0, t1);
    r.objective = tnlp->result_obj();
    r.status = tnlp->result_status();
    r.iterations = Ipopt::IsValid(app->Statistics()) ? static_cast<int>(app->Statistics()->IterationCount()) : -1;
    (void)ret;
    return r;
}

// ============================================================================
// run_smf — solve with IPOPT+smf via custom AlgorithmBuilder
// ============================================================================
static RunResult run_smf(int n)
{
    RunResult r;
    r.solver_name = "smf";

    auto app = make_quiet_app();
    // must still initialize (registers options, reads any ipopt.opt, etc.)
    if (app->Initialize() != Ipopt::Solve_Succeeded)
    {
        r.solver_name = "smf [init failed]";
        return r;
    }

    // Build the smf → TSymLinearSolver → StdAugSystemSolver chain.
    Ipopt::SmartPtr<SmfSparseInterface> iface = new SmfSparseInterface();
    Ipopt::SmartPtr<Ipopt::TSymLinearSolver> tsym = new Ipopt::TSymLinearSolver(iface, nullptr);
    Ipopt::SmartPtr<Ipopt::StdAugSystemSolver> aug = new Ipopt::StdAugSystemSolver(*tsym);

    Ipopt::SmartPtr<Ipopt::AlgorithmBuilder> alg_builder = new Ipopt::AlgorithmBuilder(aug, "smf");

    Ipopt::SmartPtr<PoissonFitNLP> tnlp = new PoissonFitNLP(n);
    // TNLPAdapter wraps TNLP → NLP so we can call OptimizeNLP with the builder.
    Ipopt::SmartPtr<Ipopt::TNLPAdapter> nlp_adapter = new Ipopt::TNLPAdapter(Ipopt::GetRawPtr(tnlp));

    auto t0 = Clock::now();
    Ipopt::ApplicationReturnStatus ret = app->OptimizeNLP(nlp_adapter, alg_builder);
    auto t1 = Clock::now();

    r.wall_ms = elapsed_ms(t0, t1);
    r.objective = tnlp->result_obj();
    r.status = tnlp->result_status();
    r.iterations = Ipopt::IsValid(app->Statistics()) ? static_cast<int>(app->Statistics()->IterationCount()) : -1;
    (void)ret;
    return r;
}

// ============================================================================
// main
// ============================================================================
int main()
{
    std::puts("=== bench_ipopt_compare: smf vs MA27 as IPOPT linear solvers ===");
    std::puts("Problem: 1-D Poisson fitting NLP  (y = sin(pi*x))");
    std::puts("");

    // Problem sizes to benchmark
    const int sizes[] = {100, 300, 600, 1000, 2000, 4000};

    std::printf("%-6s  %-8s  %5s  %-8s  %10s  %-8s  %5s  %-8s  %10s  %8s\n", "n", "ma27(ms)", "iter", "status",
                "obj(ma27)", "smf(ms)", "iter", "status", "obj(smf)", "speedup");
    std::printf("%-6s  %-8s  %5s  %-8s  %10s  %-8s  %5s  %-8s  %10s  %8s\n", "------", "--------", "-----", "--------",
                "----------", "--------", "-----", "--------", "----------", "--------");

    for (int n : sizes)
    {
        RunResult ra = run_ma27(n);
        RunResult rs = run_smf(n);

        const auto status_str = [](const RunResult &r) -> const char*
        {
            if (!r.ok())
                return "FAIL";
            if (r.status == Ipopt::SUCCESS)
                return "OK";
            return "accept.";
        };

        double speedup = (rs.wall_ms > 0.0 && ra.ok() && rs.ok()) ? ra.wall_ms / rs.wall_ms : 0.0;

        std::printf("%-6d  %8.2f  %5d  %-8s  %10.4e  %8.2f  %5d  %-8s  %10.4e  %8.3f\n", n, ra.wall_ms, ra.iterations,
                    status_str(ra), ra.objective, rs.wall_ms, rs.iterations, status_str(rs), rs.objective, speedup);
        std::fflush(stdout);
    }

    std::puts("");
    std::puts("speedup > 1 means smf is faster than MA27.");
    return 0;
}
