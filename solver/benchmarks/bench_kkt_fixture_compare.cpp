// bench_kkt_fixture_compare.cpp -- compare smf and MA27 on one captured IPOPT KKT fixture.
//
// Fixture format:
//   matrix: Matrix Market coordinate real symmetric, lower triangle is enough
//   rhs   : text file with first line "n nrhs", followed by n*nrhs values
//
// Capture from the IPOPT MA97 plugin with:
//   SMF_MA97_CAPTURE_DIR=solver/build/kkt_captures SMF_MA97_CAPTURE_LIMIT=1 <trajectory-test>

#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/info.hpp"
#include "smf/matrix_market.hpp"
#include "smf/solver.hpp"
#include "smf/types.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#ifdef SMF_HAS_MA27
extern "C" {
    void ma27id_(int* icntl, double* cntl);
    void ma27ad_(int* n,
                 int* nz,
                 int* irn,
                 int* icn,
                 int* iw,
                 int* liw,
                 int* ikeep,
                 int* iw1,
                 int* nsteps,
                 int* iflag,
                 int* icntl,
                 double* cntl,
                 int* info,
                 double* ops);
    void ma27bd_(int* n,
                 int* nz,
                 int* irn,
                 int* icn,
                 double* a,
                 int* la,
                 int* iw,
                 int* liw,
                 int* ikeep,
                 int* nsteps,
                 int* maxfrt,
                 int* iw1,
                 int* icntl,
                 double* cntl,
                 int* info);
    void ma27cd_(int* n,
                 double* a,
                 int* la,
                 int* iw,
                 int* liw,
                 double* w,
                 int* maxfrt,
                 double* rhs,
                 int* iw1,
                 int* nsteps,
                 int* icntl,
                 int* info);
}
#endif

using Clock = std::chrono::steady_clock;

struct RhsData {
    int n = 0;
    int nrhs = 0;
    std::vector<double> values;
};

struct Result {
    bool ok = false;
    int status = 0;
    int pos = 0;
    int neg = 0;
    int zero = 0;
    double residual = -1.0;
    double analyse_ms = 0.0;
    double factor_ms = 0.0;
    double solve_ms = 0.0;
    long factor_entries = 0;
    double flops = 0.0;
    int delayed = 0;
    int maxfront = 0;
    int nsteps = 0;
    int raw_info15_neg = -1;
};

struct SmfProfile {
    bool ok = false;
    int repeats = 0;
    double first_full_ms = 0.0;
    double first_forward_ms = 0.0;
    double first_diag_ms = 0.0;
    double first_backward_ms = 0.0;
    double first_diagback_ms = 0.0;
    double full_ms = 0.0;
    double forward_ms = 0.0;
    double diag_ms = 0.0;
    double backward_ms = 0.0;
    double diagback_ms = 0.0;
    double residual = -1.0;
    int nsteps = 0;
    int width1 = 0;
    int width2 = 0;
    int width3plus = 0;
    int maxfront = 0;
    long long factor_values = 0;
};

static double elapsed_ms(Clock::time_point t0, Clock::time_point t1)
{
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static RhsData read_rhs(const std::string& path)
{
    RhsData rhs;
    std::ifstream in(path);
    if (!in)
        return rhs;
    in >> rhs.n >> rhs.nrhs;
    if (rhs.n <= 0 || rhs.nrhs <= 0)
        return RhsData{};
    rhs.values.resize(static_cast<std::size_t>(rhs.n) * static_cast<std::size_t>(rhs.nrhs));
    for (double& value : rhs.values)
    {
        if (!(in >> value))
            return RhsData{};
    }
    return rhs;
}

static void spmv_sym(const smf::CscLower& A, const double* x, double* y)
{
    const int n = static_cast<int>(A.n);
    std::fill(y, y + n, 0.0);
    for (int j = 0; j < n; ++j)
    {
        for (smf::Int p = A.col_ptr[static_cast<std::size_t>(j)]; p < A.col_ptr[static_cast<std::size_t>(j + 1)]; ++p)
        {
            const int i = static_cast<int>(A.row_idx[static_cast<std::size_t>(p)]);
            const double value = A.values[static_cast<std::size_t>(p)];
            y[i] += value * x[j];
            if (i != j)
                y[j] += value * x[i];
        }
    }
}

static double vector_norm2(const std::vector<double>& values)
{
    long double sum = 0.0;
    for (double value : values)
        sum += static_cast<long double>(value) * static_cast<long double>(value);
    return std::sqrt(static_cast<double>(sum));
}

static double matrix_frobenius_norm(const smf::CscLower& A)
{
    long double sum = 0.0;
    for (int j = 0; j < A.n; ++j)
    {
        for (smf::Int p = A.col_ptr[static_cast<std::size_t>(j)]; p < A.col_ptr[static_cast<std::size_t>(j + 1)]; ++p)
        {
            const int i = static_cast<int>(A.row_idx[static_cast<std::size_t>(p)]);
            const double value = A.values[static_cast<std::size_t>(p)];
            const long double term = static_cast<long double>(value) * static_cast<long double>(value);
            sum += (i == j) ? term : 2.0L * term;
        }
    }
    return std::sqrt(static_cast<double>(sum));
}

static double relative_residual(const smf::CscLower& A, const std::vector<double>& x, const std::vector<double>& b)
{
    std::vector<double> ax(static_cast<std::size_t>(A.n), 0.0);
    spmv_sym(A, x.data(), ax.data());
    std::vector<double> r(ax.size(), 0.0);
    for (std::size_t i = 0; i < ax.size(); ++i)
        r[i] = ax[i] - b[i];

    const double numerator = vector_norm2(r);
    const double denominator = matrix_frobenius_norm(A) * vector_norm2(x) + vector_norm2(b);
    return denominator > 0.0 ? numerator / denominator : numerator;
}

static Result run_smf(const smf::CscLower& A, const std::vector<double>& b, double pivot_u, int nemin)
{
    Result result;
    smf::Control ctrl;
    ctrl.matrix_type = smf::MatrixType::RealSymmetricIndefinite;
    ctrl.pivot_u = pivot_u;
    ctrl.nemin = nemin;
    ctrl.small_pivot = 1e-20;
    ctrl.continue_on_singular = true;

    smf::Info info;
    smf::Solver solver;

    const auto t0 = Clock::now();
    auto ak = solver.analyse(A, ctrl, info);
    const auto t1 = Clock::now();
    if (!ak || info.status != smf::ErrorCode::Success)
    {
        result.status = -1;
        return result;
    }

    smf::FactorKeep fk;
    const auto t2 = Clock::now();
    const smf::FactorStatus fs = solver.factor(*ak, ctrl, info, fk);
    const auto t3 = Clock::now();
    result.status = static_cast<int>(fs);
    if (fs != smf::FactorStatus::Success && fs != smf::FactorStatus::MaxPivotDelays && fs != smf::FactorStatus::Singular)
        return result;

    std::vector<double> x = b;
    const auto t4 = Clock::now();
    const int rc = solver.solve(fk, ctrl, info, x.data(), static_cast<int>(A.n), 1, smf::SolveJob::Full);
    const auto t5 = Clock::now();
    if (rc != 0)
    {
        result.status = rc;
        return result;
    }

    result.ok = true;
    result.pos = info.num_positive;
    result.neg = info.num_negative;
    result.zero = info.num_zero;
    result.residual = relative_residual(A, x, b);
    result.analyse_ms = elapsed_ms(t0, t1);
    result.factor_ms = elapsed_ms(t2, t3);
    result.solve_ms = elapsed_ms(t4, t5);
    result.factor_entries = static_cast<long>(info.actual_factor_entries);
    result.flops = info.actual_flops;
    result.delayed = info.delayed_pivots;
    result.maxfront = info.max_front_size;
    result.nsteps = static_cast<int>(ak->supernodes.size());
    return result;
}

static SmfProfile run_smf_profile(const smf::CscLower& A, const std::vector<double>& b, double pivot_u, int repeats,
                                  int nemin)
{
    SmfProfile profile;
    profile.repeats = repeats;

    smf::Control ctrl;
    ctrl.matrix_type = smf::MatrixType::RealSymmetricIndefinite;
    ctrl.pivot_u = pivot_u;
    ctrl.nemin = nemin;
    ctrl.small_pivot = 1e-20;
    ctrl.continue_on_singular = true;

    smf::Info info;
    smf::Solver solver;
    auto ak = solver.analyse(A, ctrl, info);
    if (!ak || info.status != smf::ErrorCode::Success)
        return profile;

    smf::FactorKeep fk;
    const smf::FactorStatus fs = solver.factor(*ak, ctrl, info, fk);
    if (fs != smf::FactorStatus::Success && fs != smf::FactorStatus::MaxPivotDelays && fs != smf::FactorStatus::Singular)
        return profile;

    profile.nsteps = static_cast<int>(ak->supernodes.size());
    profile.maxfront = info.max_front_size;
    profile.factor_values = static_cast<long long>(fk.factor_values.size());
    for (const auto& sn : ak->supernodes)
    {
        const int width = sn.width();
        if (width == 1)
            ++profile.width1;
        else if (width == 2)
            ++profile.width2;
        else
            ++profile.width3plus;
    }

    const int n = static_cast<int>(A.n);
    std::vector<double> x(static_cast<std::size_t>(n));
    std::vector<double> y(static_cast<std::size_t>(n));
    std::vector<double> z(static_cast<std::size_t>(n));

    auto run_job = [&](smf::SolveJob job, std::vector<double>& values) {
        smf::Info local_info;
        return solver.solve(fk, ctrl, local_info, values.data(), n, 1, job);
    };

    double full_total = 0.0;
    double forward_total = 0.0;
    double diag_total = 0.0;
    double backward_total = 0.0;
    double diagback_total = 0.0;

    x = b;
    auto t0 = Clock::now();
    if (run_job(smf::SolveJob::Full, x) != 0)
        return profile;
    auto t1 = Clock::now();
    profile.first_full_ms = elapsed_ms(t0, t1);

    for (int r = 0; r < repeats; ++r)
    {
        x = b;
        t0 = Clock::now();
        if (run_job(smf::SolveJob::Full, x) != 0)
            return profile;
        t1 = Clock::now();
        full_total += elapsed_ms(t0, t1);
    }

    y = b;
    t0 = Clock::now();
    if (run_job(smf::SolveJob::Forward, y) != 0)
        return profile;
    t1 = Clock::now();
    profile.first_forward_ms = elapsed_ms(t0, t1);

    z = y;
    t0 = Clock::now();
    if (run_job(smf::SolveJob::DiagOnly, z) != 0)
        return profile;
    t1 = Clock::now();
    profile.first_diag_ms = elapsed_ms(t0, t1);

    t0 = Clock::now();
    if (run_job(smf::SolveJob::Backward, z) != 0)
        return profile;
    t1 = Clock::now();
    profile.first_backward_ms = elapsed_ms(t0, t1);

    z = y;
    t0 = Clock::now();
    if (run_job(smf::SolveJob::DiagBack, z) != 0)
        return profile;
    t1 = Clock::now();
    profile.first_diagback_ms = elapsed_ms(t0, t1);

    for (int r = 0; r < repeats; ++r)
    {
        y = b;
        t0 = Clock::now();
        if (run_job(smf::SolveJob::Forward, y) != 0)
            return profile;
        t1 = Clock::now();
        forward_total += elapsed_ms(t0, t1);

        z = y;
        t0 = Clock::now();
        if (run_job(smf::SolveJob::DiagOnly, z) != 0)
            return profile;
        t1 = Clock::now();
        diag_total += elapsed_ms(t0, t1);

        t0 = Clock::now();
        if (run_job(smf::SolveJob::Backward, z) != 0)
            return profile;
        t1 = Clock::now();
        backward_total += elapsed_ms(t0, t1);

        z = y;
        t0 = Clock::now();
        if (run_job(smf::SolveJob::DiagBack, z) != 0)
            return profile;
        t1 = Clock::now();
        diagback_total += elapsed_ms(t0, t1);
    }

    profile.ok = true;
    profile.full_ms = full_total / repeats;
    profile.forward_ms = forward_total / repeats;
    profile.diag_ms = diag_total / repeats;
    profile.backward_ms = backward_total / repeats;
    profile.diagback_ms = diagback_total / repeats;
    profile.residual = relative_residual(A, x, b);
    return profile;
}

#ifdef SMF_HAS_MA27
static Result run_ma27(const smf::CscLower& A, const std::vector<double>& b, double pivot_u)
{
    Result result;
    int n = static_cast<int>(A.n);
    int nnz = static_cast<int>(A.nnz());

    std::vector<int> irn(static_cast<std::size_t>(nnz));
    std::vector<int> icn(static_cast<std::size_t>(nnz));
    std::vector<double> a_in(static_cast<std::size_t>(nnz));
    int entry = 0;
    for (int j = 0; j < n; ++j)
    {
        for (smf::Int p = A.col_ptr[static_cast<std::size_t>(j)]; p < A.col_ptr[static_cast<std::size_t>(j + 1)]; ++p)
        {
            irn[static_cast<std::size_t>(entry)] = static_cast<int>(A.row_idx[static_cast<std::size_t>(p)]) + 1;
            icn[static_cast<std::size_t>(entry)] = j + 1;
            a_in[static_cast<std::size_t>(entry)] = A.values[static_cast<std::size_t>(p)];
            ++entry;
        }
    }

    int icntl[30];
    double cntl[5];
    ma27id_(icntl, cntl);
    icntl[0] = 0;
    icntl[1] = 0;
    cntl[0] = pivot_u;

    int liw = std::max(2 * nnz + 3 * n + 1, 1);
    std::vector<int> iw(static_cast<std::size_t>(liw));
    std::vector<int> ikeep(static_cast<std::size_t>(std::max(3 * n, 1)));
    std::vector<int> iw1(static_cast<std::size_t>(std::max(2 * n, 1)));
    int nsteps = 0;
    int iflag = 0;
    int info[20] = {};
    double ops = 0.0;

    const auto t0 = Clock::now();
    ma27ad_(&n, &nnz, irn.data(), icn.data(), iw.data(), &liw, ikeep.data(), iw1.data(), &nsteps, &iflag, icntl,
            cntl, info, &ops);
    const auto t1 = Clock::now();
    result.status = info[0];
    if (info[0] < 0)
        return result;

    const int min_la = std::max(info[3], nnz + 1);
    const int min_liw = std::max(info[4], liw);
    int la = 0;
    int maxfrt = 0;
    std::vector<double> a_fac;

    const auto t2 = Clock::now();
    for (int attempt = 0; attempt < 5; ++attempt)
    {
        const double grow = 2.0 + static_cast<double>(attempt);
        la = std::max(nnz + 1, static_cast<int>(grow * static_cast<double>(min_la)) + 2 * n + 200);
        const int liw_factor = std::max(liw, static_cast<int>(grow * static_cast<double>(min_liw)) + 2 * n + 200);
        if (liw_factor > static_cast<int>(iw.size()))
            iw.resize(static_cast<std::size_t>(liw_factor));
        liw = liw_factor;

        a_fac.assign(static_cast<std::size_t>(la), 0.0);
        std::copy(a_in.begin(), a_in.end(), a_fac.begin());
        std::fill(std::begin(info), std::end(info), 0);
        std::vector<int> iw1_bd(static_cast<std::size_t>(std::max(2 * n + nsteps + 100, 1)));
        ma27bd_(&n, &nnz, irn.data(), icn.data(), a_fac.data(), &la, iw.data(), &liw, ikeep.data(), &nsteps,
                &maxfrt, iw1_bd.data(), icntl, cntl, info);
        result.status = info[0];
        if (info[0] >= 0)
        {
            result.raw_info15_neg = info[14];
            break;
        }
        if (info[0] != -3 && info[0] != -4)
            return result;
    }
    const auto t3 = Clock::now();
    if (result.status < 0)
        return result;

    std::vector<double> x = b;
    std::vector<double> w(static_cast<std::size_t>(std::max(maxfrt, 1)));
    std::vector<int> iw1_cd(static_cast<std::size_t>(std::max(2 * n + nsteps + 100, 1)));
    std::fill(std::begin(info), std::end(info), 0);
    const auto t4 = Clock::now();
    ma27cd_(&n, a_fac.data(), &la, iw.data(), &liw, w.data(), &maxfrt, x.data(), iw1_cd.data(), &nsteps, icntl,
            info);
    const auto t5 = Clock::now();
    if (info[0] < 0)
    {
        result.status = info[0];
        return result;
    }

    const int neg = result.raw_info15_neg >= 0 ? result.raw_info15_neg : 0;
    result.ok = true;
    result.analyse_ms = elapsed_ms(t0, t1);
    result.factor_ms = elapsed_ms(t2, t3);
    result.solve_ms = elapsed_ms(t4, t5);
    result.neg = neg;
    result.zero = 0;
    result.pos = n - neg;
    result.residual = relative_residual(A, x, b);
    result.flops = ops;
    result.factor_entries = la;
    result.maxfront = maxfrt;
    result.nsteps = nsteps;
    return result;
}
#endif

static void print_result(const char* name, const Result& result)
{
    if (!result.ok)
    {
        std::printf("%-5s status=%d FAILED\n", name, result.status);
        return;
    }

    std::printf("%-5s status=%d inertia=(%d,%d,%d) residual=%.6e analyse=%.3fms factor=%.3fms solve=%.3fms",
                name, result.status, result.pos, result.neg, result.zero, result.residual, result.analyse_ms,
                result.factor_ms, result.solve_ms);
    std::printf(" entries=%ld flops=%.3e delayed=%d maxfront=%d nsteps=%d", result.factor_entries, result.flops,
                result.delayed, result.maxfront, result.nsteps);
    if (result.raw_info15_neg >= 0)
        std::printf(" ma27_info15_neg=%d", result.raw_info15_neg);
    std::printf("\n");
}

static void print_profile(const SmfProfile& profile)
{
    if (!profile.ok)
    {
        std::puts("smf_profile FAILED");
        return;
    }
    std::printf("smf_profile repeats=%d first_full=%.6fms full=%.6fms first_forward=%.6fms forward=%.6fms first_diag=%.6fms diag=%.6fms first_backward=%.6fms backward=%.6fms first_diagback=%.6fms diagback=%.6fms residual=%.6e\n",
                profile.repeats, profile.first_full_ms, profile.full_ms, profile.first_forward_ms,
                profile.forward_ms, profile.first_diag_ms, profile.diag_ms, profile.first_backward_ms,
                profile.backward_ms, profile.first_diagback_ms, profile.diagback_ms, profile.residual);
    std::printf("smf_structure nsteps=%d width1=%d width2=%d width3plus=%d maxfront=%d factor_values=%lld\n",
                profile.nsteps, profile.width1, profile.width2, profile.width3plus, profile.maxfront,
                profile.factor_values);
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::fprintf(stderr, "usage: %s <matrix.mtx> <rhs.rhs> [pivot_u] [profile_repeats] [nemin]\n", argv[0]);
        return 2;
    }

    const std::string matrix_path = argv[1];
    const std::string rhs_path = argv[2];
    const double pivot_u = argc >= 4 ? std::atof(argv[3]) : 1e-8;
    int profile_repeats = 0;
    if (argc >= 5)
        profile_repeats = std::max(0, std::atoi(argv[4]));
    const int nemin = argc >= 6 ? std::max(1, std::atoi(argv[5])) : 8;

    smf::MatrixMarketResult loaded = smf::read_matrix_market(matrix_path);
    if (!loaded.ok)
    {
        std::fprintf(stderr, "failed to read matrix: %s\n", loaded.error_message.c_str());
        return 1;
    }

    RhsData rhs_data = read_rhs(rhs_path);
    if (rhs_data.n != loaded.matrix.n || rhs_data.nrhs <= 0)
    {
        std::fprintf(stderr, "failed to read compatible RHS from %s\n", rhs_path.c_str());
        return 1;
    }

    std::vector<double> b(static_cast<std::size_t>(rhs_data.n));
    std::copy(rhs_data.values.begin(), rhs_data.values.begin() + rhs_data.n, b.begin());

    std::printf("fixture matrix=%s rhs=%s\n", matrix_path.c_str(), rhs_path.c_str());
    std::printf("n=%d nnz_lower=%d nrhs=%d pivot_u=%.3e nemin=%d normA_fro=%.6e normb=%.6e\n", loaded.matrix.n,
                static_cast<int>(loaded.matrix.nnz()), rhs_data.nrhs, pivot_u, nemin,
                matrix_frobenius_norm(loaded.matrix), vector_norm2(b));

    Result smf_result = run_smf(loaded.matrix, b, pivot_u, nemin);
    print_result("smf", smf_result);

#ifdef SMF_HAS_MA27
    Result ma27_result = run_ma27(loaded.matrix, b, pivot_u);
    print_result("ma27", ma27_result);
    if (smf_result.ok && ma27_result.ok)
    {
        std::printf("compare inertia_match=%s residual_ratio_smf_over_ma27=%.6e factor_speedup_smf_over_ma27=%.6e solve_speedup_smf_over_ma27=%.6e\n",
                    (smf_result.neg == ma27_result.neg && smf_result.zero == ma27_result.zero) ? "yes" : "no",
                    ma27_result.residual > 0.0 ? smf_result.residual / ma27_result.residual : -1.0,
                    smf_result.factor_ms > 0.0 ? ma27_result.factor_ms / smf_result.factor_ms : -1.0,
                    smf_result.solve_ms > 0.0 ? ma27_result.solve_ms / smf_result.solve_ms : -1.0);
    }
#else
    std::puts("ma27 not compiled in: configure with CoinHSL/MA27 available for oracle comparison");
#endif

    if (profile_repeats > 0)
        print_profile(run_smf_profile(loaded.matrix, b, pivot_u, profile_repeats, nemin));

    return smf_result.ok ? 0 : 1;
}
