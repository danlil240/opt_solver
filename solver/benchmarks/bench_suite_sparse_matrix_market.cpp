// bench_suite_sparse_matrix_market.cpp — Matrix Market file benchmark for smf
// Part of the smf MA97-class solver benchmark suite.

#include "common/validation.hpp"
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/info.hpp"
#include "smf/matrix_market.hpp"
#include "smf/solver.hpp"
#include "smf/types.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char *argv[]) {
    std::printf("=== bench_suite_sparse_matrix_market ===\n");

    if (argc < 2) {
        std::fprintf(stderr, "Usage: bench_suite_sparse_matrix_market <path/to/matrix.mtx> [nrhs=1]\n");
        return 2;
    }

    const char *path = argv[1];
    int nrhs = 1;
    if (argc >= 3) {
        nrhs = std::atoi(argv[2]);
        if (nrhs < 1) {
            std::fprintf(stderr, "Invalid nrhs=%d\n", nrhs);
            return 2;
        }
    }

    smf::MatrixMarketResult loaded = smf::read_matrix_market(path);
    if (!loaded.ok) {
        std::fprintf(stderr, "bench_suite_sparse_matrix_market: failed to parse '%s': %s\n",
                     path, loaded.error_message.c_str());
        return 1;
    }

    const smf::CscLower &A = loaded.matrix;
    const int N = static_cast<int>(A.n);

    std::printf("File  : %s\n", path);
    std::printf("Matrix: N=%d nnz=%d symmetric=%s\n",
                N, static_cast<int>(A.nnz()), loaded.is_symmetric ? "yes" : "no (lower triangle)");
    std::printf("nrhs  : %d\n", nrhs);

    std::vector<double> b(static_cast<std::size_t>(N) * static_cast<std::size_t>(nrhs), 1.0);
    std::vector<double> x(b);

    smf::Control ctrl;
    ctrl.matrix_type = smf::MatrixType::RealSymmetricIndefinite;
    smf::Info info;
    smf::Solver solver;

    auto t0 = std::chrono::steady_clock::now();
    auto ak = solver.analyse(A, ctrl, info);
    auto t1 = std::chrono::steady_clock::now();
    if (!ak || info.status != smf::ErrorCode::Success) {
        std::fprintf(stderr, "bench_suite_sparse_matrix_market: analyse() failed\n");
        return 1;
    }
    double t_analyse = std::chrono::duration<double, std::milli>(t1 - t0).count();

    smf::FactorKeep fkeep;
    auto t2 = std::chrono::steady_clock::now();
    smf::FactorStatus fs = solver.factor(*ak, ctrl, info, fkeep);
    auto t3 = std::chrono::steady_clock::now();
    if (fs != smf::FactorStatus::Success) {
        std::fprintf(stderr, "bench_suite_sparse_matrix_market: factor() returned status %d\n",
                     static_cast<int>(fs));
        return 1;
    }
    double t_factor = std::chrono::duration<double, std::milli>(t3 - t2).count();

    auto t4 = std::chrono::steady_clock::now();
    int rc = solver.solve(fkeep, ctrl, info, x.data(), N, nrhs);
    auto t5 = std::chrono::steady_clock::now();
    if (rc != 0) {
        std::fprintf(stderr, "bench_suite_sparse_matrix_market: solve() returned %d\n", rc);
        return 1;
    }
    double t_solve = std::chrono::duration<double, std::milli>(t5 - t4).count();

    smf::bench::Validation val = smf::bench::validate(A, b, x, nrhs);
    smf::bench::Outcome outcome = smf::bench::qualify_outcome(smf::bench::Outcome::Success, val);

    std::printf("Analyse : %.3f ms\n", t_analyse);
    std::printf("Factor  : %.3f ms\n", t_factor);
    std::printf("Solve   : %.3f ms\n", t_solve);
    std::printf("eta_inf : %.3e\n", val.eta_inf);
    std::printf("omega   : %.3e\n", val.omega);
    std::printf("Residual: %.3e\n", val.relative_residual);
    std::printf("Outcome : %s\n", smf::bench::outcome_to_string(outcome));
    std::printf("Inertia : pos=%d neg=%d zero=%d\n",
                info.num_positive, info.num_negative, info.num_zero);

    int64_t peak_rss = smf::bench::read_peak_rss_bytes();
    if (peak_rss >= 0) {
        std::printf("Peak RSS: %lld kB\n", static_cast<long long>(peak_rss / 1024));
    } else {
        std::printf("Peak RSS: N/A\n");
    }

    return outcome == smf::bench::Outcome::Success ? 0 : 1;
}
