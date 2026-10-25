// In-process smf vs baseline profiler for fast optimisation iteration.
//
// Runs every case of a suite manifest through the same warm_sequence /
// cold protocol as bench_replay (1 analyse + 5 refactor/solve, or
// analyse+factor+solve), but in one process with best-of-R repeats so a
// whole suite finishes in seconds. Prints per-phase times, supernode
// statistics and solution quality side by side.
//
// Usage: bench_profile [--manifest path] [--solvers smf,ma27] [--repeats R]
//                      [--case substring] [-v] [--nemin k] [--ordering m]
//                      [--scaling s]
//        bench_profile --sample analyse|factor|solve|all --case substring
//                      [--seconds S] [--lines]
//
// The --sample mode loops one phase of one case through the smf adapter
// under an in-process sampling profiler (Windows; build with /Zi and link
// with /DEBUG for symbols) and prints the hottest functions or lines.

#include "adapters/adapter.hpp"
#include "smf/control.hpp"
#include "smf/matrix_market.hpp"
#include "smf/solver.hpp"
#include "common/sampler.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Case {
    std::string id;
    std::string mtx;
    bool spd = false;
    bool warm = false;
};

std::string json_field(const std::string &obj, const std::string &key) {
    const std::string pat = "\"" + key + "\"";
    auto p = obj.find(pat);
    if (p == std::string::npos) return {};
    p = obj.find(':', p);
    p = obj.find('"', p);
    const auto e = obj.find('"', p + 1);
    return obj.substr(p + 1, e - p - 1);
}

std::vector<Case> load_manifest(const std::string &path) {
    std::ifstream in(path);
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string s = ss.str();
    std::vector<Case> cases;
    std::size_t pos = s.find("\"cases\"");
    while (pos != std::string::npos) {
        const auto b = s.find('{', pos);
        if (b == std::string::npos) break;
        const auto e = s.find('}', b);
        const std::string obj = s.substr(b, e - b + 1);
        Case c;
        c.id = json_field(obj, "matrix_id");
        c.mtx = json_field(obj, "mtx_path");
        c.spd = json_field(obj, "matrix_type") == "spd";
        c.warm = json_field(obj, "mode") == "warm_sequence";
        if (!c.id.empty()) cases.push_back(c);
        pos = e + 1;
        if (s.find('{', pos) == std::string::npos) break;
    }
    return cases;
}

struct Timing {
    double analyse = 0, factor = 0, solve = 0, wall = 0;
    bool ok = false;
    double eta = 0;
    int delayed = 0;
    long long fent = 0;
    double flops = 0;
    std::string why;  // failure stage / accuracy, shown with -v
};

Timing run_once(smf::bench::SolverAdapter &ad, const smf::CscLower &A,
                const Case &c, const std::vector<std::vector<double>> &seq) {
    using Clock = std::chrono::steady_clock;
    Timing t;
    const int n = static_cast<int>(A.n);
    std::vector<double> b(static_cast<std::size_t>(n), 1.0);
    std::vector<double> x;
    const auto t0 = Clock::now();
    if (ad.analyse(A) != smf::bench::Outcome::Success) {
        t.why = "analyse: " + ad.stats().reason;
        return t;
    }
    t.analyse = ad.stats().analyse_ms;
    const int steps = c.warm ? static_cast<int>(seq.size()) : 1;
    for (int s = 0; s < steps; ++s) {
        const auto &vals = c.warm ? seq[static_cast<std::size_t>(s)] : A.values;
        if (ad.factor(vals) != smf::bench::Outcome::Success) {
            t.why = "factor: " + ad.stats().reason;
            return t;
        }
        t.factor += ad.stats().factor_ms;
        x = b;
        if (ad.solve(x, 1) != smf::bench::Outcome::Success) {
            t.why = "solve: " + ad.stats().reason;
            return t;
        }
        t.solve += ad.stats().solve_ms;
    }
    t.wall = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    smf::CscLower Al = A;
    if (c.warm) Al.values = seq.back();
    const auto v = smf::bench::validate(Al, b, x, 1);
    t.ok = smf::bench::is_qualified_validation(v);
    t.eta = v.eta_inf;
    if (!t.ok) {
        char buf[96];
        std::snprintf(buf, sizeof buf, "eta=%.2e omega=%.2e", v.eta_inf, v.omega);
        t.why = buf;
    }
    t.delayed = ad.stats().delayed_pivots;
    t.fent = ad.stats().factor_entries;
    t.flops = ad.stats().flops;
    return t;
}

} // namespace

int main(int argc, char **argv) {
    std::string manifest = "benchmarks/manifests/optim20.json";
    std::string solvers_arg = "smf,ma27";
    std::string filter;
    int repeats = 7;
    bool verbose = false;
    int nemin = -1, ordering = -1, scaling = -1;
    std::string sample_phase;
    double sample_seconds = 3.0;
    bool sample_lines = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-v") verbose = true;
        else if (a == "--nemin" && i + 1 < argc) nemin = std::atoi(argv[++i]);
        else if (a == "--ordering" && i + 1 < argc) ordering = std::atoi(argv[++i]);
        else if (a == "--scaling" && i + 1 < argc) scaling = std::atoi(argv[++i]);
        else if (a == "--sample" && i + 1 < argc) sample_phase = argv[++i];
        else if (a == "--seconds" && i + 1 < argc) sample_seconds = std::atof(argv[++i]);
        else if (a == "--lines") sample_lines = true;
        if (a == "--manifest" && i + 1 < argc) manifest = argv[++i];
        else if (a == "--solvers" && i + 1 < argc) solvers_arg = argv[++i];
        else if (a == "--repeats" && i + 1 < argc) repeats = std::atoi(argv[++i]);
        else if (a == "--case" && i + 1 < argc) filter = argv[++i];
    }
    std::vector<std::string> solvers;
    {
        std::stringstream ss(solvers_arg);
        std::string tok;
        while (std::getline(ss, tok, ',')) solvers.push_back(tok);
    }

    const auto cases = load_manifest(manifest);

#if defined(_WIN32)
    if (!sample_phase.empty()) {
        // Sampling mode: loop one phase of one case (smf only) under the
        // in-process sampler and print the hottest functions / lines.
        for (const auto &c : cases) {
            if (!filter.empty() && c.id.find(filter) == std::string::npos) continue;
            auto mm = smf::read_matrix_market(c.mtx);
            if (!mm.ok) return 1;
            const smf::CscLower A = std::move(mm.matrix);
            smf::Control ctrl;
            ctrl.matrix_type = c.spd ? smf::MatrixType::RealSymmetricPositiveDefinite
                                     : smf::MatrixType::RealSymmetricIndefinite;
            if (nemin > 0) ctrl.nemin = nemin;
            if (ordering >= 0) ctrl.ordering = static_cast<smf::OrderingMethod>(ordering);
            auto ad = smf::bench::make_solver_adapter("smf");
            ad->configure(ctrl, "recommended");
            std::vector<double> x(static_cast<std::size_t>(A.n), 1.0);
            ad->analyse(A);
            ad->factor(A.values);
            ad->solve(x, 1);
            smf::bench::Sampler sampler;
            long iters = 0;
            sampler.start();
            const auto t0 = std::chrono::steady_clock::now();
            while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() <
                   sample_seconds) {
                if (sample_phase == "analyse" || sample_phase == "all") ad->analyse(A);
                if (sample_phase == "factor" || sample_phase == "all") ad->factor(A.values);
                if (sample_phase == "solve" || sample_phase == "all") {
                    std::fill(x.begin(), x.end(), 1.0);
                    ad->solve(x, 1);
                }
                ++iters;
            }
            sampler.stop();
            std::printf("%s: %ld iterations of %s\n", c.id.c_str(), iters, sample_phase.c_str());
            sampler.report(sample_lines);
            return 0;
        }
        return 1;
    }
#endif
    std::printf("%-22s %5s %4s", "case", "n", "nsn");
    for (const auto &sv : solvers)
        std::printf(" | %-6s %7s %7s %6s %7s %4s %3s", sv.c_str(), "an", "fac",
                    "sol", "wall", "dly", "ok");
    if (solvers.size() >= 2) std::printf(" | ratio");
    std::printf("\n");

    double log_sum = 0.0;
    int log_cnt = 0;
    for (const auto &c : cases) {
        if (!filter.empty() && c.id.find(filter) == std::string::npos) continue;
        auto mm = smf::read_matrix_market(c.mtx);
        if (!mm.ok) {
            std::printf("%-22s load failed: %s\n", c.id.c_str(), mm.error_message.c_str());
            continue;
        }
        const smf::CscLower A = std::move(mm.matrix);

        std::vector<std::vector<double>> seq(5, A.values);
        for (int step = 0; step < 5; ++step) {
            const double delta = 1.0 + 0.05 * step;
            for (smf::Int j = 0; j < A.n; ++j) {
                const smf::Int p0 = A.col_ptr[static_cast<std::size_t>(j)];
                if (p0 < A.col_ptr[static_cast<std::size_t>(j) + 1] &&
                    A.row_idx[static_cast<std::size_t>(p0)] == j &&
                    A.values[static_cast<std::size_t>(p0)] != 0.0) {
                    seq[static_cast<std::size_t>(step)][static_cast<std::size_t>(p0)] =
                        A.values[static_cast<std::size_t>(p0)] * delta;
                }
            }
        }

        smf::Control ctrl;
        ctrl.matrix_type = c.spd ? smf::MatrixType::RealSymmetricPositiveDefinite
                                 : smf::MatrixType::RealSymmetricIndefinite;
        if (nemin > 0) ctrl.nemin = nemin;
        if (ordering >= 0) ctrl.ordering = static_cast<smf::OrderingMethod>(ordering);
        if (scaling >= 0) ctrl.scaling = static_cast<smf::ScalingMethod>(scaling);
        int nsn = 0;
        {
            smf::Solver s;
            smf::Info info;
            smf::Control cc = ctrl;
            if (c.spd && ordering < 0) cc.ordering = smf::OrderingMethod::AutoParallel;
            auto ak = s.analyse(A, cc, info);
            if (ak) nsn = static_cast<int>(ak->supernodes.size());
            if (ak && verbose) {
                long long sum_f2 = 0, sum_pf = 0, sum_assembly = 0;
                int max_f = 0, max_ch = 0, max_p = 0;
                for (std::size_t s2 = 0; s2 < ak->supernodes.size(); ++s2) {
                    const int f = static_cast<int>(ak->fronts[s2].front_size());
                    const int p = static_cast<int>(ak->supernodes[s2].width());
                    sum_f2 += 1LL * f * f;
                    sum_pf += 1LL * p * f;
                    max_f = std::max(max_f, f);
                    max_p = std::max(max_p, p);
                    max_ch = std::max(max_ch, static_cast<int>(ak->supernodes[s2].children.size()));
                    for (smf::Int ch : ak->supernodes[s2].children) {
                        const int qc = static_cast<int>(ak->fronts[static_cast<std::size_t>(ch)].front_size() -
                                                        ak->supernodes[static_cast<std::size_t>(ch)].width());
                        sum_assembly += 1LL * qc * f;
                    }
                }
                std::printf("  [%s] ordering=%d maxf=%d maxp=%d maxch=%d sum_f2=%lld sum_pf=%lld "
                            "sum_qc*f=%lld pred_flops=%.3g\n",
                            c.id.c_str(), info.ordering_used, max_f, max_p, max_ch, sum_f2,
                            sum_pf, sum_assembly, info.predicted_flops);
            }
        }

        std::printf("%-22s %5d %4d", c.id.c_str(), static_cast<int>(A.n), nsn);
        std::vector<double> walls;
        for (const auto &sv : solvers) {
            auto ad = smf::bench::make_solver_adapter(sv);
            const auto id = ad ? ad->identity() : smf::bench::AdapterIdentity{};
            if (!ad || !id.available || (c.spd && !id.supports_spd) ||
                (!c.spd && !id.supports_indefinite)) {
                std::printf(" | %-6s %7s %7s %6s %7s %4s %3s", sv.c_str(), "-", "-", "-", "-", "-", "-");
                walls.push_back(std::numeric_limits<double>::quiet_NaN());
                continue;
            }
            ad->configure(ctrl, "recommended");
            run_once(*ad, A, c, seq); // warmup
            Timing best;
            best.wall = std::numeric_limits<double>::infinity();
            Timing last;
            for (int r = 0; r < repeats; ++r) {
                last = run_once(*ad, A, c, seq);
                if (!last.ok) break;
                if (last.wall < best.wall) best = last;
            }
            if (!last.ok) {
                std::printf(" | %-6s %7s %7s %6s %7s %4d %3s", sv.c_str(), "FAIL", "", "", "", last.delayed, "no");
                if (verbose) std::printf(" [%s]", last.why.c_str());
                walls.push_back(std::numeric_limits<double>::quiet_NaN());
                continue;
            }
            std::printf(" | %-6s %7.3f %7.3f %6.3f %7.3f %4d %3s", sv.c_str(), best.analyse,
                        best.factor, best.solve, best.wall, best.delayed, "ok");
            if (verbose) std::printf(" [L=%lld fl=%.3g]", best.fent, best.flops);
            walls.push_back(best.wall);
        }
        if (walls.size() >= 2 && std::isfinite(walls[0]) && std::isfinite(walls[1])) {
            const double ratio = walls[1] / walls[0];
            std::printf(" | %5.2fx", ratio);
            log_sum += std::log(ratio);
            ++log_cnt;
        }
        std::printf("\n");
        std::fflush(stdout);
    }
    if (log_cnt > 0)
        std::printf("geomean %s/%s speed ratio (>1 = %s faster): %.3fx over %d cases\n",
                    solvers.size() > 1 ? solvers[1].c_str() : "", solvers[0].c_str(),
                    solvers[0].c_str(), std::exp(log_sum / log_cnt), log_cnt);
    return 0;
}
