// bench_matching_trial: survey of analyse()'s auto matching trial.
//
// For each matrix of a list, analyses it (1 thread, default Control
// otherwise) with Control::matching_search_budget = -1 (the exact trial),
// the default budget, and 0 (always decide on the initial matching first,
// which records its flop ratio), and prints per matrix: how each trial was
// decided, the initial and exact matching orderings' flop ratios to the base
// ordering (Info::matching_initial_ratio / matching_exact_ratio), whether
// the default-budget analysis equals the exact one (ordering, matching
// scale, rejected flag), the median analyse wall time of both, and the
// full matching's search work in units of the default budget's (2 x lower
// entries of the matrix as read; analyse() matches its cleaned copy).
//
// Usage: bench_matching_trial --list file [--trials T] [--ordering m]
//                             [--budget b] [--refactors r] [--case substring]
//   file: one "<id> <path.mtx>" per line. --budget replaces the default
//   budget of the second analysis. --ordering is the OrderingMethod value
//   and --refactors Control::expected_refactors (the IPOPT plugin's
//   configuration is --ordering 1 --refactors 10).

#include "smf/control.hpp"
#include "smf/info.hpp"
#include "smf/matrix_market.hpp"
#include "smf/scaling_matching.hpp"
#include "smf/solver.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

const char *trial_name(smf::MatchingTrial t) {
    switch (t) {
    case smf::MatchingTrial::None: return "none";
    case smf::MatchingTrial::Exact: return "exact";
    case smf::MatchingTrial::RejectedInitial: return "rej-init";
    case smf::MatchingTrial::ResumedExact: return "resumed";
    }
    return "?";
}

template <class T> std::uint64_t fnv(const std::vector<T> &v) {
    std::uint64_t h = 1469598103934665603ull;
    const auto *b = reinterpret_cast<const unsigned char *>(v.data());
    for (std::size_t i = 0; i < v.size() * sizeof(T); ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

struct Run {
    smf::Info info;
    std::vector<smf::Int> perm;
    std::vector<double> scale;
    bool rejected = false;
    double ms = 0.0;  // median wall time of analyse()
    bool ok = false;
};

template <class I> double initial_ratio(const I &info) {
    if constexpr (requires { info.matching_initial_ratio; }) return info.matching_initial_ratio;
    else return 0.0;
}
template <class I> double exact_ratio(const I &info) {
    if constexpr (requires { info.matching_exact_ratio; }) return info.matching_exact_ratio;
    else return 0.0;
}

Run analyse(const smf::CscLower &A, const smf::Control &ctrl, int trials) {
    Run r;
    std::vector<double> ms;
    for (int t = 0; t < trials; ++t) {
        smf::Solver s;
        smf::Info info;
        const auto t0 = std::chrono::steady_clock::now();
        auto ak = s.analyse(A, ctrl, info);
        const auto t1 = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        if (t == 0) {
            r.ok = ak != nullptr && info.status == smf::ErrorCode::Success;
            r.info = info;
            if (ak) {
                r.perm = ak->perm;
                r.scale = ak->matching_scale;
                r.rejected = ak->matching_rejected;
            }
        }
    }
    std::sort(ms.begin(), ms.end());
    r.ms = ms[ms.size() / 2];
    return r;
}

}  // namespace

int main(int argc, char **argv) {
    std::string list, filter;
    int trials = 3, ordering = -1, refactors = -1;
    double budget = smf::Control{}.matching_search_budget;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--list" && i + 1 < argc) list = argv[++i];
        else if (a == "--trials" && i + 1 < argc) trials = std::max(1, std::atoi(argv[++i]));
        else if (a == "--ordering" && i + 1 < argc) ordering = std::atoi(argv[++i]);
        else if (a == "--budget" && i + 1 < argc) budget = std::atof(argv[++i]);
        else if (a == "--refactors" && i + 1 < argc) refactors = std::atoi(argv[++i]);
        else if (a == "--case" && i + 1 < argc) filter = argv[++i];
    }
    if (list.empty()) {
        std::fprintf(stderr, "usage: bench_matching_trial --list file [--trials T] [--ordering m] "
                             "[--budget b] [--case s]\n");
        return 2;
    }
    std::ifstream in(list);
    std::string line;
    std::printf("%-28s %7s %8s | %-8s %-8s %-8s | %6s %6s | %3s %4s %4s | %9s %9s | %7s %6s | %16s\n", "matrix",
                "n", "nnz_low", "exact", "default", "zero", "r_init", "r_exact", "eq", "ordE", "ordD", "ms_exact",
                "ms_default", "work_x", "free0", "perm_hash");
    double tot_exact = 0.0, tot_default = 0.0;
    int rows = 0, mismatches = 0;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string id, path;
        if (!(ls >> id >> path)) continue;
        if (!filter.empty() && id.find(filter) == std::string::npos) continue;
        auto mm = smf::read_matrix_market(path);
        if (!mm.ok) {
            std::printf("%-28s load failed: %s\n", id.c_str(), mm.error_message.c_str());
            continue;
        }
        smf::Control base;
        base.matrix_type = smf::MatrixType::RealSymmetricIndefinite;
        base.num_threads = 1;
        if (ordering >= 0) base.ordering = static_cast<smf::OrderingMethod>(ordering);
        if (refactors >= 1) base.expected_refactors = refactors;
        smf::Control ce = base, cd = base, cz = base;
        ce.matching_search_budget = -1.0;
        cd.matching_search_budget = budget;
        cz.matching_search_budget = 0.0;
        const Run e = analyse(mm.matrix, ce, trials);
        const Run d = analyse(mm.matrix, cd, trials);
        const Run z = analyse(mm.matrix, cz, 1);
        const bool eq = e.ok && d.ok && e.perm == d.perm && e.scale == d.scale && e.rejected == d.rejected &&
                        e.info.ordering_used == d.info.ordering_used;
        mismatches += eq ? 0 : 1;
        tot_exact += e.ms;
        tot_default += d.ms;
        ++rows;
        double work_x = 0.0, init_free = 0.0;
        if (z.info.matching_trial != smf::MatchingTrial::None) {
            smf::SymmetricMatcher m(mm.matrix);
            init_free = 1.0 - static_cast<double>(m.initial().matched_rank) / static_cast<double>(mm.matrix.n);
            m.advance(-1);
            work_x = static_cast<double>(m.work()) / (2.0 * static_cast<double>(mm.matrix.row_idx.size()));
        }
        double ri = initial_ratio(z.info);
        if (ri == 0.0) ri = initial_ratio(d.info);
        std::printf("%-28s %7d %8zu | %-8s %-8s %-8s | %6.3f %6.3f | %3s %4d %4d | %9.2f %9.2f | %7.2f %6.3f | %016llx\n",
                    id.c_str(), static_cast<int>(mm.matrix.n), mm.matrix.row_idx.size(),
                    trial_name(e.info.matching_trial), trial_name(d.info.matching_trial),
                    trial_name(z.info.matching_trial), ri, exact_ratio(e.info), eq ? "yes" : "NO",
                    e.info.ordering_used, d.info.ordering_used, e.ms, d.ms, work_x, init_free,
                    static_cast<unsigned long long>(fnv(d.perm) ^ (fnv(d.scale) * 31u) ^ (d.rejected ? 1u : 0u)));
        std::fflush(stdout);
    }
    std::printf("total %d matrices, %d differ from the exact trial; analyse ms exact %.1f default %.1f\n", rows,
                mismatches, tot_exact, tot_default);
    return mismatches == 0 ? 0 : 1;
}
