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
//
// Also printed: mm_def = median Info::analyse_phases.matching of the
// default-budget analysis (ms); mt_ms = median wall time of a standalone
// SymmetricMatcher run to the end (construct + advance(-1) + current());
// all_hash = one hash over every output of the three analyses (ordering,
// matching scale, rejected flag, ordering used, trial verdict, both flop
// ratios) and of the standalone matcher (initial and final matching and
// scale, search work, status), for bit-identity checks across builds.
// --sample S (with --case) instead loops the default-budget analyse() of
// the first matching case for S seconds under the in-process sampler
// (--lines: by source line; needs a /Zi + /DEBUG build).

#include "smf/control.hpp"
#include "smf/info.hpp"
#include "smf/matrix_market.hpp"
#include "smf/scaling_matching.hpp"
#include "smf/solver.hpp"

#include "common/sampler.hpp"

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

// Order-sensitive hash accumulator over raw bytes.
struct Hasher {
    std::uint64_t h = 1469598103934665603ull;
    void bytes(const void *p, std::size_t n) {
        const auto *b = static_cast<const unsigned char *>(p);
        for (std::size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    }
    template <class T> void pod(const T &x) { bytes(&x, sizeof(T)); }
    template <class T> void vec(const std::vector<T> &v) {
        pod(v.size());
        bytes(v.data(), v.size() * sizeof(T));
    }
};

struct Run {
    smf::Info info;
    std::vector<smf::Int> perm;
    std::vector<double> scale;
    bool rejected = false;
    double ms = 0.0;  // median wall time of analyse()
    double match_ms = 0.0;  // median Info::analyse_phases.matching (ms)
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
    std::vector<double> ms, mms;
    for (int t = 0; t < trials; ++t) {
        smf::Solver s;
        smf::Info info;
        const auto t0 = std::chrono::steady_clock::now();
        auto ak = s.analyse(A, ctrl, info);
        const auto t1 = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        mms.push_back(1000.0 * info.analyse_phases.matching);
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
    std::sort(mms.begin(), mms.end());
    r.match_ms = mms[mms.size() / 2];
    return r;
}

void hash_run(Hasher &h, const Run &r) {
    h.pod(r.ok);
    h.vec(r.perm);
    h.vec(r.scale);
    h.pod(r.rejected);
    h.pod(r.info.ordering_used);
    h.pod(r.info.matching_trial);
    h.pod(initial_ratio(r.info));
    h.pod(exact_ratio(r.info));
}

void hash_matching(Hasher &h, const smf::MatchingResult &m) {
    h.pod(m.status);
    h.vec(m.scale);
    h.vec(m.row_to_col);
    h.vec(m.col_to_row);
    h.pod(m.matched_rank);
}

}  // namespace

int main(int argc, char **argv) {
    std::string list, filter;
    int trials = 3, ordering = -1, refactors = -1;
    double sample_seconds = 0.0;
    bool sample_lines = false;
    double budget = smf::Control{}.matching_search_budget;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--list" && i + 1 < argc) list = argv[++i];
        else if (a == "--trials" && i + 1 < argc) trials = std::max(1, std::atoi(argv[++i]));
        else if (a == "--ordering" && i + 1 < argc) ordering = std::atoi(argv[++i]);
        else if (a == "--budget" && i + 1 < argc) budget = std::atof(argv[++i]);
        else if (a == "--refactors" && i + 1 < argc) refactors = std::atoi(argv[++i]);
        else if (a == "--case" && i + 1 < argc) filter = argv[++i];
        else if (a == "--sample" && i + 1 < argc) sample_seconds = std::atof(argv[++i]);
        else if (a == "--lines") sample_lines = true;
    }
    if (list.empty()) {
        std::fprintf(stderr, "usage: bench_matching_trial --list file [--trials T] [--ordering m] "
                             "[--budget b] [--case s]\n");
        return 2;
    }
    std::ifstream in(list);
    std::string line;
    if (sample_seconds > 0.0) {
        while (std::getline(in, line)) {
            std::istringstream ls(line);
            std::string id, path;
            if (!(ls >> id >> path)) continue;
            if (!filter.empty() && id.find(filter) == std::string::npos) continue;
            auto mm = smf::read_matrix_market(path);
            if (!mm.ok) return 2;
            smf::Control c;
            c.matrix_type = smf::MatrixType::RealSymmetricIndefinite;
            c.num_threads = 1;
            if (ordering >= 0) c.ordering = static_cast<smf::OrderingMethod>(ordering);
            if (refactors >= 1) c.expected_refactors = refactors;
            c.matching_search_budget = budget;
            int loops = 0;
            smf::bench::Sampler sampler;
            const auto t0 = std::chrono::steady_clock::now();
            sampler.start();
            while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < sample_seconds) {
                smf::Solver s;
                smf::Info info;
                (void)s.analyse(mm.matrix, c, info);
                ++loops;
            }
            sampler.stop();
            std::printf("sampled %s: %d analyses\n", id.c_str(), loops);
            sampler.report(sample_lines);
            return 0;
        }
        return 2;
    }
    std::printf("%-28s %7s %8s | %-8s %-8s %-8s | %6s %6s | %3s %4s %4s | %9s %9s | %7s %6s | %16s | %8s %8s | %16s\n",
                "matrix", "n", "nnz_low", "exact", "default", "zero", "r_init", "r_exact", "eq", "ordE", "ordD",
                "ms_exact", "ms_default", "work_x", "free0", "perm_hash", "mm_def", "mt_ms", "all_hash");
    double tot_exact = 0.0, tot_default = 0.0, tot_mm = 0.0, tot_mt = 0.0;
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
        // Standalone matcher: timed (median of `trials`) and hashed.
        Hasher h;
        hash_run(h, e);
        hash_run(h, d);
        hash_run(h, z);
        std::vector<double> mt;
        for (int t = 0; t < trials; ++t) {
            const auto t0 = std::chrono::steady_clock::now();
            smf::SymmetricMatcher m(mm.matrix);
            const smf::MatchingResult init = t == 0 ? m.initial() : smf::MatchingResult{};
            m.advance(-1);
            const smf::MatchingResult fin = m.current();
            const auto t1 = std::chrono::steady_clock::now();
            mt.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            if (t == 0) {
                hash_matching(h, init);
                hash_matching(h, fin);
                h.pod(m.work());
                h.pod(m.status());
            }
        }
        std::sort(mt.begin(), mt.end());
        const double mt_ms = mt[mt.size() / 2];
        tot_mm += d.match_ms;
        tot_mt += mt_ms;
        double ri = initial_ratio(z.info);
        if (ri == 0.0) ri = initial_ratio(d.info);
        std::printf("%-28s %7d %8zu | %-8s %-8s %-8s | %6.3f %6.3f | %3s %4d %4d | %9.2f %9.2f | %7.2f %6.3f | %016llx | %8.2f %8.2f | %016llx\n",
                    id.c_str(), static_cast<int>(mm.matrix.n), mm.matrix.row_idx.size(),
                    trial_name(e.info.matching_trial), trial_name(d.info.matching_trial),
                    trial_name(z.info.matching_trial), ri, exact_ratio(e.info), eq ? "yes" : "NO",
                    e.info.ordering_used, d.info.ordering_used, e.ms, d.ms, work_x, init_free,
                    static_cast<unsigned long long>(fnv(d.perm) ^ (fnv(d.scale) * 31u) ^ (d.rejected ? 1u : 0u)),
                    d.match_ms, mt_ms, static_cast<unsigned long long>(h.h));
        std::fflush(stdout);
    }
    std::printf("total %d matrices, %d differ from the exact trial; analyse ms exact %.1f default %.1f; "
                "matching phase (default) %.1f; standalone matcher %.1f\n",
                rows, mismatches, tot_exact, tot_default, tot_mm, tot_mt);
    return mismatches == 0 ? 0 : 1;
}
