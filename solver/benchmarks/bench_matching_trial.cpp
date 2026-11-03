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
// The default-budget analysis keeps every other Control default, the
// first budget of Control::matching_initial_decide_passes included (it
// caps --budget for matrices whose initial phase leaves >= 5% free).
//
// Usage: bench_matching_trial --list file [--trials T] [--ordering m]
//                             [--budget b] [--refactors r] [--case substring]
//                             [--sample S [--lines]] [--pieces] [--matcher ssp|hybrid]
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
// The trial's matcher runs in MatchingSearchOrder::Degree order, so the
// survey also reports that run to the end (work_deg = its edges scanned in
// passes of 2 x lower entries, the baseline an alternative matcher is gated
// against; mtd_ms = its median wall time; logprod = sum_j log|a(col_to_row[j],
// j)|, the matching's product, printed in full and as an fnv hash of
// col_to_row). The totals line adds edges_deg = the summed edges scanned.
// --matcher selects Control::matching_algorithm on every analyse() call
// (the trial's matcher: ssp = ShortestPath, hybrid = Hybrid; the bench
// default is ssp, the library default hybrid), so the verdict, ordD and
// perm_hash columns show the trial under it. Under hybrid the Degree
// ShortestPath columns are still produced and a Degree-order Hybrid run
// (default options) follows them: work_auc = its edges scanned in passes
// (as work_deg), repair = SymmetricMatcher::repaired(), mta_ms = its median
// wall time, prod_ok = yes when its status equals the ShortestPath one and,
// on success, its log product equals the ShortestPath one within
// 1e-9 |p| + 1e-9 and its scaling holds |s_i a_ij s_j| <= 1 + 1e-10 with
// matched entries within 1e-10 of 1 (NO otherwise: prod_fail counts them),
// sw = SymmetricMatcher::switched() (y/n). The totals line adds edges_auc,
// prod_fail and the worst work_auc / work_deg (over all matrices, and over
// those with work_deg > 1). Exit code 1 when an analysis differs from the
// exact trial or (hybrid) on any prod_fail.
// --sample S (with --case) instead loops the default-budget analyse() of
// the first matching case for S seconds under the in-process sampler
// (--lines: by source line; needs a /Zi + /DEBUG build).

#include "smf/control.hpp"
#include "smf/etree.hpp"
#include "smf/info.hpp"
#include "smf/matrix_market.hpp"
#include "smf/ordering.hpp"
#include "smf/scaling_matching.hpp"
#include "smf/solver.hpp"
#include "smf/supernode.hpp"

#include "../src/matching_internal.hpp"

#include "common/sampler.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
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
    case smf::MatchingTrial::Diagonal: return "diagonal";
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

// sum_j log|a(col_to_row[j], j)| of a (symmetric) matching of the lower-triangle
// A: entry (r, j) is stored in column min(r, j); a duplicate counts its largest
// magnitude. Unmatched columns are skipped.
double matching_logprod(const smf::CscLower &A, const std::vector<smf::Int> &col_to_row) {
    double sum = 0.0;
    for (smf::Int j = 0; j < A.n; ++j) {
        const smf::Int r = col_to_row[static_cast<std::size_t>(j)];
        if (r < 0) continue;
        const auto c = static_cast<std::size_t>(std::min(r, j));
        const smf::Int row = std::max(r, j);
        double a = 0.0;
        for (auto k = static_cast<std::size_t>(A.col_ptr[c]); k < static_cast<std::size_t>(A.col_ptr[c + 1]); ++k)
            if (A.row_idx[k] == row) a = std::max(a, std::abs(A.values[k]));
        sum += std::log(a);
    }
    return sum;
}

// The hybrid's checks, as test_auction_matching.cpp's same_optimum and
// scaling_bounds: sum of log|a_ij| over the matched merged entries
// (|a_ij| = col_max_j exp(-cost)).
double cost_graph_logprod(const smf::detail::CostGraph &g, const std::vector<smf::Int> &col_to_row) {
    double s = 0.0;
    for (int j = 0; j < static_cast<int>(col_to_row.size()); ++j)
        for (smf::Int q = g.ptr[j]; q < g.ptr[j + 1]; ++q)
            if (g.row[q] == col_to_row[static_cast<std::size_t>(j)])
                s += std::log(g.col_max[static_cast<std::size_t>(j)]) - g.cost[q];
    return s;
}

// No scaled entry above 1 + 1e-10, every matched one within 1e-10 of 1.
bool scaling_bounds_hold(const smf::detail::CostGraph &g, const smf::MatchingResult &r) {
    const int n = static_cast<int>(g.ptr.size()) - 1;
    if (r.scale.size() != static_cast<std::size_t>(n)) return false;
    for (int j = 0; j < n; ++j)
        for (smf::Int q = g.ptr[j]; q < g.ptr[j + 1]; ++q) {
            const int i = g.row[q];
            const double a = g.col_max[static_cast<std::size_t>(j)] * std::exp(-g.cost[q]);
            const double s = r.scale[static_cast<std::size_t>(i)] * a * r.scale[static_cast<std::size_t>(j)];
            if (!(s <= 1.0 + 1e-10)) return false;
            if (r.col_to_row[static_cast<std::size_t>(j)] == i && !(std::abs(s - 1.0) <= 1e-10)) return false;
        }
    return true;
}

// prod_ok: the hybrid's matching `a` against the ShortestPath one `b`.
bool product_ok(const smf::CscLower &A, const smf::MatchingResult &a, const smf::MatchingResult &b) {
    if (a.status != b.status) return false;
    if (a.status != smf::ErrorCode::Success) return true;
    if (a.matched_rank != A.n || b.matched_rank != A.n) return false;
    smf::detail::CostGraph g;
    if (smf::detail::build_cost_graph(A, g) != smf::ErrorCode::Success) return false;
    const double pa = cost_graph_logprod(g, a.col_to_row), pb = cost_graph_logprod(g, b.col_to_row);
    return std::abs(pa - pb) <= 1e-9 * std::abs(pb) + 1e-9 && scaling_bounds_hold(g, a);
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
    bool sample_lines = false, pieces = false;
    double budget = smf::Control{}.matching_search_budget;
    std::string matcher = "ssp";  // the bench default; the library default is hybrid
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--list" && i + 1 < argc) list = argv[++i];
        else if (a == "--trials" && i + 1 < argc) trials = std::max(1, std::atoi(argv[++i]));
        else if (a == "--ordering" && i + 1 < argc) ordering = std::atoi(argv[++i]);
        else if (a == "--budget" && i + 1 < argc) budget = std::atof(argv[++i]);
        else if (a == "--refactors" && i + 1 < argc) refactors = std::atoi(argv[++i]);
        else if (a == "--case" && i + 1 < argc) filter = argv[++i];
        else if (a == "--sample" && i + 1 < argc) sample_seconds = std::atof(argv[++i]);
        else if (a == "--matcher" && i + 1 < argc) matcher = argv[++i];
        else if (a == "--lines") sample_lines = true;
        else if (a == "--pieces") pieces = true;
    }
    if (list.empty() || (matcher != "ssp" && matcher != "hybrid")) {
        std::fprintf(stderr, "usage: bench_matching_trial --list file [--trials T] [--ordering m] "
                             "[--budget b] [--refactors r] [--case s] "
                             "[--sample S [--lines]] [--pieces] [--matcher ssp|hybrid]\n");
        return 2;
    }
    const smf::MatchingAlgorithm matcher_alg =
        matcher == "hybrid" ? smf::MatchingAlgorithm::Hybrid : smf::MatchingAlgorithm::ShortestPath;
    std::ifstream in(list);
    std::string line;
    if (pieces) {
        // The trial's pieces timed standalone (median of `trials`, ms):
        // ctor = SymmetricMatcher construction (cost graph, duals, greedy,
        // length-2), budget = the searches up to the default budget, rest =
        // the remaining searches, oi_amd / oi_auto = the initial matching's
        // compressed ordering with AMD only / Auto (AMD, METIS when dear),
        // ox_auto = the exact matching's (Auto); r_amd = predicted flops of
        // the initial matching's compressed AMD ordering over the base AMD
        // ordering's, and fl0/n = the base AMD ordering's predicted flops
        // per row; then analyse()'s phases.
        std::printf("%-28s %7s | %8s %8s %8s | %8s %8s %4s | %8s %4s | %6s %9s | %8s %8s %8s %8s | %-8s\n",
                    "matrix", "n", "ctor", "budget", "rest", "oi_amd", "oi_auto", "usd", "ox_auto", "usd",
                    "r_amd", "fl0/n", "an_ord", "an_cnt", "an_mat", "analyse", "trial");
        const auto med = [](std::vector<double> v) {
            std::sort(v.begin(), v.end());
            return v[v.size() / 2];
        };
        const auto ms_since = [](std::chrono::steady_clock::time_point t0) {
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        };
        while (std::getline(in, line)) {
            std::istringstream ls(line);
            std::string id, path;
            if (!(ls >> id >> path)) continue;
            if (!filter.empty() && id.find(filter) == std::string::npos) continue;
            auto mm = smf::read_matrix_market(path);
            if (!mm.ok) continue;
            const smf::CscLower &A = mm.matrix;
            const smf::Int n = A.n;
            std::vector<smf::Int> xadj(static_cast<std::size_t>(n) + 1),
                adj(smf::lower_csc_adjacency_size(n, A.col_ptr.data(), A.row_idx.data()));
            smf::lower_csc_adjacency(n, A.col_ptr.data(), A.row_idx.data(), xadj.data(), adj.data());
            const double edges = (budget >= 0.0 ? budget : 2.0) * 2.0 * static_cast<double>(A.row_idx.size());
            // The base AMD ordering and its predicted flops.
            std::vector<smf::Int> base_perm(static_cast<std::size_t>(n)), amd_w(smf::amd_adjacency_work_size(n, adj.size()));
            smf::amd_order_adjacency(n, xadj.data(), adj.data(), base_perm.data(), amd_w.data());
            double fl0 = 0.0;
            {
                const auto sn = static_cast<std::size_t>(n);
                std::vector<smf::Int> ip(sn), par(sn), post(sn), cc(sn), w(4 * sn + 1);
                for (smf::Int i = 0; i < n; ++i) ip[static_cast<std::size_t>(base_perm[static_cast<std::size_t>(i)])] = i;
                smf::etree_from_adjacency(n, xadj.data(), adj.data(), base_perm.data(), ip.data(), par.data(), w.data());
                smf::etree_postorder(n, par.data(), post.data(), w.data());
                smf::filled_column_counts_from_adjacency(n, xadj.data(), adj.data(), base_perm.data(), ip.data(),
                                                         par.data(), post.data(), cc.data(), w.data());
                for (smf::Int c : cc) fl0 += static_cast<double>(c) * static_cast<double>(c);
            }
            double r_amd = 0.0;
            std::vector<double> t_ctor, t_budget, t_rest, t_oia, t_oiu, t_oxu, t_ord, t_cnt, t_mat, t_an;
            int used_i = -1, used_x = -1;
            bool paused = false;
            smf::Info last;
            for (int t = 0; t < trials; ++t) {
                auto t0 = std::chrono::steady_clock::now();
                smf::SymmetricMatcher m(A);
                t_ctor.push_back(ms_since(t0));
                t0 = std::chrono::steady_clock::now();
                paused = !m.advance(static_cast<long long>(edges));
                t_budget.push_back(ms_since(t0));
                const smf::MatchingResult init = m.initial();
                t0 = std::chrono::steady_clock::now();
                m.advance(-1);
                t_rest.push_back(ms_since(t0));
                const smf::MatchingResult fin = m.current();
                std::vector<smf::Int> perm, work;
                smf::MatchingOrderStats st;
                t0 = std::chrono::steady_clock::now();
                smf::matching_order_from(A, init, 0.1, perm, work, &st, smf::CompressedOrder::Amd, 1, xadj.data(),
                                         adj.data());
                t_oia.push_back(ms_since(t0));
                r_amd = st.flops / fl0;
                t0 = std::chrono::steady_clock::now();
                smf::matching_order_from(A, init, 0.1, perm, work, &st, smf::CompressedOrder::Auto, 1, xadj.data(),
                                         adj.data());
                t_oiu.push_back(ms_since(t0));
                used_i = static_cast<int>(st.used);
                t0 = std::chrono::steady_clock::now();
                if (fin.status == smf::ErrorCode::Success)
                    smf::matching_order_from(A, fin, 0.1, perm, work, &st, smf::CompressedOrder::Auto, 1,
                                             xadj.data(), adj.data());
                t_oxu.push_back(ms_since(t0));
                used_x = static_cast<int>(st.used);
                smf::Control c;
                c.matrix_type = smf::MatrixType::RealSymmetricIndefinite;
                c.num_threads = 1;
                if (ordering >= 0) c.ordering = static_cast<smf::OrderingMethod>(ordering);
                if (refactors >= 1) c.expected_refactors = refactors;
                c.matching_search_budget = budget;
                c.matching_algorithm = matcher_alg;
                smf::Solver s;
                t0 = std::chrono::steady_clock::now();
                (void)s.analyse(A, c, last);
                t_an.push_back(ms_since(t0));
                t_ord.push_back(1000.0 * last.analyse_phases.ordering);
                t_cnt.push_back(1000.0 * last.analyse_phases.etree_counts);
                t_mat.push_back(1000.0 * last.analyse_phases.matching);
            }
            std::printf("%-28s %7d | %8.2f %8.2f %8.2f | %8.2f %8.2f %4d | %8.2f %4d | %6.3f %9.3g | %8.2f "
                        "%8.2f %8.2f %8.2f | %-8s%s\n",
                        id.c_str(), static_cast<int>(n), med(t_ctor), med(t_budget), med(t_rest), med(t_oia),
                        med(t_oiu), used_i, med(t_oxu), used_x, r_amd, fl0 / static_cast<double>(n), med(t_ord), med(t_cnt),
                        med(t_mat), med(t_an), trial_name(last.matching_trial), paused ? "" : " (in budget)");
            std::fflush(stdout);
        }
        return 0;
    }
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
            c.matching_algorithm = matcher_alg;
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
    // The extra standalone matcher columns: Hybrid (ssp: none).
    const bool use_alt = matcher_alg != smf::MatchingAlgorithm::ShortestPath;
    std::printf("%-28s %7s %8s | %-8s %-8s %-8s | %6s %6s | %3s %4s %4s | %9s %9s | %7s %6s | %16s | %8s %8s | %16s | %8s %8s | %16s %21s",
                "matrix", "n", "nnz_low", "exact", "default", "zero", "r_init", "r_exact", "eq", "ordE", "ordD",
                "ms_exact", "ms_default", "work_x", "free0", "perm_hash", "mm_def", "mt_ms", "all_hash",
                "work_deg", "mtd_ms", "logprod_hash", "logprod");
    if (use_alt) std::printf(" | %8s %6s %8s %7s %2s", "work_auc", "repair", "mta_ms", "prod_ok", "sw");
    std::printf("\n");
    double tot_exact = 0.0, tot_default = 0.0, tot_mm = 0.0, tot_mt = 0.0, edges_deg = 0.0, edges_auc = 0.0;
    double worst_ratio = 0.0, worst_ratio_gt1 = 0.0;
    std::string worst_id, worst_id_gt1;
    int rows = 0, mismatches = 0, prod_fail = 0;
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
        base.matching_algorithm = matcher_alg;
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
        std::vector<double> mt, mtd;
        double work_deg = 0.0, logprod = 0.0, work_auc = 0.0;
        std::vector<double> mta;
        int repair = 0;
        bool prod_ok = true, switched = false;
        smf::MatchingResult fd0;
        std::uint64_t logprod_hash = 0;
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
            // The trial's matcher: Degree search order, run to the end.
            const auto d0 = std::chrono::steady_clock::now();
            smf::SymmetricMatcher md(mm.matrix, smf::MatchingSearchOrder::Degree);
            md.advance(-1);
            const smf::MatchingResult fd = md.current();
            const auto d1 = std::chrono::steady_clock::now();
            mtd.push_back(std::chrono::duration<double, std::milli>(d1 - d0).count());
            if (t == 0) {
                hash_matching(h, fd);
                h.pod(md.work());
                h.pod(md.status());
                work_deg = static_cast<double>(md.work()) / (2.0 * static_cast<double>(mm.matrix.row_idx.size()));
                edges_deg += static_cast<double>(md.work());
                logprod = matching_logprod(mm.matrix, fd.col_to_row);
                logprod_hash = fnv(fd.col_to_row);
                if (use_alt) fd0 = fd;
            }
            if (use_alt) {
                // The same, with the hybrid in the sparse part.
                const auto a0 = std::chrono::steady_clock::now();
                smf::SymmetricMatcher ma(mm.matrix, smf::MatchingSearchOrder::Degree, matcher_alg, nullptr);
                const bool done = ma.advance(-1);
                const smf::MatchingResult fa = done ? ma.current() : smf::MatchingResult{};
                const auto a1 = std::chrono::steady_clock::now();
                mta.push_back(std::chrono::duration<double, std::milli>(a1 - a0).count());
                if (t == 0) {
                    work_auc = static_cast<double>(ma.work()) / (2.0 * static_cast<double>(mm.matrix.row_idx.size()));
                    edges_auc += static_cast<double>(ma.work());
                    repair = ma.repaired();
                    switched = ma.switched();
                    prod_ok = done && product_ok(mm.matrix, fa, fd0);
                }
            }
        }
        std::sort(mtd.begin(), mtd.end());
        const double mtd_ms = mtd[mtd.size() / 2];
        double mta_ms = 0.0;
        if (use_alt) {
            std::sort(mta.begin(), mta.end());
            mta_ms = mta[mta.size() / 2];
            prod_fail += prod_ok ? 0 : 1;
            if (work_deg > 0.0 && work_auc / work_deg > worst_ratio) {
                worst_ratio = work_auc / work_deg;
                worst_id = id;
            }
            if (work_deg > 1.0 && work_auc / work_deg > worst_ratio_gt1) {
                worst_ratio_gt1 = work_auc / work_deg;
                worst_id_gt1 = id;
            }
        }
        std::sort(mt.begin(), mt.end());
        const double mt_ms = mt[mt.size() / 2];
        tot_mm += d.match_ms;
        tot_mt += mt_ms;
        double ri = initial_ratio(z.info);
        if (ri == 0.0) ri = initial_ratio(d.info);
        std::printf("%-28s %7d %8zu | %-8s %-8s %-8s | %6.3f %6.3f | %3s %4d %4d | %9.2f %9.2f | %7.2f %6.3f | %016llx | %8.2f %8.2f | %016llx | %8.2f %8.2f | %016llx %21.12e",
                    id.c_str(), static_cast<int>(mm.matrix.n), mm.matrix.row_idx.size(),
                    trial_name(e.info.matching_trial), trial_name(d.info.matching_trial),
                    trial_name(z.info.matching_trial), ri, exact_ratio(e.info), eq ? "yes" : "NO",
                    e.info.ordering_used, d.info.ordering_used, e.ms, d.ms, work_x, init_free,
                    static_cast<unsigned long long>(fnv(d.perm) ^ (fnv(d.scale) * 31u) ^ (d.rejected ? 1u : 0u)),
                    d.match_ms, mt_ms, static_cast<unsigned long long>(h.h), work_deg, mtd_ms,
                    static_cast<unsigned long long>(logprod_hash), logprod);
        if (use_alt)
            std::printf(" | %8.2f %6d %8.2f %7s %2s", work_auc, repair, mta_ms, prod_ok ? "yes" : "NO",
                        switched ? "y" : "n");
        std::printf("\n");
        std::fflush(stdout);
    }
    std::printf("total %d matrices, %d differ from the exact trial; analyse ms exact %.1f default %.1f; "
                "matching phase (default) %.1f; standalone matcher %.1f; edges_deg %.0f",
                rows, mismatches, tot_exact, tot_default, tot_mm, tot_mt, edges_deg);
    if (use_alt)
        std::printf("; edges_auc %.0f; prod_fail %d; worst work_auc/work_deg %.3f (%s), over work_deg > 1: %.3f (%s)",
                    edges_auc, prod_fail, worst_ratio, worst_id.c_str(), worst_ratio_gt1, worst_id_gt1.c_str());
    std::printf("\n");
    return mismatches == 0 && prod_fail == 0 ? 0 : 1;
}
