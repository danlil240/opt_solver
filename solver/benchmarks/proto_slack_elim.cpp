// proto_slack_elim: feasibility prototype of analysis-level slack-pair
// elimination (not wired into smf).
//
// An IPOPT KKT with inequality constraints d(x) - s = 0 carries, per
// inequality, a slack s (diagonal Sigma, one off-diagonal: -1 to its
// multiplier y) and the multiplier y (diagonal -delta_c, the slack and the
// k state columns of the constraint's Jacobian row b). Eliminating s then y
// (two 1x1 pivots) leaves the rank-1 term -b b^T / d_y on the states,
// d_y = a_yy - a_ys^2 / a_ss. This program detects the pairs, forms the
// reduced matrix once per value step in a flat loop, factors it with smf and
// solves; it times that against smf and MA27 on the full KKT, and checks the
// solution's residual on the full matrix.
//
// Usage: proto_slack_elim <kkt.mtx> <kkt.vals> [steps] [repeats] [u]
//   kkt.vals: the plugin's capture (SMF_KKT_DUMP_DIR), raw doubles, one
//   value array per factorization in the .mtx entry order.

#include "adapters/adapter.hpp"
#include "common/validation.hpp"
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/matrix_market.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using smf::Int;

namespace {

struct Pair {
    Int s = -1, y = -1;
    long long ss = -1, ys = -1, yy = -1;  // value slots in A
    std::vector<long long> b;             // value slots of y's state entries
    std::vector<Int> bx;                  // their state rows (reduced index)
    std::vector<long long> blk;           // reduced slots of the k x k lower block, row-major i >= j
};

double now_ms() {
    using C = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(C::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: proto_slack_elim kkt.mtx kkt.vals [steps] [repeats] [u]\n");
        return 2;
    }
    const int steps_req = argc > 3 ? std::atoi(argv[3]) : 40;
    const int repeats = argc > 4 ? std::atoi(argv[4]) : 3;
    const double u = argc > 5 ? std::atof(argv[5]) : 1e-8;
    auto mm = smf::read_matrix_market(argv[1]);
    if (!mm.ok) {
        std::fprintf(stderr, "load: %s\n", mm.error_message.c_str());
        return 1;
    }
    const smf::CscLower A = std::move(mm.matrix);
    const std::size_t nnz = A.values.size();
    const Int n = A.n;
    std::vector<std::vector<double>> seq;
    {
        std::ifstream f(argv[2], std::ios::binary);
        std::vector<double> v(nnz);
        while (static_cast<int>(seq.size()) < steps_req &&
               f.read(reinterpret_cast<char *>(v.data()), static_cast<std::streamsize>(nnz * sizeof(double))))
            seq.push_back(v);
    }
    const int steps = static_cast<int>(seq.size());
    std::printf("n %d nnz %zu steps %d u %.1e\n", static_cast<int>(n), nnz, steps, u);

    // Adjacency (symmetric, off-diagonal) with the value slot of each edge.
    std::vector<std::vector<std::pair<Int, long long>>> adj(static_cast<std::size_t>(n));
    std::vector<long long> diag(static_cast<std::size_t>(n), -1);
    for (Int j = 0; j < n; ++j)
        for (Int p = A.col_ptr[static_cast<std::size_t>(j)]; p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
            const Int i = A.row_idx[static_cast<std::size_t>(p)];
            if (i == j) diag[static_cast<std::size_t>(j)] = p;
            else {
                adj[static_cast<std::size_t>(i)].push_back({j, p});
                adj[static_cast<std::size_t>(j)].push_back({i, p});
            }
        }
    // Pairs: s of degree 1 with a diagonal; its neighbour y of degree <= 9
    // whose other neighbours are not slacks.
    std::vector<int> role(static_cast<std::size_t>(n), 0);  // 1 slack, 2 multiplier
    std::vector<Pair> pairs;
    for (Int s = 0; s < n; ++s) {
        const auto &as = adj[static_cast<std::size_t>(s)];
        if (as.size() != 1 || diag[static_cast<std::size_t>(s)] < 0) continue;
        const Int y = as[0].first;
        if (role[static_cast<std::size_t>(y)] != 0 || role[static_cast<std::size_t>(s)] != 0) continue;
        const auto &ay = adj[static_cast<std::size_t>(y)];
        if (ay.size() > 9) continue;
        bool ok = true;
        for (const auto &[w, p] : ay)
            if (w != s && (adj[static_cast<std::size_t>(w)].size() == 1 || role[static_cast<std::size_t>(w)] != 0)) ok = false;
        if (!ok) continue;
        Pair pr;
        pr.s = s;
        pr.y = y;
        pr.ss = diag[static_cast<std::size_t>(s)];
        pr.ys = as[0].second;
        pr.yy = diag[static_cast<std::size_t>(y)];
        for (const auto &[w, p] : ay)
            if (w != s) {
                pr.b.push_back(p);
                pr.bx.push_back(w);  // original index for now
            }
        role[static_cast<std::size_t>(s)] = 1;
        role[static_cast<std::size_t>(y)] = 2;
        pairs.push_back(std::move(pr));
    }
    // States touched by a pair must not themselves be in a pair.
    std::vector<Int> red(static_cast<std::size_t>(n), -1);
    Int nr = 0;
    for (Int i = 0; i < n; ++i)
        if (role[static_cast<std::size_t>(i)] == 0) red[static_cast<std::size_t>(i)] = nr++;
    for (auto &pr : pairs)
        for (Int &w : pr.bx) w = red[static_cast<std::size_t>(w)];
    std::printf("pairs %zu, reduced n %d (%.1f%% of n)\n", pairs.size(), static_cast<int>(nr),
                100.0 * static_cast<double>(nr) / static_cast<double>(n));

    // Reduced pattern: A on the kept rows plus every pair's k x k block.
    std::map<std::pair<Int, Int>, long long> ent;  // (col, row) lower -> slot (assigned below)
    std::vector<std::pair<long long, std::pair<Int, Int>>> keep_src;  // A slot -> (col,row)
    for (Int j = 0; j < n; ++j) {
        const Int rj = red[static_cast<std::size_t>(j)];
        if (rj < 0) continue;
        for (Int p = A.col_ptr[static_cast<std::size_t>(j)]; p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
            const Int ri = red[static_cast<std::size_t>(A.row_idx[static_cast<std::size_t>(p)])];
            if (ri < 0) continue;
            ent[{rj, ri}] = -1;
            keep_src.push_back({p, {rj, ri}});
        }
    }
    for (const auto &pr : pairs)
        for (std::size_t a = 0; a < pr.bx.size(); ++a)
            for (std::size_t c = 0; c <= a; ++c) {
                const Int i = std::max(pr.bx[a], pr.bx[c]), j = std::min(pr.bx[a], pr.bx[c]);
                ent[{j, i}] = -1;
            }
    smf::CscLower R;
    R.n = nr;
    R.col_ptr.assign(static_cast<std::size_t>(nr) + 1, 0);
    {
        long long k = 0;
        for (auto &[key, slot] : ent) {
            slot = k++;
            R.row_idx.push_back(key.second);
            ++R.col_ptr[static_cast<std::size_t>(key.first) + 1];
        }
        for (Int j = 0; j < nr; ++j) R.col_ptr[static_cast<std::size_t>(j) + 1] += R.col_ptr[static_cast<std::size_t>(j)];
    }
    const std::size_t rnnz = R.row_idx.size();
    std::vector<long long> keep_dst(keep_src.size());
    for (std::size_t t = 0; t < keep_src.size(); ++t) keep_dst[t] = ent[keep_src[t].second];
    for (auto &pr : pairs)
        for (std::size_t a = 0; a < pr.bx.size(); ++a)
            for (std::size_t c = 0; c <= a; ++c) {
                const Int i = std::max(pr.bx[a], pr.bx[c]), j = std::min(pr.bx[a], pr.bx[c]);
                pr.blk.push_back(ent[{j, i}]);
            }
    std::printf("reduced nnz %zu (full %zu)\n", rnnz, nnz);

    smf::Control ctrl;
    ctrl.matrix_type = smf::MatrixType::RealSymmetricIndefinite;
    ctrl.ordering = smf::OrderingMethod::AutoParallel;
    ctrl.pivot_u = u;
    ctrl.num_threads = 1;

    // The pair pass: reduced values of a step; returns the pairs whose 2x2
    // block fails the threshold test. Each pair is eliminated as its 2x2
    // block P = [a_ss a_ys; a_ys a_yy] (exact whatever the slack diagonal,
    // also 0, as for a variable in one constraint only): the states get
    // -(P^-1)_yy b b^T, (P^-1)_yy = a_ss / det.
    std::vector<double> rv(rnnz);
    std::vector<double> iss(pairs.size()), isy(pairs.size()), iyy(pairs.size());
    int pos_pairs = 0, neg_pairs = 0;
    const auto pair_pass = [&](const std::vector<double> &v) {
        std::fill(rv.begin(), rv.end(), 0.0);
        for (std::size_t t = 0; t < keep_src.size(); ++t) rv[static_cast<std::size_t>(keep_dst[t])] += v[static_cast<std::size_t>(keep_src[t].first)];
        int fails = 0;
        pos_pairs = neg_pairs = 0;
        for (std::size_t q = 0; q < pairs.size(); ++q) {
            const Pair &pr = pairs[q];
            const double ass = v[static_cast<std::size_t>(pr.ss)], ays = v[static_cast<std::size_t>(pr.ys)];
            const double ayy = pr.yy >= 0 ? v[static_cast<std::size_t>(pr.yy)] : 0.0;
            const double det = ass * ayy - ays * ays;
            double bmax = 0.0;
            for (long long p : pr.b) bmax = std::max(bmax, std::abs(v[static_cast<std::size_t>(p)]));
            // Growth bound of the block pivot (as the TPP 2x2 test): the
            // multipliers |P^-1| times the off-block column entries must stay
            // below 1/u.
            const double g = (std::abs(ass) + std::abs(ays)) * bmax / std::abs(det);
            if (!(det != 0.0) || !(u * g <= 1.0)) ++fails;
            iss[q] = ayy / det;
            isy[q] = -ays / det;
            iyy[q] = ass / det;
            if (det < 0) { ++pos_pairs; ++neg_pairs; }
            else if (ass + ayy > 0) pos_pairs += 2;
            else neg_pairs += 2;
            const double w = iyy[q];
            std::size_t t = 0;
            for (std::size_t a2 = 0; a2 < pr.b.size(); ++a2) {
                const double ba = v[static_cast<std::size_t>(pr.b[a2])] * w;
                for (std::size_t c = 0; c <= a2; ++c)
                    rv[static_cast<std::size_t>(pr.blk[t++])] -= ba * v[static_cast<std::size_t>(pr.b[c])];
            }
        }
        return fails;
    };

    // Reduced factorization with smf; full with smf and MA27.
    auto red_ad = smf::bench::make_solver_adapter("smf");
    red_ad->configure(ctrl, "recommended");
    R.values = rv;
    pair_pass(seq[0]);
    R.values = rv;
    if (red_ad->analyse(R) != smf::bench::Outcome::Success) {
        std::fprintf(stderr, "reduced analyse failed\n");
        return 1;
    }
    double best_red = 1e300, best_pass = 1e300;
    int fails_total = 0;
    for (int r = 0; r < repeats; ++r) {
        double t_pass = 0, t_fac = 0;
        fails_total = 0;
        for (int st = 0; st < steps; ++st) {
            const double t0 = now_ms();
            fails_total += pair_pass(seq[static_cast<std::size_t>(st)]);
            const double t1 = now_ms();
            if (red_ad->factor(rv) != smf::bench::Outcome::Success) std::fprintf(stderr, "reduced factor failed at %d\n", st);
            t_fac += red_ad->stats().factor_ms;
            t_pass += t1 - t0;
        }
        best_red = std::min(best_red, t_pass + t_fac);
        best_pass = std::min(best_pass, t_pass);
    }
    std::printf("reduced: pair pass %.3f ms + smf factor %.3f ms per step (%.3f total); threshold fails %d\n",
                best_pass / steps, (best_red - best_pass) / steps, best_red / steps, fails_total);
    const auto rs = red_ad->stats();
    std::printf("reduced inertia +%d -%d (pairs +%d -%d)\n", rs.inertia_pos, rs.inertia_neg, pos_pairs, neg_pairs);

    for (const char *name : {"smf", "ma27"}) {
        auto ad = smf::bench::make_solver_adapter(name);
        if (!ad || !ad->identity().available) {
            std::printf("%s: unavailable\n", name);
            continue;
        }
        ad->configure(ctrl, "recommended");
        smf::CscLower F = A;
        F.values = seq[0];
        if (ad->analyse(F) != smf::bench::Outcome::Success) {
            std::printf("%s: analyse failed\n", name);
            continue;
        }
        double best = 1e300;
        for (int r = 0; r < repeats; ++r) {
            double t = 0;
            for (int st = 0; st < steps; ++st) {
                ad->factor(seq[static_cast<std::size_t>(st)]);
                t += ad->stats().factor_ms;
            }
            best = std::min(best, t);
        }
        const auto s = ad->stats();
        std::printf("full %s: factor %.3f ms per step; inertia +%d -%d\n", name, best / steps, s.inertia_pos, s.inertia_neg);
    }

    // Solve A x = b (b = 1, last step) through the reduced system and check
    // the residual on the full matrix.
    const std::vector<double> &v = seq.back();
    pair_pass(v);
    red_ad->factor(rv);
    std::vector<double> b(static_cast<std::size_t>(n), 1.0), x(static_cast<std::size_t>(n), 0.0);
    std::vector<double> rr(static_cast<std::size_t>(nr));
    for (Int i = 0; i < n; ++i)
        if (red[static_cast<std::size_t>(i)] >= 0) rr[static_cast<std::size_t>(red[static_cast<std::size_t>(i)])] = b[static_cast<std::size_t>(i)];
    // Reduced right-hand side r_r - b (P^-1 r_p)_y, then the pairs from
    // x_p = P^-1 (r_p - [0; b^T x_r]).
    for (std::size_t q = 0; q < pairs.size(); ++q) {
        const Pair &pr = pairs[q];
        const double bs = b[static_cast<std::size_t>(pr.s)], by = b[static_cast<std::size_t>(pr.y)];
        const double zy = isy[q] * bs + iyy[q] * by;
        for (std::size_t a2 = 0; a2 < pr.b.size(); ++a2)
            rr[static_cast<std::size_t>(pr.bx[a2])] -= v[static_cast<std::size_t>(pr.b[a2])] * zy;
    }
    red_ad->solve(rr, 1);
    for (Int i = 0; i < n; ++i)
        if (red[static_cast<std::size_t>(i)] >= 0) x[static_cast<std::size_t>(i)] = rr[static_cast<std::size_t>(red[static_cast<std::size_t>(i)])];
    for (std::size_t q = 0; q < pairs.size(); ++q) {
        const Pair &pr = pairs[q];
        double bx = 0.0;
        for (std::size_t a2 = 0; a2 < pr.b.size(); ++a2)
            bx += v[static_cast<std::size_t>(pr.b[a2])] * rr[static_cast<std::size_t>(pr.bx[a2])];
        const double rs = b[static_cast<std::size_t>(pr.s)], ry = b[static_cast<std::size_t>(pr.y)] - bx;
        x[static_cast<std::size_t>(pr.s)] = iss[q] * rs + isy[q] * ry;
        x[static_cast<std::size_t>(pr.y)] = isy[q] * rs + iyy[q] * ry;
    }
    smf::CscLower Al = A;
    Al.values = v;
    const auto val = smf::bench::validate(Al, b, x, 1);
    std::printf("reduced-path solution on the full KKT: eta_inf %.3e omega %.3e (%s)\n", val.eta_inf, val.omega,
                smf::bench::is_qualified_validation(val) ? "accurate" : "NOT accurate");
    {
        auto ad = smf::bench::make_solver_adapter("smf");
        ad->configure(ctrl, "recommended");
        ad->analyse(Al);
        ad->factor(v);
        std::vector<double> xf(b);
        ad->solve(xf, 1);
        const auto vf = smf::bench::validate(Al, b, xf, 1);
        double dmax = 0, xmax = 0;
        for (Int i = 0; i < n; ++i) {
            const double d = std::abs(xf[static_cast<std::size_t>(i)] - x[static_cast<std::size_t>(i)]);
            dmax = std::isfinite(d) ? std::max(dmax, d) : INFINITY;
            xmax = std::max(xmax, std::abs(xf[static_cast<std::size_t>(i)]));
        }
        std::printf("full smf solution: eta_inf %.3e omega %.3e; max |x_red - x_full| / max|x_full| = %.3e\n",
                    vf.eta_inf, vf.omega, dmax / xmax);
    }
    return 0;
}
