#include "smf/slack_elimination.hpp"

#include "smf/analysis.hpp"
#include "smf/check_matrix.hpp"
#include "smf/control.hpp"
#include "smf/info.hpp"
#include "smf/solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <utility>

namespace smf {

namespace {

// Largest number of states a multiplier may couple (k): the pair's state
// block adds k(k+1)/2 entries to the reduced matrix.
constexpr Int kMaxPairStates = 8;

} // namespace

std::shared_ptr<const SlackPlan> build_slack_plan(const CscLower &A, const Control &ctrl) {
    if (ctrl.slack_elimination == SlackElimination::Off ||
        ctrl.matrix_type != MatrixType::RealSymmetricIndefinite || A.n <= 0)
        return nullptr;
    const Int n = A.n;
    const std::size_t sn = static_cast<std::size_t>(n);
    // Cheap bound first (O(n)): a slack's lower column holds at most its
    // diagonal and one entry, so the pairs are at most the columns of at
    // most two entries. Most matrices stop here.
    const double need = ctrl.slack_elimination == SlackElimination::Auto ? ctrl.slack_min_share : 0.0;
    {
        Int shortcols = 0;
        for (Int j = 0; j < n; ++j)
            shortcols += A.col_ptr[static_cast<std::size_t>(j) + 1] - A.col_ptr[static_cast<std::size_t>(j)] <= 2;
        if (shortcols == 0 || 2.0 * static_cast<double>(shortcols) < need * static_cast<double>(n)) return nullptr;
    }
    // Symmetric adjacency (off-diagonal) with each edge's cleaned slot, and
    // the diagonal slots.
    std::vector<Int> deg(sn, 0);
    std::vector<LongInt> diag(sn, -1);
    for (Int j = 0; j < n; ++j)
        for (Int p = A.col_ptr[static_cast<std::size_t>(j)]; p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
            const Int i = A.row_idx[static_cast<std::size_t>(p)];
            if (i == j) diag[static_cast<std::size_t>(j)] = p;
            else {
                ++deg[static_cast<std::size_t>(i)];
                ++deg[static_cast<std::size_t>(j)];
            }
        }
    {
        Int cand = 0;  // degree-1 columns with a diagonal: the possible slacks
        for (std::size_t i = 0; i < sn; ++i) cand += deg[i] == 1 && diag[i] >= 0;
        if (cand == 0 || 2.0 * static_cast<double>(cand) < need * static_cast<double>(n)) return nullptr;
    }
    std::vector<LongInt> aptr(sn + 1, 0);
    for (std::size_t i = 0; i < sn; ++i) aptr[i + 1] = aptr[i] + deg[i];
    std::vector<Int> anb(static_cast<std::size_t>(aptr[sn]));
    std::vector<LongInt> aslot(static_cast<std::size_t>(aptr[sn]));
    {
        std::vector<LongInt> fill(aptr.begin(), aptr.end() - 1);
        for (Int j = 0; j < n; ++j)
            for (Int p = A.col_ptr[static_cast<std::size_t>(j)]; p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
                const Int i = A.row_idx[static_cast<std::size_t>(p)];
                if (i == j) continue;
                anb[static_cast<std::size_t>(fill[static_cast<std::size_t>(i)])] = j;
                aslot[static_cast<std::size_t>(fill[static_cast<std::size_t>(i)]++)] = p;
                anb[static_cast<std::size_t>(fill[static_cast<std::size_t>(j)])] = i;
                aslot[static_cast<std::size_t>(fill[static_cast<std::size_t>(j)]++)] = p;
            }
    }
    // Pairs: s of degree 1 with a stored diagonal; its neighbour y of degree
    // at most kMaxPairStates + 1, neither already in a pair, and none of y's
    // other neighbours a degree-1 column (another pair's slack).
    std::vector<std::int8_t> role(sn, 0);
    auto plan = std::make_shared<SlackPlan>();
    plan->n_full = n;
    for (Int s = 0; s < n; ++s) {
        const std::size_t ss = static_cast<std::size_t>(s);
        if (deg[ss] != 1 || diag[ss] < 0 || role[ss] != 0) continue;
        const Int y = anb[static_cast<std::size_t>(aptr[ss])];
        const std::size_t sy = static_cast<std::size_t>(y);
        if (role[sy] != 0 || deg[sy] > kMaxPairStates + 1) continue;
        bool ok = true;
        for (LongInt q = aptr[sy]; q < aptr[sy + 1] && ok; ++q) {
            const Int w = anb[static_cast<std::size_t>(q)];
            const std::int8_t rw = role[static_cast<std::size_t>(w)];
            if (w != s && (deg[static_cast<std::size_t>(w)] <= 1 || rw == 1 || rw == 2)) ok = false;
        }
        if (!ok) continue;
        SlackPair pr;
        pr.s = s;
        pr.y = y;
        pr.ss = diag[ss];
        pr.ys = aslot[static_cast<std::size_t>(aptr[ss])];
        pr.yy = diag[sy];
        pr.b0 = static_cast<LongInt>(plan->b_slot.size());
        for (LongInt q = aptr[sy]; q < aptr[sy + 1]; ++q) {
            const Int w = anb[static_cast<std::size_t>(q)];
            if (w == s) continue;
            plan->b_slot.push_back(aslot[static_cast<std::size_t>(q)]);
            plan->b_row.push_back(w);  // original index, mapped below
        }
        pr.nb = static_cast<Int>(static_cast<LongInt>(plan->b_slot.size()) - pr.b0);
        role[ss] = 1;
        role[sy] = 2;
        // Its states stay in the reduced matrix: never a later pair's slack
        // or multiplier.
        for (Int t = 0; t < pr.nb; ++t) role[static_cast<std::size_t>(plan->b_row[static_cast<std::size_t>(pr.b0 + t)])] = 3;
        plan->pairs.push_back(pr);
    }
    const double share = 2.0 * static_cast<double>(plan->pairs.size()) / static_cast<double>(n);
    if (plan->pairs.empty() ||
        (ctrl.slack_elimination == SlackElimination::Auto && share < ctrl.slack_min_share))
        return nullptr;

    std::vector<Int> red(sn, -1);
    for (Int i = 0; i < n; ++i)
        if (role[static_cast<std::size_t>(i)] == 0 || role[static_cast<std::size_t>(i)] == 3) {
            red[static_cast<std::size_t>(i)] = plan->nr++;
            plan->red_to_full.push_back(i);
        }
    for (Int &w : plan->b_row) w = red[static_cast<std::size_t>(w)];
    plan->sdiag.resize(static_cast<std::size_t>(plan->nr));
    for (Int r = 0; r < plan->nr; ++r)
        plan->sdiag[static_cast<std::size_t>(r)] = diag[static_cast<std::size_t>(plan->red_to_full[static_cast<std::size_t>(r)])];

    // Reduced pattern: the kept entries and every pair's lower state block,
    // column-major with rows ascending (its input order is its value order).
    // red[] keeps the order, so the kept entries of each column are sorted;
    // the blocks' entries are bucketed by column and merged in.
    const Int nr = plan->nr;
    const std::size_t snr = static_cast<std::size_t>(nr);
    std::vector<LongInt> kcnt(snr + 1, 0), bcnt(snr + 1, 0);
    for (Int j = 0; j < n; ++j) {
        const Int rj = red[static_cast<std::size_t>(j)];
        if (rj < 0) continue;
        for (Int p = A.col_ptr[static_cast<std::size_t>(j)]; p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p)
            if (red[static_cast<std::size_t>(A.row_idx[static_cast<std::size_t>(p)])] >= 0) ++kcnt[static_cast<std::size_t>(rj) + 1];
    }
    std::size_t nblk = 0;
    for (const SlackPair &pr : plan->pairs) nblk += static_cast<std::size_t>(pr.nb) * (pr.nb + 1) / 2;
    // Block entry e (in pair order, a >= c): its (col, row).
    std::vector<Int> bcol(nblk), brow(nblk);
    {
        std::size_t e = 0;
        for (const SlackPair &pr : plan->pairs)
            for (Int a2 = 0; a2 < pr.nb; ++a2)
                for (Int c = 0; c <= a2; ++c) {
                    const Int ra = plan->b_row[static_cast<std::size_t>(pr.b0 + a2)];
                    const Int rc = plan->b_row[static_cast<std::size_t>(pr.b0 + c)];
                    bcol[e] = std::min(ra, rc);
                    brow[e] = std::max(ra, rc);
                    ++bcnt[static_cast<std::size_t>(bcol[e]) + 1];
                    ++e;
                }
    }
    for (std::size_t j = 0; j < snr; ++j) {
        kcnt[j + 1] += kcnt[j];
        bcnt[j + 1] += bcnt[j];
    }
    std::vector<std::size_t> border(nblk);  // block entries bucketed by column
    {
        std::vector<LongInt> f(bcnt.begin(), bcnt.end() - 1);
        for (std::size_t e = 0; e < nblk; ++e) border[static_cast<std::size_t>(f[static_cast<std::size_t>(bcol[e])]++)] = e;
    }
    CscLower R;
    R.n = nr;
    R.col_ptr.assign(snr + 1, 0);
    R.row_idx.reserve(static_cast<std::size_t>(kcnt[snr]) + nblk);
    plan->keep_src.reserve(static_cast<std::size_t>(kcnt[snr]));
    plan->keep_dst.reserve(static_cast<std::size_t>(kcnt[snr]));
    std::vector<LongInt> bslot(nblk);
    std::vector<std::pair<Int, std::size_t>> col;  // (row, block entry) of one column
    for (Int j = 0; j < n; ++j) {
        const Int rj = red[static_cast<std::size_t>(j)];
        if (rj < 0) continue;
        col.clear();
        for (LongInt t = bcnt[static_cast<std::size_t>(rj)]; t < bcnt[static_cast<std::size_t>(rj) + 1]; ++t) {
            const std::size_t e = border[static_cast<std::size_t>(t)];
            col.push_back({brow[e], e});
        }
        std::sort(col.begin(), col.end());
        std::size_t q = 0;
        Int last = -1;
        const auto emit_blocks_below = [&](Int bound) {
            for (; q < col.size() && col[q].first < bound; ++q) {
                if (col[q].first != last) {
                    R.row_idx.push_back(col[q].first);
                    last = col[q].first;
                }
                bslot[col[q].second] = static_cast<LongInt>(R.row_idx.size()) - 1;
            }
        };
        for (Int p = A.col_ptr[static_cast<std::size_t>(j)]; p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
            const Int ri = red[static_cast<std::size_t>(A.row_idx[static_cast<std::size_t>(p)])];
            if (ri < 0) continue;
            emit_blocks_below(ri);
            R.row_idx.push_back(ri);
            last = ri;
            plan->keep_src.push_back(p);
            plan->keep_dst.push_back(static_cast<LongInt>(R.row_idx.size()) - 1);
            for (; q < col.size() && col[q].first == ri; ++q) bslot[col[q].second] = static_cast<LongInt>(R.row_idx.size()) - 1;
        }
        emit_blocks_below(std::numeric_limits<Int>::max());
        R.col_ptr[static_cast<std::size_t>(rj) + 1] = static_cast<Int>(R.row_idx.size());
    }
    plan->reduced_nnz = R.row_idx.size();
    plan->blk = std::move(bslot);
    {
        LongInt e = 0;
        for (SlackPair &pr : plan->pairs) {
            pr.blk0 = e;
            e += static_cast<LongInt>(pr.nb) * (pr.nb + 1) / 2;
        }
    }

    // The reduced matrix, valued as the analysed matrix reduces (the matching
    // trial and scaling decisions of its analysis read them).
    SlackFactor sf;
    std::vector<double> rv;
    int pos = 0, neg = 0;
    Control rc = ctrl;
    rc.slack_elimination = SlackElimination::Off;
    if (!slack_pair_pass(*plan, A.values.data(), ctrl, rv, sf, pos, neg)) {
        // Not usable on the analysed values: the reduced matrix keeps their
        // full-matrix values where it can (the pattern is what matters).
        rv.assign(plan->reduced_nnz, 0.0);
        for (std::size_t t = 0; t < plan->keep_src.size(); ++t)
            rv[static_cast<std::size_t>(plan->keep_dst[t])] = A.values[static_cast<std::size_t>(plan->keep_src[t])];
    }
    R.values = std::move(rv);
    Solver solver;
    Info ri;
    auto rk = solver.analyse(R, rc, ri);
    if (!rk) return nullptr;
    plan->reduced = std::move(rk);
    return plan;
}

std::unique_ptr<AnalysisKeep> analyse_with_slack_plan(Solver &solver, const CscLower &A,
                                                      const Control &ctrl, Info &info) {
    if (ctrl.slack_elimination == SlackElimination::Off ||
        ctrl.matrix_type != MatrixType::RealSymmetricIndefinite || !(ctrl.pivot_u <= kSlackMaxPivotU) ||
        (ctrl.ordering != OrderingMethod::AutoSerial && ctrl.ordering != OrderingMethod::AutoParallel) ||
        A.n < kSlackMinOrder || A.col_ptr.size() != static_cast<std::size_t>(A.n) + 1)
        return nullptr;
    // The O(n) bound on the input's columns first (a slack's lower column
    // holds at most two entries), so that matrices without enough pairs pay
    // nothing more.
    const double need = ctrl.slack_elimination == SlackElimination::Auto ? ctrl.slack_min_share : 0.0;
    {
        Int shortcols = 0;
        for (Int j = 0; j < A.n; ++j)
            shortcols += A.col_ptr[static_cast<std::size_t>(j) + 1] - A.col_ptr[static_cast<std::size_t>(j)] <= 2;
        if (shortcols == 0 || 2.0 * static_cast<double>(shortcols) < need * static_cast<double>(A.n)) return nullptr;
    }
    const auto t0 = std::chrono::steady_clock::now();
    Info ci;
    const CleanedPattern cp = clean_lower_csc(A, ci, ctrl.check_input);
    if (cp.clean.n != A.n) return nullptr;
    auto cplan = build_slack_plan(cp.clean, ctrl);
    if (!cplan || cplan->nr <= 0) return nullptr;
    auto plan = std::const_pointer_cast<SlackPlan>(cplan);
    plan->input = A;
    plan->clean_ptr.assign(cp.clean.col_ptr.begin(), cp.clean.col_ptr.end());
    plan->clean_row.assign(cp.clean.row_idx.begin(), cp.clean.row_idx.end());
    plan->actrl = ctrl;
    plan->actrl.slack_elimination = SlackElimination::Off;
    plan->actrl.user_scale = nullptr;
    plan->actrl.reg_diag_sign = nullptr;
    // The full matrix's analysis, for the factorizations that take it: the
    // pairs first (each a 2x2 pivot pair: slack, multiplier), then the
    // reduced matrix's order. No ordering, matching or METIS runs for it.
    const std::size_t n = static_cast<std::size_t>(A.n);
    std::vector<Int> perm;
    perm.reserve(n);
    std::vector<Int> pf;
    pf.reserve(2 * plan->pairs.size());
    for (const SlackPair &pr : plan->pairs) {
        perm.push_back(pr.s);
        perm.push_back(pr.y);
        pf.push_back(pr.s);
        pf.push_back(pr.y);
    }
    for (const Int r : plan->reduced->perm) perm.push_back(plan->red_to_full[static_cast<std::size_t>(r)]);
    if (perm.size() != n) return nullptr;
    Control fc = ctrl;
    fc.slack_elimination = SlackElimination::Off;
    fc.ordering = OrderingMethod::UserSupplied;
    fc.user_perm = perm.data();
    fc.user_pair_first = pf.data();
    fc.user_num_pairs = static_cast<Int>(plan->pairs.size());
    const double plan_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    auto keep = solver.analyse(A, fc, info);
    if (!keep) return nullptr;
    keep->slack = std::move(plan);
    info.slack_pairs = static_cast<int>(keep->slack->pairs.size());
    info.analyse_seconds += plan_seconds;
    return keep;
}

const AnalysisKeep *slack_plan_full_analysis(const SlackPlan &plan) {
    std::lock_guard<std::mutex> lock(plan.full_mu);
    if (!plan.full) {
        Info fi;
        plan.full = Solver().analyse(plan.input, plan.actrl, fi);
    }
    return plan.full.get();
}

const AnalysisKeep &slack_full_analysis(const AnalysisKeep &keep) {
    if (!keep.slack) return keep;
    const AnalysisKeep *full = slack_plan_full_analysis(*keep.slack);
    return full ? *full : keep;
}

bool slack_owns(const AnalysisKeep &keep, const AnalysisKeep *analysis) {
    if (!keep.slack || analysis == nullptr) return false;
    const SlackPlan &plan = *keep.slack;
    if (analysis == plan.reduced.get()) return true;
    std::lock_guard<std::mutex> lock(plan.full_mu);
    return analysis == plan.full.get();
}

bool slack_armed(const AnalysisKeep &keep, const Control &ctrl) {
    return keep.slack != nullptr && ctrl.slack_elimination != SlackElimination::Off &&
           ctrl.matrix_type == MatrixType::RealSymmetricIndefinite && ctrl.pivoting == Pivoting::Threshold &&
           !(ctrl.static_pivot_tol > 0.0) && ctrl.reg_diag_sign == nullptr &&
           ctrl.scaling != ScalingMethod::UserSupplied && ctrl.max_negative_pivots < 0 &&
           ctrl.pivot_u <= kSlackMaxPivotU;
}

bool slack_pair_pass(const SlackPlan &plan, const double *cv, const Control &ctrl,
                     std::vector<double> &rv, SlackFactor &sf, int &pos, int &neg) {
    rv.assign(plan.reduced_nnz, 0.0);
    for (std::size_t t = 0; t < plan.keep_src.size(); ++t)
        rv[static_cast<std::size_t>(plan.keep_dst[t])] = cv[static_cast<std::size_t>(plan.keep_src[t])];
    const std::size_t np = plan.pairs.size();
    sf.fails = 0;
    sf.max_growth = 0.0;
    sf.max_slack = 0.0;
    sf.pinv.resize(3 * np);
    sf.bval.resize(plan.b_slot.size());
    // The guard (Control::slack_update_limit): a pair whose condensed update
    // |(P^-1)_yy| b_a^2 exceeds the limit times the nonzero state diagonal it
    // lands on would leave that diagonal below roundoff in the reduced
    // matrix; the factorization then takes the full matrix.
    const double limit = ctrl.slack_update_limit > 0.0 ? ctrl.slack_update_limit
                                                       : std::numeric_limits<double>::infinity();
    for (std::size_t q = 0; q < np; ++q) {
        const SlackPair &pr = plan.pairs[q];
        const double ass = cv[static_cast<std::size_t>(pr.ss)];
        const double ays = cv[static_cast<std::size_t>(pr.ys)];
        const double ayy = pr.yy >= 0 ? cv[static_cast<std::size_t>(pr.yy)] : 0.0;
        // The 2x2 inverse. In the common range the determinant is formed
        // directly; a block whose largest entry is near under- or overflow is
        // first normalized by it (as the LDL^T kernel's 2x2 pivots), so that
        // the determinant does not underflow. A determinant lost to
        // cancellation (or not finite) fails the pair.
        const double amax = std::max(std::max(std::abs(ass), std::abs(ays)), std::abs(ayy));
        const double sc = (amax > 1e-150 && amax < 1e150) ? 1.0 : (amax > 0.0 ? 1.0 / amax : 0.0);
        const double hs = ass * sc, hsy = ays * sc, hy = ayy * sc;
        const double hdet = hs * hy - hsy * hsy;
        const bool det_ok = sc > 0.0 && std::isfinite(sc) &&
                            std::abs(hdet) > 1e3 * std::numeric_limits<double>::epsilon() *
                                                 (std::abs(hs * hy) + hsy * hsy);
        const double det = hdet;  // its sign (the pair's inertia)
        for (Int t = 0; t < pr.nb; ++t)
            sf.bval[static_cast<std::size_t>(pr.b0 + t)] =
                cv[static_cast<std::size_t>(plan.b_slot[static_cast<std::size_t>(pr.b0 + t)])];
        sf.max_slack = std::max(sf.max_slack, std::abs(ass));
        if (!det_ok) {
            ++sf.fails;
            continue;
        }
        const double inv = sc / hdet;  // P^-1 = sc adj(P sc) / det(P sc)
        const double iss = hy * inv, isy = -hsy * inv, iyy = hs * inv;
        if (!std::isfinite(inv)) {
            ++sf.fails;
            continue;
        }
        double ratio = 0.0;
        for (Int t = 0; t < pr.nb; ++t) {
            const LongInt ds = plan.sdiag[static_cast<std::size_t>(plan.b_row[static_cast<std::size_t>(pr.b0 + t)])];
            const double d = ds >= 0 ? std::abs(cv[static_cast<std::size_t>(ds)]) : 0.0;
            const double bv = sf.bval[static_cast<std::size_t>(pr.b0 + t)];
            if (d > 0.0) ratio = std::max(ratio, (std::abs(iyy) * std::abs(bv)) * std::abs(bv) / d);
        }
        sf.max_growth = std::max(sf.max_growth, ratio);
        if (!(ratio <= limit)) {
            ++sf.fails;
            continue;
        }
        sf.pinv[3 * q] = iss;
        sf.pinv[3 * q + 1] = isy;
        sf.pinv[3 * q + 2] = iyy;
        if (det < 0.0) {
            ++pos;
            ++neg;
        } else if (ass + ayy > 0.0) {
            pos += 2;
        } else {
            neg += 2;
        }
        const LongInt *blk = plan.blk.data() + pr.blk0;
        std::size_t t = 0;
        for (Int a = 0; a < pr.nb; ++a) {
            const double ba = sf.bval[static_cast<std::size_t>(pr.b0 + a)] * iyy;
            for (Int c = 0; c <= a; ++c)
                rv[static_cast<std::size_t>(blk[t++])] -= ba * sf.bval[static_cast<std::size_t>(pr.b0 + c)];
        }
    }
    return sf.fails == 0;
}

} // namespace smf
