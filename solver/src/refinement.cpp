#include "smf/refinement.hpp"
#include "smf/solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace smf {

namespace {

struct NeumaierAcc {
    long double sum = 0.0L;
    long double c = 0.0L;

    void add(long double val) noexcept {
        long double t = sum + val;
        if (std::fabsl(sum) >= std::fabsl(val)) {
            c += (sum - t) + val;
        } else {
            c += (val - t) + sum;
        }
        sum = t;
    }

    long double result() const noexcept { return sum + c; }
};

struct CanonicalSymEntry {
    Int row = 0;
    Int col = 0;
    long double val = 0.0L;
};

constexpr double kTargetEtaInf = 1e-10;
constexpr double kTargetOmega = 1e-8;

bool compute_residual_and_backward_error(
    int n,
    int nrhs,
    const std::vector<CanonicalSymEntry> &entries,
    long double norm_A_inf,
    std::span<const double> b,
    std::span<const double> x,
    std::vector<double> &r_out,
    double &eta_inf_out,
    double &omega_out) {
    const double inf = std::numeric_limits<double>::infinity();
    eta_inf_out = 0.0;
    omega_out = 0.0;
    r_out.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(nrhs), 0.0);

    for (int rhs = 0; rhs < nrhs; ++rhs) {
        const double *b_col = b.data() + static_cast<std::size_t>(rhs) * n;
        const double *x_col = x.data() + static_cast<std::size_t>(rhs) * n;
        double *r_col = r_out.data() + static_cast<std::size_t>(rhs) * n;

        long double norm_b_inf = 0.0L;
        long double norm_x_inf = 0.0L;
        for (int i = 0; i < n; ++i) {
            if (!std::isfinite(b_col[i]) || !std::isfinite(x_col[i])) {
                eta_inf_out = inf;
                omega_out = inf;
                return false;
            }
            norm_b_inf = std::max(norm_b_inf, std::fabsl(static_cast<long double>(b_col[i])));
            norm_x_inf = std::max(norm_x_inf, std::fabsl(static_cast<long double>(x_col[i])));
        }

        std::vector<NeumaierAcc> ax(static_cast<std::size_t>(n));
        std::vector<NeumaierAcc> abs_ax(static_cast<std::size_t>(n));

        for (const auto &e : entries) {
            const std::size_t r = static_cast<std::size_t>(e.row);
            const std::size_t c = static_cast<std::size_t>(e.col);
            const long double v = e.val;
            const long double av = std::fabsl(v);
            const long double xc = static_cast<long double>(x_col[c]);
            const long double xr = static_cast<long double>(x_col[r]);
            ax[r].add(v * xc);
            abs_ax[r].add(av * std::fabsl(xc));
            if (r != c) {
                ax[c].add(v * xr);
                abs_ax[c].add(av * std::fabsl(xr));
            }
        }

        long double col_res_inf = 0.0L;
        long double col_omega = 0.0L;
        for (int i = 0; i < n; ++i) {
            const std::size_t idx = static_cast<std::size_t>(i);
            const long double bi = static_cast<long double>(b_col[i]);
            const long double res_i = bi - ax[idx].result();
            r_col[i] = static_cast<double>(res_i);
            const long double abs_res = std::fabsl(res_i);
            col_res_inf = std::max(col_res_inf, abs_res);

            const long double comp_denom = abs_ax[idx].result() + std::fabsl(bi);
            long double om_i = 0.0L;
            if (comp_denom > 0.0L) {
                om_i = abs_res / comp_denom;
            } else if (abs_res == 0.0L) {
                om_i = 0.0L;
            } else {
                eta_inf_out = inf;
                omega_out = inf;
                return false;
            }
            col_omega = std::max(col_omega, om_i);
        }

        const long double eta_denom = norm_A_inf * norm_x_inf + norm_b_inf;
        long double col_eta = 0.0L;
        if (eta_denom > 0.0L) {
            col_eta = col_res_inf / eta_denom;
        } else if (col_res_inf == 0.0L) {
            col_eta = 0.0L;
        } else {
            eta_inf_out = inf;
            omega_out = inf;
            return false;
        }

        eta_inf_out = std::max(eta_inf_out, static_cast<double>(col_eta));
        omega_out = std::max(omega_out, static_cast<double>(col_omega));
    }

    return std::isfinite(eta_inf_out) && std::isfinite(omega_out);
}

} // namespace

RefinementResult refine(const CscLower &original,
                        const FactorKeep &fkeep,
                        const Control &ctrl,
                        std::span<const double> b,
                        std::span<double> x,
                        int nrhs,
                        SolveWorkspace &ws) {
    const double inf = std::numeric_limits<double>::infinity();
    const int n = static_cast<int>(original.n);
    if (n <= 0 || nrhs <= 0 || original.validate_shape() != ErrorCode::Success) {
        return RefinementResult{ErrorCode::IllegalValue, 0, inf, inf};
    }
    const std::size_t total_len = static_cast<std::size_t>(n) * static_cast<std::size_t>(nrhs);
    if (b.size() < total_len || x.size() < total_len) {
        return RefinementResult{ErrorCode::IllegalValue, 0, inf, inf};
    }

    std::vector<CanonicalSymEntry> raw_entries;
    raw_entries.reserve(original.values.size());
    for (int j = 0; j < n; ++j) {
        for (Int p = original.col_ptr[static_cast<std::size_t>(j)];
             p < original.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
            const Int i = original.row_idx[static_cast<std::size_t>(p)];
            const double v = original.values[static_cast<std::size_t>(p)];
            if (!std::isfinite(v)) {
                return RefinementResult{ErrorCode::IllegalValue, 0, inf, inf};
            }
            if (i >= j) {
                raw_entries.push_back({i, j, static_cast<long double>(v)});
            } else {
                raw_entries.push_back({j, i, static_cast<long double>(v)});
            }
        }
    }
    std::sort(raw_entries.begin(), raw_entries.end(), [](const auto &a, const auto &b_e) {
        if (a.col != b_e.col) return a.col < b_e.col;
        return a.row < b_e.row;
    });

    std::vector<CanonicalSymEntry> entries;
    entries.reserve(raw_entries.size());
    for (const auto &e : raw_entries) {
        if (!entries.empty() && entries.back().row == e.row && entries.back().col == e.col) {
            entries.back().val += e.val;
        } else {
            entries.push_back(e);
        }
    }

    std::vector<NeumaierAcc> row_abs(static_cast<std::size_t>(n));
    for (const auto &e : entries) {
        const long double av = std::fabsl(e.val);
        row_abs[static_cast<std::size_t>(e.row)].add(av);
        if (e.row != e.col) {
            row_abs[static_cast<std::size_t>(e.col)].add(av);
        }
    }
    long double norm_A_inf = 0.0L;
    for (int i = 0; i < n; ++i) {
        norm_A_inf = std::max(norm_A_inf, row_abs[static_cast<std::size_t>(i)].result());
    }

    double best_eta = inf;
    double best_omega = inf;
    if (!compute_residual_and_backward_error(
            n, nrhs, entries, norm_A_inf, b, x, ws.refine_r, best_eta, best_omega)) {
        return RefinementResult{ErrorCode::IllegalValue, 0, best_eta, best_omega};
    }

    if (best_eta <= kTargetEtaInf && best_omega <= kTargetOmega) {
        return RefinementResult{ErrorCode::Success, 0, best_eta, best_omega};
    }

    const int max_iters = std::max(0, ctrl.refinement_max_iters);
    if (max_iters == 0) {
        return RefinementResult{ErrorCode::MaxIterations, 0, best_eta, best_omega};
    }

    ws.refine_best_x.assign(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(total_len));
    Solver solver;
    Control solve_ctrl = ctrl;
    solve_ctrl.refinement_max_iters = 0;

    int iters_done = 0;
    int stagnated_steps = 0;
    auto metric_score = [](double eta, double om) noexcept -> double {
        return std::max(eta / kTargetEtaInf, om / kTargetOmega);
    };
    double best_score = metric_score(best_eta, best_omega);

    for (int iter = 1; iter <= max_iters; ++iter) {
        iters_done = iter;
        ws.refine_dx = ws.refine_r;
        Info step_info;
        const int rc = solver.solve(
            fkeep, solve_ctrl, step_info, ws.refine_dx.data(), n, nrhs, ws, SolveJob::Full);
        if (rc != 0) {
            break;
        }

        for (std::size_t idx = 0; idx < total_len; ++idx) {
            x[idx] = ws.refine_best_x[idx] + ws.refine_dx[idx];
        }

        double cand_eta = inf;
        double cand_omega = inf;
        if (!compute_residual_and_backward_error(
                n, nrhs, entries, norm_A_inf, b, x, ws.refine_r, cand_eta, cand_omega)) {
            std::copy(ws.refine_best_x.begin(), ws.refine_best_x.end(), x.begin());
            break;
        }

        const double cand_score = metric_score(cand_eta, cand_omega);
        if (cand_score < best_score) {
            const double ratio = cand_score / best_score;
            best_eta = cand_eta;
            best_omega = cand_omega;
            best_score = cand_score;
            ws.refine_best_x.assign(
                x.begin(), x.begin() + static_cast<std::ptrdiff_t>(total_len));

            if (best_eta <= kTargetEtaInf && best_omega <= kTargetOmega) {
                return RefinementResult{ErrorCode::Success, iters_done, best_eta, best_omega};
            }
            if (ratio > 0.85) {
                ++stagnated_steps;
                if (stagnated_steps >= 2) {
                    break;
                }
            } else {
                stagnated_steps = 0;
            }
        } else {
            // Candidate was worse than best_x: restore best_x and stop on stagnation
            std::copy(ws.refine_best_x.begin(), ws.refine_best_x.end(), x.begin());
            break;
        }
    }

    const bool qualified =
        std::isfinite(best_eta) && std::isfinite(best_omega) &&
        best_eta <= kTargetEtaInf && best_omega <= kTargetOmega;
    return RefinementResult{
        qualified ? ErrorCode::Success : ErrorCode::MaxIterations,
        iters_done,
        best_eta,
        best_omega};
}

} // namespace smf
