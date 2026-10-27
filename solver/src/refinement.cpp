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

using CanonicalSymEntry = RefineEntry;

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

long double norm_inf_of(int n, const std::vector<RefineEntry> &entries) {
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
    return norm_A_inf;
}

RefinementResult refine_core(int n,
                             const std::vector<RefineEntry> &entries,
                             long double norm_A_inf,
                             const FactorKeep &fkeep,
                             const Control &ctrl,
                             std::span<const double> b,
                             std::span<double> x,
                             int nrhs,
                             SolveWorkspace &ws);

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

    return refine_core(n, entries, norm_inf_of(n, entries), fkeep, ctrl, b, x, nrhs, ws);
}

RefinementResult refine(const RefineMatrix &original,
                        const FactorKeep &fkeep,
                        const Control &ctrl,
                        std::span<const double> b,
                        std::span<double> x,
                        int nrhs,
                        SolveWorkspace &ws) {
    const double inf = std::numeric_limits<double>::infinity();
    const int n = static_cast<int>(original.n);
    if (n <= 0 || nrhs <= 0 || !original.finite) {
        return RefinementResult{ErrorCode::IllegalValue, 0, inf, inf};
    }
    const std::size_t total_len = static_cast<std::size_t>(n) * static_cast<std::size_t>(nrhs);
    if (b.size() < total_len || x.size() < total_len) {
        return RefinementResult{ErrorCode::IllegalValue, 0, inf, inf};
    }
    return refine_core(n, original.entries, original.norm_inf, fkeep, ctrl, b, x, nrhs, ws);
}

ErrorCode prepare_refine_matrix(const CscLower &A, RefineMatrix &m) {
    m = RefineMatrix{};
    if (A.n <= 0 || A.validate_shape() != ErrorCode::Success) return ErrorCode::IllegalValue;
    const int n = static_cast<int>(A.n);
    struct Item {
        Int row, col, k;
    };
    std::vector<Item> items;
    items.reserve(A.row_idx.size());
    for (int j = 0; j < n; ++j) {
        for (Int p = A.col_ptr[static_cast<std::size_t>(j)];
             p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
            const Int i = A.row_idx[static_cast<std::size_t>(p)];
            items.push_back(i >= j ? Item{i, static_cast<Int>(j), p} : Item{static_cast<Int>(j), i, p});
        }
    }
    std::stable_sort(items.begin(), items.end(), [](const Item &a, const Item &c) {
        return a.col != c.col ? a.col < c.col : a.row < c.row;
    });
    m.n = A.n;
    m.slot.assign(items.size(), 0);
    m.entries.reserve(items.size());
    for (const Item &it : items) {
        if (m.entries.empty() || m.entries.back().row != it.row || m.entries.back().col != it.col)
            m.entries.push_back({it.row, it.col, 0.0L});
        m.slot[static_cast<std::size_t>(it.k)] = static_cast<Int>(m.entries.size() - 1);
    }
    m.finite = false;  // until set_refine_values
    return ErrorCode::Success;
}

void set_refine_values(std::span<const double> values, RefineMatrix &m) {
    for (auto &e : m.entries) e.val = 0.0L;
    bool finite = values.size() == m.slot.size();
    if (finite) {
        for (std::size_t k = 0; k < values.size(); ++k) {
            const double v = values[k];
            finite = finite && std::isfinite(v);
            m.entries[static_cast<std::size_t>(m.slot[k])].val += static_cast<long double>(v);
        }
    }
    m.finite = finite;
    m.norm_inf = finite ? norm_inf_of(static_cast<int>(m.n), m.entries) : 0.0L;
}

namespace {

double metric_score(double eta, double om) noexcept {
    return std::max(eta / kTargetEtaInf, om / kTargetOmega);
}

bool meets_gate(double eta, double om) noexcept {
    return eta <= kTargetEtaInf && om <= kTargetOmega;
}

// Classical refinement x += M^-1 (b - A x) with the factor M. `cap` steps at
// most. Classical mode (escalate == false) is the refinement as it was before
// FGMRES: stop after two steps that contract the score by less than 0.85 or
// on one that does not improve it. Auto mode (escalate == true) instead asks
// for FGMRES (ClassicalOutcome::escalate) when a step contracts by less than
// 0.5, or, on a statically pivoted factor, when the gate is still not met
// after min(3, cap) steps. x is left at the best iterate.
struct ClassicalOutcome {
    RefinementResult result;
    bool escalate = false;
};

ClassicalOutcome classical_refine(int n,
                                  const std::vector<RefineEntry> &entries,
                                  long double norm_A_inf,
                                  const FactorKeep &fkeep,
                                  const Control &solve_ctrl,
                                  std::span<const double> b,
                                  std::span<double> x,
                                  int nrhs,
                                  SolveWorkspace &ws,
                                  double best_eta,
                                  double best_omega,
                                  int cap,
                                  bool escalate) {
    const double inf = std::numeric_limits<double>::infinity();
    const std::size_t total_len = static_cast<std::size_t>(n) * static_cast<std::size_t>(nrhs);
    const int static_check = std::min(3, cap);
    ClassicalOutcome out;

    ws.refine_best_x.assign(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(total_len));
    Solver solver;

    int iters_done = 0;
    int stagnated_steps = 0;
    double best_score = metric_score(best_eta, best_omega);

    for (int iter = 1; iter <= cap; ++iter) {
        iters_done = iter;
        ws.refine_dx = ws.refine_r;
        Info step_info;
        // All nrhs corrections in one call: the serial solve takes each
        // front once for every column (BLAS-3, blocked multi-RHS solve).
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

            if (meets_gate(best_eta, best_omega)) {
                out.result = RefinementResult{ErrorCode::Success, iters_done, best_eta, best_omega};
                return out;
            }
            if (escalate) {
                if (ratio > 0.5 || (fkeep.static_pivots > 0 && iter >= static_check)) {
                    out.escalate = true;
                    break;
                }
            } else if (ratio > 0.85) {
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
            out.escalate = escalate;
            break;
        }
    }

    const bool qualified =
        std::isfinite(best_eta) && std::isfinite(best_omega) && meets_gate(best_eta, best_omega);
    out.result = RefinementResult{
        qualified ? ErrorCode::Success : ErrorCode::MaxIterations,
        iters_done,
        best_eta,
        best_omega};
    return out;
}

double dot(const double *a, const double *b, int n) noexcept {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

// y = A z for the symmetric matrix (lower canonical entries), in FP64.
void sym_matvec(int n, const std::vector<RefineEntry> &entries, const double *z, double *y) {
    std::fill(y, y + n, 0.0);
    for (const auto &e : entries) {
        const double v = static_cast<double>(e.val);
        y[e.row] += v * z[e.col];
        if (e.row != e.col) y[e.col] += v * z[e.row];
    }
}

// Right-preconditioned FGMRES(restart) on one right-hand side, in FP64, with
// the factor as the preconditioner (one solve per iteration) and modified
// Gram-Schmidt. Restarts from the best iterate with its extended-precision
// residual. After every iteration the iterate is formed and both backward
// errors of the true residual are checked: it stops as soon as both meet the
// gate, after `budget` iterations, on a breakdown with no improvement, a
// failed solve, or a restart cycle that did not improve the best iterate.
// An iterate whose ||x||inf exceeds kMaxGrowth times the reference
// max(||x_start||inf, ||b||inf / ||A||inf) is never accepted: on a singular,
// inconsistent system the least-squares iterates run off along the null
// space, where both backward errors can be driven under the gate by an x of
// absurd norm (1e29 on a pressure-null-space Stokes system). A legitimate
// correction of a static-pivoted factor grows x by at most about
// lambda(A + E) / lambda(A), beyond 1e8 only when A is numerically singular.
// x_col is left at the best iterate; returns the iterations used.
constexpr double kMaxGrowth = 1e8;

int fgmres_column(int n,
                  const std::vector<RefineEntry> &entries,
                  long double norm_A_inf,
                  const FactorKeep &fkeep,
                  const Control &solve_ctrl,
                  std::span<const double> b_col,
                  std::span<double> x_col,
                  int budget,
                  int restart,
                  SolveWorkspace &ws,
                  double &eta_out,
                  double &omega_out) {
    const double inf = std::numeric_limits<double>::infinity();
    const std::size_t un = static_cast<std::size_t>(n);
    std::vector<double> r_best;
    if (!compute_residual_and_backward_error(
            n, 1, entries, norm_A_inf, b_col, x_col, r_best, eta_out, omega_out)) {
        eta_out = inf;
        omega_out = inf;
        return 0;
    }
    if (meets_gate(eta_out, omega_out) || budget <= 0) return 0;

    double x_ref = 0.0;
    for (double v : x_col) x_ref = std::max(x_ref, std::abs(v));
    if (norm_A_inf > 0.0L) {
        double b_inf = 0.0;
        for (double v : b_col) b_inf = std::max(b_inf, std::abs(v));
        x_ref = std::max(x_ref, b_inf / static_cast<double>(norm_A_inf));
    }
    const double x_limit = kMaxGrowth * x_ref;

    // A cycle never runs past the budget: size the basis for what can run.
    const int m_max = std::max(1, std::min(restart, budget));
    const std::size_t mm = static_cast<std::size_t>(m_max);
    std::vector<double> V((mm + 1) * un);  // Krylov basis v_0..v_m
    std::vector<double> Z(mm * un);        // preconditioned directions z_j = M^-1 v_j
    std::vector<double> H((mm + 1) * mm);  // Hessenberg, column-major, rotated to R in place
    std::vector<double> cs(mm), sn(mm), g(mm + 1), y(mm);
    std::vector<double> x0(x_col.begin(), x_col.end());
    std::vector<double> xc(un), r_cand;
    auto h = [&](int i, int j) -> double & {
        return H[static_cast<std::size_t>(j) * (mm + 1) + static_cast<std::size_t>(i)];
    };

    Solver solver;
    double best_score = metric_score(eta_out, omega_out);
    int used = 0;
    bool done = false;
    while (used < budget && !done) {
        // r_best is the true residual of x0 (= the best iterate).
        const double beta = std::sqrt(dot(r_best.data(), r_best.data(), n));
        if (!(beta > 0.0) || !std::isfinite(beta)) break;
        for (std::size_t i = 0; i < un; ++i) V[i] = r_best[i] / beta;
        std::fill(g.begin(), g.end(), 0.0);
        g[0] = beta;
        const int m = std::min(m_max, budget - used);
        bool improved = false;
        bool stop = false;
        for (int j = 0; j < m; ++j) {
            const double *vj = V.data() + static_cast<std::size_t>(j) * un;
            double *zj = Z.data() + static_cast<std::size_t>(j) * un;
            std::copy(vj, vj + n, zj);
            Info step_info;
            ++used;
            if (solver.solve(fkeep, solve_ctrl, step_info, zj, n, 1, ws, SolveJob::Full) != 0) {
                stop = true;
                break;
            }
            double *w = V.data() + static_cast<std::size_t>(j + 1) * un;
            sym_matvec(n, entries, zj, w);
            for (int i = 0; i <= j; ++i) {
                const double *vi = V.data() + static_cast<std::size_t>(i) * un;
                const double hij = dot(w, vi, n);
                h(i, j) = hij;
                for (int k = 0; k < n; ++k) w[k] -= hij * vi[k];
            }
            const double hnext = std::sqrt(dot(w, w, n));
            if (!std::isfinite(hnext)) {
                stop = true;
                break;
            }
            // Previous rotations on column j, then a new one zeroing h(j+1, j).
            for (int i = 0; i < j; ++i) {
                const double a = h(i, j);
                const double c = h(i + 1, j);
                h(i, j) = cs[static_cast<std::size_t>(i)] * a + sn[static_cast<std::size_t>(i)] * c;
                h(i + 1, j) = -sn[static_cast<std::size_t>(i)] * a + cs[static_cast<std::size_t>(i)] * c;
            }
            const double diag = h(j, j);
            const double rho = std::hypot(diag, hnext);
            const std::size_t sj = static_cast<std::size_t>(j);
            if (rho == 0.0) {
                stop = true;  // A M^-1 singular on this direction
                break;
            }
            cs[sj] = diag / rho;
            sn[sj] = hnext / rho;
            h(j, j) = rho;
            h(j + 1, j) = 0.0;
            g[sj + 1] = -sn[sj] * g[sj];
            g[sj] = cs[sj] * g[sj];

            // Least-squares solution of the k = j + 1 column problem and the
            // iterate x0 + Z y; checked against the true residual.
            const int k = j + 1;
            for (int i = k - 1; i >= 0; --i) {
                double s = g[static_cast<std::size_t>(i)];
                for (int l = i + 1; l < k; ++l) s -= h(i, l) * y[static_cast<std::size_t>(l)];
                y[static_cast<std::size_t>(i)] = s / h(i, i);
            }
            std::copy(x0.begin(), x0.end(), xc.begin());
            for (int l = 0; l < k; ++l) {
                const double yl = y[static_cast<std::size_t>(l)];
                const double *zl = Z.data() + static_cast<std::size_t>(l) * un;
                for (std::size_t i = 0; i < un; ++i) xc[i] += yl * zl[i];
            }
            double xc_inf = 0.0;
            for (double v : xc) xc_inf = std::max(xc_inf, std::abs(v));
            double cand_eta = inf;
            double cand_omega = inf;
            if (xc_inf <= x_limit &&
                compute_residual_and_backward_error(
                    n, 1, entries, norm_A_inf, b_col, xc, r_cand, cand_eta, cand_omega)) {
                const double cand_score = metric_score(cand_eta, cand_omega);
                if (cand_score < best_score) {
                    best_score = cand_score;
                    eta_out = cand_eta;
                    omega_out = cand_omega;
                    std::copy(xc.begin(), xc.end(), x_col.begin());
                    r_best.swap(r_cand);
                    improved = true;
                    if (meets_gate(eta_out, omega_out)) {
                        done = true;
                        break;
                    }
                }
            }
            if (hnext <= 0.0) break;  // breakdown: the Krylov space is invariant
            for (int i = 0; i < n; ++i) w[i] /= hnext;
        }
        if (stop || !improved) break;
        std::copy(x_col.begin(), x_col.end(), x0.begin());
    }
    return used;
}

// FGMRES on every column of x that does not meet the gate, `budget`
// iterations each. Returns the most iterations any column used. Single-RHS
// solves: each Arnoldi step's preconditioner solve needs the previous one,
// and the columns converge independently, so there is no block of solves to
// batch (classical refinement batches its columns, above).
int fgmres_refine(int n,
                  const std::vector<RefineEntry> &entries,
                  long double norm_A_inf,
                  const FactorKeep &fkeep,
                  const Control &solve_ctrl,
                  std::span<const double> b,
                  std::span<double> x,
                  int nrhs,
                  int budget,
                  SolveWorkspace &ws,
                  double &eta_out,
                  double &omega_out) {
    int most = 0;
    eta_out = 0.0;
    omega_out = 0.0;
    for (int c = 0; c < nrhs; ++c) {
        const std::size_t off = static_cast<std::size_t>(c) * static_cast<std::size_t>(n);
        double eta = 0.0;
        double omega = 0.0;
        const int used = fgmres_column(n, entries, norm_A_inf, fkeep, solve_ctrl,
                                       b.subspan(off, static_cast<std::size_t>(n)),
                                       x.subspan(off, static_cast<std::size_t>(n)), budget,
                                       solve_ctrl.fgmres_restart, ws, eta, omega);
        most = std::max(most, used);
        eta_out = std::max(eta_out, eta);
        omega_out = std::max(omega_out, omega);
    }
    return most;
}

RefinementResult refine_core(int n,
                             const std::vector<RefineEntry> &entries,
                             long double norm_A_inf,
                             const FactorKeep &fkeep,
                             const Control &ctrl,
                             std::span<const double> b,
                             std::span<double> x,
                             int nrhs,
                             SolveWorkspace &ws) {
    const double inf = std::numeric_limits<double>::infinity();
    double best_eta = inf;
    double best_omega = inf;
    if (!compute_residual_and_backward_error(
            n, nrhs, entries, norm_A_inf, b, x, ws.refine_r, best_eta, best_omega)) {
        return RefinementResult{ErrorCode::IllegalValue, 0, best_eta, best_omega};
    }

    if (meets_gate(best_eta, best_omega)) {
        return RefinementResult{ErrorCode::Success, 0, best_eta, best_omega};
    }

    const int max_iters = std::max(0, ctrl.refinement_max_iters);
    if (max_iters == 0) {
        return RefinementResult{ErrorCode::MaxIterations, 0, best_eta, best_omega};
    }

    Control solve_ctrl = ctrl;
    solve_ctrl.refinement_max_iters = 0;

    if (ctrl.refine_method == RefineMethod::Classical) {
        return classical_refine(n, entries, norm_A_inf, fkeep, solve_ctrl, b, x, nrhs, ws,
                                best_eta, best_omega, max_iters, /*escalate=*/false)
            .result;
    }

    const int total = std::max(0, ctrl.refine_max_steps);
    int classical_steps = 0;
    if (ctrl.refine_method == RefineMethod::Auto) {
        const ClassicalOutcome co =
            classical_refine(n, entries, norm_A_inf, fkeep, solve_ctrl, b, x, nrhs, ws,
                             best_eta, best_omega, std::min(max_iters, total), /*escalate=*/true);
        if (!co.escalate || co.result.iterations >= total) return co.result;
        classical_steps = co.result.iterations;
    }

    double eta = inf;
    double omega = inf;
    const int used = fgmres_refine(n, entries, norm_A_inf, fkeep, solve_ctrl, b, x, nrhs,
                                   total - classical_steps, ws, eta, omega);
    const bool qualified = std::isfinite(eta) && std::isfinite(omega) && meets_gate(eta, omega);
    RefinementResult rr{qualified ? ErrorCode::Success : ErrorCode::MaxIterations,
                        classical_steps + used, eta, omega};
    rr.method_used = RefineMethod::Fgmres;
    return rr;
}

} // namespace

} // namespace smf
