#include "smf/numeric_front.hpp"
#include "smf/pivoting.hpp"

#include <algorithm>
#include <cmath>
#include <new>

namespace smf {

NumericFront::NumericFront(std::vector<Int> row_ids, int num_eligible)
    : row_ids_(std::move(row_ids)),
      num_eligible_(std::clamp(num_eligible, 0, static_cast<int>(row_ids_.size()))),
      data_(row_ids_.size() * row_ids_.size(), 0.0) {}

void NumericFront::sym_swap(int p, int q) noexcept {
    if (p == q) return;
    if (p > q) std::swap(p, q);
    const int f = num_rows();
    if (p < 0 || q >= f) return;

    std::swap(at(p, p), at(q, q));
    for (int m = 0; m < p; ++m)
        std::swap(at(p, m), at(q, m));
    for (int m = p + 1; m < q; ++m)
        std::swap(at(m, p), at(q, m));
    for (int m = q + 1; m < f; ++m)
        std::swap(at(m, p), at(m, q));

    std::swap(row_ids_[static_cast<std::size_t>(p)],
              row_ids_[static_cast<std::size_t>(q)]);
}

void NumericFront::scatter_original_columns(const CscLower& Ap, Int col_start, Int col_end) {
    const int f = num_rows();
    if (f == 0 || col_start >= col_end) return;

    auto find_local = [&](Int glob) noexcept -> int {
        for (int idx = 0; idx < f; ++idx) {
            if (row_ids_[static_cast<std::size_t>(idx)] == glob) {
                return idx;
            }
        }
        return -1;
    };

    for (Int j = col_start; j < col_end; ++j) {
        const int lj = find_local(j);
        if (lj < 0) continue;
        for (Int p = Ap.col_ptr[static_cast<std::size_t>(j)];
             p < Ap.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
            const Int r_glob = Ap.row_idx[static_cast<std::size_t>(p)];
            const int li = find_local(r_glob);
            if (li < 0) continue;
            const double val = Ap.values[static_cast<std::size_t>(p)];
            at(std::max(li, lj), std::min(li, lj)) += val;
        }
    }
}

void NumericFront::assemble_contribution(const std::vector<Int>& child_rows,
                                         const std::vector<double>& child_contrib) {
    const int qc = static_cast<int>(child_rows.size());
    const int f = num_rows();
    if (qc <= 0 || f <= 0 ||
        child_contrib.size() < static_cast<std::size_t>(qc) * static_cast<std::size_t>(qc)) {
        return;
    }

    std::vector<int> local_map(static_cast<std::size_t>(qc), -1);
    for (int i = 0; i < qc; ++i) {
        const Int target = child_rows[static_cast<std::size_t>(i)];
        for (int idx = 0; idx < f; ++idx) {
            if (row_ids_[static_cast<std::size_t>(idx)] == target) {
                local_map[static_cast<std::size_t>(i)] = idx;
                break;
            }
        }
    }

    for (int cj = 0; cj < qc; ++cj) {
        const int lj = local_map[static_cast<std::size_t>(cj)];
        if (lj < 0) continue;
        for (int ci = cj; ci < qc; ++ci) {
            const int li = local_map[static_cast<std::size_t>(ci)];
            if (li < 0) continue;
            const double val = child_contrib[static_cast<std::size_t>(cj) * static_cast<std::size_t>(qc) +
                                             static_cast<std::size_t>(ci)];
            at(std::max(li, lj), std::min(li, lj)) += val;
        }
    }
}

FrontResult factor_front_reference(NumericFront& front,
                                   const Control& ctrl,
                                   double effective_small,
                                   bool is_root) {
    FrontResult res;
    try {
        const int f = front.num_rows();
        const int num_elig = front.num_eligible();
        res.front_size = f;

        if (f < 0 || num_elig < 0 || num_elig > f) {
            res.status = ErrorCode::IllegalValue;
            return res;
        }
        if (f == 0) {
            res.status = ErrorCode::Success;
            return res;
        }

        double front_max_abs = 0.0;
        for (int j = 0; j < f; ++j) {
            for (int i = j; i < f; ++i) {
                const double v = std::abs(front.at(i, j));
                if (!std::isfinite(v)) {
                    res.status = ErrorCode::IllegalValue;
                    return res;
                }
                if (v > front_max_abs) {
                    front_max_abs = v;
                }
            }
        }

        const double eff_small =
            (effective_small > 0.0)
                ? effective_small
                : ((front_max_abs > 0.0 && front_max_abs < ctrl.small_pivot)
                       ? (ctrl.small_pivot * front_max_abs)
                       : ctrl.small_pivot);

        auto abs_elem = [&](int r_idx, int c_idx) noexcept -> double {
            return std::abs(front.at(std::max(r_idx, c_idx), std::min(r_idx, c_idx)));
        };

        auto elem = [&](int r_idx, int c_idx) noexcept -> double {
            return front.at(std::max(r_idx, c_idx), std::min(r_idx, c_idx));
        };

        int k = 0;
        while (k < num_elig) {
            bool accepted_step = false;

            for (int cand = k; cand < num_elig; ++cand) {
                // Evaluate candidate column `cand` against active rows [k, f)
                double omega1_elig = 0.0;
                int r_best = -1;
                for (int i = k; i < num_elig; ++i) {
                    if (i == cand) continue;
                    const double v = abs_elem(i, cand);
                    if (v > omega1_elig) {
                        omega1_elig = v;
                        r_best = i;
                    }
                }

                double omega1_ext = 0.0;
                for (int i = num_elig; i < f; ++i) {
                    const double v = abs_elem(i, cand);
                    if (v > omega1_ext) {
                        omega1_ext = v;
                    }
                }

                const double omega1_full = std::max(omega1_elig, omega1_ext);
                const double a_cand = std::abs(front.at(cand, cand));

                // 1. Primary 1x1 test on `cand`
                if (a_cand >= eff_small &&
                    (omega1_full == 0.0 || a_cand >= ctrl.pivot_u * omega1_full)) {
                    if (cand != k) {
                        front.sym_swap(k, cand);
                    }
                    const double d00 = front.at(k, k);
                    add_inertia_1x1(d00, eff_small, res.inertia);
                    res.pivot_tags.push_back(int8_t{1});
                    res.d_blocks.push_back(FrontDiagBlock{
                        1, front.row_ids()[static_cast<std::size_t>(k)], 0, d00, 0.0, 0.0});

                    const double inv_d = 1.0 / d00;
                    double *col_k = front.data() + static_cast<std::size_t>(k) * f;
                    for (int i = k + 1; i < f; ++i) {
                        col_k[i] *= inv_d;
                    }
                    for (int j = k + 1; j < f; ++j) {
                        const double w_jk = d00 * col_k[j];
                        if (w_jk == 0.0) continue;
                        double *col_j = front.data() + static_cast<std::size_t>(j) * f;
                        for (int i = j; i < f; ++i) {
                            col_j[i] -= col_k[i] * w_jk;
                        }
                    }
                    const double rem = static_cast<double>(f - (k + 1));
                    res.flops += rem + rem * (rem + 1.0);
                    ++k;
                    accepted_step = true;
                    break;
                }

                // 2. If there is a coupling partner `r_best` in [k, num_elig), check secondary 1x1 or 2x2
                if (r_best >= k && omega1_elig > 0.0) {
                    double omega2 = 0.0;
                    for (int i = k; i < f; ++i) {
                        if (i == r_best) continue;
                        const double v = abs_elem(i, r_best);
                        if (v > omega2) omega2 = v;
                    }
                    const double a_r = std::abs(front.at(r_best, r_best));

                    // 2x2 test on (cand, r_best)
                    const double d00_cand = front.at(cand, cand);
                    const double d10_cand = elem(r_best, cand);
                    const double d11_cand = front.at(r_best, r_best);
                    double s2 = 0.0;
                    const double det_s = scaled_det_2x2(d00_cand, d10_cand, d11_cand, s2);

                    if (s2 >= eff_small && std::abs(det_s) > ctrl.small_pivot) {
                        double off_0 = 0.0;
                        double off_1 = 0.0;
                        for (int i = k; i < f; ++i) {
                            if (i == cand || i == r_best) continue;
                            const double v0 = abs_elem(i, cand);
                            const double v1 = abs_elem(i, r_best);
                            if (v0 > off_0) off_0 = v0;
                            if (v1 > off_1) off_1 = v1;
                        }

                        bool ok_2x2 = true;
                        if (off_0 > 0.0 || off_1 > 0.0) {
                            const double a = std::abs(d00_cand) / s2;
                            const double b = std::abs(d10_cand) / s2;
                            const double c = std::abs(d11_cand) / s2;
                            const double bound0 = (c * off_0 + b * off_1) / s2;
                            const double bound1 = (b * off_0 + a * off_1) / s2;
                            if (ctrl.pivot_u * std::max(bound0, bound1) > std::abs(det_s)) {
                                ok_2x2 = false;
                            }
                        }

                        if (ok_2x2) {
                            int partner = r_best;
                            if (cand != k) {
                                front.sym_swap(k, cand);
                                if (partner == k) partner = cand;
                            }
                            if (partner != k + 1) {
                                front.sym_swap(k + 1, partner);
                            }

                            const double d00 = front.at(k, k);
                            const double d10 = front.at(k + 1, k);
                            const double d11 = front.at(k + 1, k + 1);
                            add_inertia_2x2(d00, d10, d11, ctrl.small_pivot, res.inertia);
                            res.pivot_tags.push_back(int8_t{2});
                            res.pivot_tags.push_back(int8_t{-1});
                            res.d_blocks.push_back(FrontDiagBlock{
                                2,
                                front.row_ids()[static_cast<std::size_t>(k)],
                                front.row_ids()[static_cast<std::size_t>(k + 1)],
                                d00,
                                d10,
                                d11});

                            double *col_k = front.data() + static_cast<std::size_t>(k) * f;
                            double *col_k1 = front.data() + static_cast<std::size_t>(k + 1) * f;
                            for (int i = k + 2; i < f; ++i) {
                                const double a0 = col_k[i];
                                const double a1 = col_k1[i];
                                double l0 = 0.0;
                                double l1 = 0.0;
                                solve_2x2_scaled(d00, d10, d11, a0, a1, l0, l1);
                                col_k[i] = l0;
                                col_k1[i] = l1;
                            }

                            for (int j = k + 2; j < f; ++j) {
                                const double lj0 = col_k[j];
                                const double lj1 = col_k1[j];
                                const double wj0 = d00 * lj0 + d10 * lj1;
                                const double wj1 = d10 * lj0 + d11 * lj1;
                                if (wj0 == 0.0 && wj1 == 0.0) continue;
                                double *col_j = front.data() + static_cast<std::size_t>(j) * f;
                                for (int i = j; i < f; ++i) {
                                    col_j[i] -= col_k[i] * wj0 + col_k1[i] * wj1;
                                }
                            }
                            const double rem = static_cast<double>(f - (k + 2));
                            res.flops += 6.0 * rem + 2.0 * rem * (rem + 1.0);
                            k += 2;
                            accepted_step = true;
                            break;
                        }
                    }

                    // Secondary 1x1 on `r_best`, only once the 2x2 is rejected
                    // (MA27's order): at a tiny u a near-null partner diagonal
                    // passes this test and puts up to 1/u into L.
                    if (a_r >= eff_small &&
                        (omega2 == 0.0 || a_r >= ctrl.pivot_u * omega2)) {
                        if (r_best != k) {
                            front.sym_swap(k, r_best);
                        }
                        const double d00 = front.at(k, k);
                        add_inertia_1x1(d00, eff_small, res.inertia);
                        res.pivot_tags.push_back(int8_t{1});
                        res.d_blocks.push_back(FrontDiagBlock{
                            1, front.row_ids()[static_cast<std::size_t>(k)], 0, d00, 0.0, 0.0});

                        const double inv_d = 1.0 / d00;
                        double *col_k = front.data() + static_cast<std::size_t>(k) * f;
                        for (int i = k + 1; i < f; ++i) {
                            col_k[i] *= inv_d;
                        }
                        for (int j = k + 1; j < f; ++j) {
                            const double w_jk = d00 * col_k[j];
                            if (w_jk == 0.0) continue;
                            double *col_j = front.data() + static_cast<std::size_t>(j) * f;
                            for (int i = j; i < f; ++i) {
                                col_j[i] -= col_k[i] * w_jk;
                            }
                        }
                        const double rem = static_cast<double>(f - (k + 1));
                        res.flops += rem + rem * (rem + 1.0);
                        ++k;
                        accepted_step = true;
                        break;
                    }
                }
            }

            if (!accepted_step) {
                break;
            }
        }

        const int p_acc = k;
        const int num_unaccepted = num_elig - p_acc;

        if (is_root && num_unaccepted > 0) {
            // Root front has no parent to delay to: unaccepted columns are zero pivots (nullity)
            res.inertia.zero += num_unaccepted;
            res.delayed_count = 0;
            for (int c = p_acc; c < num_elig; ++c) {
                res.pivot_tags.push_back(int8_t{0});
                res.d_blocks.push_back(FrontDiagBlock{
                    1, front.row_ids()[static_cast<std::size_t>(c)], 0, 0.0, 0.0, 0.0});
            }
            const int p_root = num_elig;
            res.num_accepted = p_root;
            res.accepted_row_ids.assign(
                front.row_ids().begin(), front.row_ids().begin() + p_root);
            res.l_row_ids = front.row_ids();
            res.l_panel.assign(
                static_cast<std::size_t>(f) * static_cast<std::size_t>(p_root), 0.0);

            for (int col = 0; col < p_root; ++col) {
                const int8_t tag = res.pivot_tags[static_cast<std::size_t>(col)];
                if (tag == 0) {
                    res.l_panel[static_cast<std::size_t>(col) * f + col] = 1.0;
                } else {
                    res.l_panel[static_cast<std::size_t>(col) * f + col] = 1.0;
                    const int start_row = (tag == 2) ? (col + 2) : (col + 1);
                    for (int r = start_row; r < f; ++r) {
                        res.l_panel[static_cast<std::size_t>(col) * f + r] = front.at(r, col);
                    }
                }
            }
            res.status = ErrorCode::SingularMatrix;
            return res;
        }

        // Non-root front (or root with 0 unaccepted columns)
        res.num_accepted = p_acc;
        res.delayed_count = num_unaccepted;
        res.accepted_row_ids.assign(
            front.row_ids().begin(), front.row_ids().begin() + p_acc);
        res.l_row_ids = front.row_ids();
        res.delayed_col_ids.assign(
            front.row_ids().begin() + p_acc, front.row_ids().begin() + num_elig);
        res.contrib_row_ids.assign(
            front.row_ids().begin() + p_acc, front.row_ids().end());

        // Pack f x p_acc unit-block lower-triangular L panel
        res.l_panel.assign(
            static_cast<std::size_t>(f) * static_cast<std::size_t>(p_acc), 0.0);
        for (int col = 0; col < p_acc; ++col) {
            const int8_t tag = res.pivot_tags[static_cast<std::size_t>(col)];
            res.l_panel[static_cast<std::size_t>(col) * f + col] = 1.0;
            const int start_row = (tag == 2) ? (col + 2) : (col + 1);
            for (int r = start_row; r < f; ++r) {
                res.l_panel[static_cast<std::size_t>(col) * f + r] = front.at(r, col);
            }
        }

        // Pack (f - p_acc) x (f - p_acc) lower-triangular contribution block (delayed + extension)
        const int q_rem = f - p_acc;
        if (q_rem > 0 && !is_root) {
            res.contrib.assign(
                static_cast<std::size_t>(q_rem) * static_cast<std::size_t>(q_rem), 0.0);
            for (int cj = 0; cj < q_rem; ++cj) {
                for (int ci = cj; ci < q_rem; ++ci) {
                    res.contrib[static_cast<std::size_t>(cj) * static_cast<std::size_t>(q_rem) +
                                static_cast<std::size_t>(ci)] =
                        front.at(p_acc + ci, p_acc + cj);
                }
            }
        }

        res.status = ErrorCode::Success;
        return res;
    } catch (const std::bad_alloc&) {
        res.status = ErrorCode::AllocationError;
        return res;
    }
}

} // namespace smf
