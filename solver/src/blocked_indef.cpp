#include "smf/blocked_indef.hpp"
#include "smf/dense_kernel.hpp"
#include "smf/pivoting.hpp"

#include <algorithm>
#include <cmath>
#include <new>
#include <vector>

namespace smf {

FrontResult factor_front_blocked(NumericFront &front,
                                 const Control &ctrl,
                                 NumericWorkspace &ws,
                                 double effective_small,
                                 bool is_root) {
    const int f = front.num_rows();
    const int num_elig = front.num_eligible();
    const int nb = std::max(1, ctrl.indef_panel_size);

    if (ctrl.indef_use_reference_kernel || nb <= 1 || f < 4) {
        return factor_front_reference(front, ctrl, effective_small, is_root);
    }

    FrontResult res;
    try {
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

        const std::size_t w_needed =
            static_cast<std::size_t>(f) * static_cast<std::size_t>(nb + 2);
        if (ws.front_scratch.size() < w_needed) {
            ws.front_scratch.resize(w_needed, 0.0);
        }
        double *W_ptr = ws.front_scratch.data();

        auto abs_elem = [&](int r_idx, int c_idx) noexcept -> double {
            return std::abs(front.at(std::max(r_idx, c_idx), std::min(r_idx, c_idx)));
        };
        auto elem = [&](int r_idx, int c_idx) noexcept -> double {
            return front.at(std::max(r_idx, c_idx), std::min(r_idx, c_idx));
        };

        auto swap_front_and_w = [&](int a, int b, int jb_active) noexcept {
            if (a == b) return;
            front.sym_swap(a, b);
            for (int p = 0; p < jb_active; ++p) {
                std::swap(W_ptr[static_cast<std::size_t>(p) * f + a],
                          W_ptr[static_cast<std::size_t>(p) * f + b]);
            }
        };

        // Apply blocked trailing update front(col_start..f-1, col_start..f-1) -= L21 * W^T.
        // One GEMM over the whole trailing rectangle: it also writes the upper
        // triangle of the diagonal block, which is never read (all accesses in
        // this file, the reference kernel, and front_max_abs use i >= j).
        auto flush_panel = [&](int p_start, int p_end, int col_start_to_update) {
            const int jb = p_end - p_start;
            const int m_rem = f - col_start_to_update;
            if (jb <= 0 || m_rem <= 0) return;
            smf_dgemm_nt(&front.at(col_start_to_update, col_start_to_update),
                         m_rem,
                         m_rem,
                         jb,
                         &front.at(col_start_to_update, p_start),
                         f,
                         W_ptr + col_start_to_update,
                         f,
                         f,
                         -1.0,
                         1.0);
        };

        // Try to select and eliminate one 1x1 or 2x2 pivot within [k, search_limit),
        // updating columns up to `update_col_limit` and storing W at `k - panel_start`.
        auto try_step_within = [&](int &k,
                                   int panel_start,
                                   int search_limit,
                                   int update_col_limit) -> bool {
            const int jb_active = k - panel_start;

            for (int cand = k; cand < search_limit; ++cand) {
                // Find max off-diagonal in [k, num_elig) and in [num_elig, f)
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
                    swap_front_and_w(k, cand, jb_active);
                    const double d00 = front.at(k, k);
                    add_inertia_1x1(d00, eff_small, res.inertia);
                    res.pivot_tags.push_back(int8_t{1});
                    res.d_blocks.push_back(FrontDiagBlock{
                        1, front.row_ids()[static_cast<std::size_t>(k)], 0, d00, 0.0, 0.0});

                    const int p_col = k - panel_start;
                    const double inv_d = 1.0 / d00;
                    double *col_k = front.data() + static_cast<std::size_t>(k) * f;
                    double *w_p = W_ptr + static_cast<std::size_t>(p_col) * f;
                    for (int i = k + 1; i < f; ++i) {
                        const double w_ik = col_k[i];
                        w_p[i] = w_ik;
                        col_k[i] = w_ik * inv_d;
                    }
                    for (int j = k + 1; j < update_col_limit; ++j) {
                        const double w_jk = w_p[j];
                        if (w_jk == 0.0) continue;
                        const double *col_k2 = col_k;
                        double *col_j = front.data() + static_cast<std::size_t>(j) * f;
                        for (int i = j; i < f; ++i) {
                            col_j[i] -= col_k2[i] * w_jk;
                        }
                    }
                    const double rem = static_cast<double>(f - (k + 1));
                    res.flops += rem + rem * (rem + 1.0);
                    ++k;
                    return true;
                }

                // 2. Secondary 1x1 or 2x2 with `r_best`:
                // Only valid inside the panel if `r_best < search_limit` and (for 2x2) `k + 1 < search_limit`!
                if (r_best >= k && r_best < search_limit && omega1_elig > 0.0) {
                    double omega2 = 0.0;
                    for (int i = k; i < f; ++i) {
                        if (i == r_best) continue;
                        const double v = abs_elem(i, r_best);
                        if (v > omega2) omega2 = v;
                    }
                    const double a_r = std::abs(front.at(r_best, r_best));

                    if (a_r >= eff_small &&
                        (omega2 == 0.0 || a_r >= ctrl.pivot_u * omega2)) {
                        swap_front_and_w(k, r_best, jb_active);
                        const double d00 = front.at(k, k);
                        add_inertia_1x1(d00, eff_small, res.inertia);
                        res.pivot_tags.push_back(int8_t{1});
                        res.d_blocks.push_back(FrontDiagBlock{
                            1, front.row_ids()[static_cast<std::size_t>(k)], 0, d00, 0.0, 0.0});

                        const int p_col = k - panel_start;
                        const double inv_d = 1.0 / d00;
                        double *col_k = front.data() + static_cast<std::size_t>(k) * f;
                        double *w_p = W_ptr + static_cast<std::size_t>(p_col) * f;
                        for (int i = k + 1; i < f; ++i) {
                            const double w_ik = col_k[i];
                            w_p[i] = w_ik;
                            col_k[i] = w_ik * inv_d;
                        }
                        for (int j = k + 1; j < update_col_limit; ++j) {
                            const double w_jk = w_p[j];
                            if (w_jk == 0.0) continue;
                            const double *col_k2 = col_k;
                            double *col_j = front.data() + static_cast<std::size_t>(j) * f;
                            for (int i = j; i < f; ++i) {
                                col_j[i] -= col_k2[i] * w_jk;
                            }
                        }
                        const double rem = static_cast<double>(f - (k + 1));
                        res.flops += rem + rem * (rem + 1.0);
                        ++k;
                        return true;
                    }

                    // 2x2 test on (cand, r_best): requires k + 1 < search_limit
                    if (k + 1 < search_limit) {
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
                                    swap_front_and_w(k, cand, jb_active);
                                    if (partner == k) partner = cand;
                                }
                                if (partner != k + 1) {
                                    swap_front_and_w(k + 1, partner, jb_active);
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

                                const int p_col = k - panel_start;
                                double *col_k = front.data() + static_cast<std::size_t>(k) * f;
                                double *col_k1 = front.data() + static_cast<std::size_t>(k + 1) * f;
                                double *w_p0 = W_ptr + static_cast<std::size_t>(p_col) * f;
                                double *w_p1 = W_ptr + static_cast<std::size_t>(p_col + 1) * f;
                                for (int i = k + 2; i < f; ++i) {
                                    const double a0 = col_k[i];
                                    const double a1 = col_k1[i];
                                    w_p0[i] = a0;
                                    w_p1[i] = a1;
                                    double l0 = 0.0;
                                    double l1 = 0.0;
                                    solve_2x2_scaled(d00, d10, d11, a0, a1, l0, l1);
                                    col_k[i] = l0;
                                    col_k1[i] = l1;
                                }

                                for (int j = k + 2; j < update_col_limit; ++j) {
                                    const double wj0 = w_p0[j];
                                    const double wj1 = w_p1[j];
                                    if (wj0 == 0.0 && wj1 == 0.0) continue;
                                    double *col_j = front.data() + static_cast<std::size_t>(j) * f;
                                    for (int i = j; i < f; ++i) {
                                        col_j[i] -= col_k[i] * wj0 + col_k1[i] * wj1;
                                    }
                                }
                                const double rem = static_cast<double>(f - (k + 2));
                                res.flops += 6.0 * rem + 2.0 * rem * (rem + 1.0);
                                k += 2;
                                return true;
                            }
                        }
                    }
                }
            }
            return false;
        };

        int k = 0;
        while (k < num_elig) {
            const int panel_start = k;
            const int panel_limit = std::min(num_elig, panel_start + nb);

            bool panel_stalled = false;
            while (k < panel_limit) {
                if (!try_step_within(k, panel_start, panel_limit, panel_limit)) {
                    panel_stalled = true;
                    break;
                }
            }

            // Flush accumulated panel [panel_start, k) into trailing submatrix [panel_limit, f) via BLAS-3
            if (k > panel_start) {
                flush_panel(panel_start, k, panel_limit);
            }

            if (panel_stalled && k < num_elig) {
                // Now that [k, f) is 100% updated, try a full-range [k, num_elig) pivot step
                // (handles panel-boundary 2x2 pivots and cross-panel swaps!)
                const int step_start = k;
                if (try_step_within(k, step_start, num_elig, step_start)) {
                    flush_panel(step_start, k, k);
                } else {
                    // No candidate in [k, num_elig) can pivot -> remaining eligible columns are delayed
                    break;
                }
            }
        }

        const int p_acc = k;
        const int num_unaccepted = num_elig - p_acc;

        if (is_root && num_unaccepted > 0) {
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

        res.num_accepted = p_acc;
        res.delayed_count = num_unaccepted;
        res.accepted_row_ids.assign(
            front.row_ids().begin(), front.row_ids().begin() + p_acc);
        res.l_row_ids = front.row_ids();
        res.delayed_col_ids.assign(
            front.row_ids().begin() + p_acc, front.row_ids().begin() + num_elig);
        res.contrib_row_ids.assign(
            front.row_ids().begin() + p_acc, front.row_ids().end());

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
    } catch (const std::bad_alloc &) {
        res.status = ErrorCode::AllocationError;
        return res;
    }
}

} // namespace smf
