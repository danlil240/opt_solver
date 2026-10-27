#include "validation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

namespace smf::bench {

namespace {

struct NeumaierSum {
    long double sum = 0.0L;
    long double c = 0.0L;

    void add(long double x) noexcept {
        long double t = sum + x;
        if (std::fabsl(sum) >= std::fabsl(x)) {
            c += (sum - t) + x;
        } else {
            c += (x - t) + sum;
        }
        sum = t;
    }

    long double result() const noexcept { return sum + c; }
};

struct CanonicalEntry {
    Int row = 0;
    Int col = 0;
    long double val = 0.0L;
};

std::string json_escape(const std::string& s) {
    std::ostringstream oss;
    for (unsigned char ch : s) {
        switch (ch) {
            case '"': oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\b': oss << "\\b"; break;
            case '\f': oss << "\\f"; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default:
                if (ch < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(ch));
                    oss << buf;
                } else {
                    oss << static_cast<char>(ch);
                }
        }
    }
    return oss.str();
}

void write_json_double(std::ostringstream& oss, double v) {
    if (!std::isfinite(v)) {
        oss << "null";
    } else {
        oss << std::setprecision(17) << v;
    }
}

void write_validation_json(std::ostringstream& oss, const Validation& v) {
    oss << "{\"finite\":" << (v.finite ? "true" : "false") << ",\"eta_inf\":";
    write_json_double(oss, v.eta_inf);
    oss << ",\"omega\":";
    write_json_double(oss, v.omega);
    oss << ",\"relative_residual\":";
    write_json_double(oss, v.relative_residual);
    oss << "}";
}

} // namespace

const char* outcome_to_string(Outcome outcome) noexcept {
    switch (outcome) {
        case Outcome::Success: return "Success";
        case Outcome::Inaccurate: return "Inaccurate";
        case Outcome::Singular: return "Singular";
        case Outcome::InvalidInput: return "InvalidInput";
        case Outcome::Unavailable: return "Unavailable";
        case Outcome::Timeout: return "Timeout";
        case Outcome::OutOfMemory: return "OutOfMemory";
        case Outcome::Failed:
        default: return "Failed";
    }
}

std::string RunRecord::to_json_line() const {
    std::ostringstream oss;
    oss << "{"
        << "\"schema_version\":" << schema_version << ","
        << "\"matrix_id\":\"" << json_escape(matrix_id) << "\","
        << "\"sequence_id\":\"" << json_escape(sequence_id) << "\","
        << "\"family_id\":\"" << json_escape(family_id) << "\","
        << "\"matrix_sha256\":\"" << json_escape(matrix_sha256) << "\","
        << "\"rhs_sha256\":\"" << json_escape(rhs_sha256) << "\","
        << "\"solver_id\":\"" << json_escape(solver_id) << "\","
        << "\"solver_version\":\"" << json_escape(solver_version) << "\","
        << "\"library_path\":\"" << json_escape(library_path) << "\","
        << "\"blas_backend\":\"" << json_escape(blas_backend) << "\","
        << "\"mode\":\"" << json_escape(mode) << "\","
        << "\"track\":\"" << json_escape(track) << "\","
        << "\"threads\":" << threads << ","
        << "\"blas_threads\":" << blas_threads << ","
        << "\"n\":" << n << ","
        << "\"nnz\":" << nnz << ","
        << "\"nrhs\":" << nrhs << ","
        << "\"trial_id\":" << trial_id << ","
        << "\"warmup\":" << (warmup ? "true" : "false") << ","
        << "\"batch_count\":" << batch_count << ","
        << "\"outcome\":\"" << outcome_to_string(outcome) << "\","
        << "\"reason\":\"" << json_escape(reason) << "\","
        << "\"analyse_ms\":";
    write_json_double(oss, analyse_ms);
    if (has_analyse_phases) {
        const auto ms = [&](const char *key, double sec) {
            oss << ",\"" << key << "\":";
            write_json_double(oss, 1e3 * sec);
        };
        oss << ",\"analyse_phases\":{";
        oss << "\"check\":";
        write_json_double(oss, 1e3 * analyse_phases.check);
        ms("ordering", analyse_phases.ordering);
        ms("matching", analyse_phases.matching);
        ms("etree_counts", analyse_phases.etree_counts);
        ms("amalgamate", analyse_phases.amalgamate);
        ms("assembly_tree", analyse_phases.assembly_tree);
        ms("keep_build", analyse_phases.keep_build);
        oss << "}";
    }
    oss << ",\"factor_ms\":";
    write_json_double(oss, factor_ms);
    if (first_factor_ms >= 0.0) {
        oss << ",\"first_factor_ms\":";
        write_json_double(oss, first_factor_ms);
    }
    if (refactor_ms >= 0.0) {
        oss << ",\"refactor_ms\":";
        write_json_double(oss, refactor_ms);
    }
    oss << ",\"solve_ms\":";
    write_json_double(oss, solve_ms);
    oss << ",\"refine_ms\":";
    write_json_double(oss, refine_ms);
    oss << ",\"wall_ms\":";
    write_json_double(oss, wall_ms);
    oss << ",\"peak_rss_bytes\":" << peak_rss_bytes << ","
        << "\"inertia\":{\"pos\":" << inertia_pos
        << ",\"neg\":" << inertia_neg
        << ",\"zero\":" << inertia_zero << "},"
        << "\"delayed_pivots\":" << delayed_pivots << ","
        << "\"factor_entries\":" << factor_entries << ","
        << "\"flops\":";
    write_json_double(oss, flops);
    oss << ",\"validation\":";
    write_validation_json(oss, validation);
    oss << ",\"per_rhs_validation\":[";
    for (std::size_t i = 0; i < per_rhs_validation.size(); ++i) {
        if (i > 0) oss << ",";
        write_validation_json(oss, per_rhs_validation[i]);
    }
    oss << "],\"per_update_validation\":[";
    for (std::size_t i = 0; i < per_update_validation.size(); ++i) {
        if (i > 0) oss << ",";
        write_validation_json(oss, per_update_validation[i]);
    }
    oss << "]}";
    return oss.str();
}

bool append_run_record_jsonl(const std::string& path, const RunRecord& record) {
    std::ofstream ofs(path, std::ios::out | std::ios::app);
    if (!ofs.is_open()) {
        return false;
    }
    ofs << record.to_json_line() << "\n";
    return static_cast<bool>(ofs);
}

int64_t read_peak_rss_bytes() noexcept {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<int64_t>(pmc.PeakWorkingSetSize);
    }
    return -1;
#else
    std::ifstream ifs("/proc/self/status");
    if (!ifs.is_open()) return -1;
    std::string line;
    int64_t vm_rss_kb = -1;
    while (std::getline(ifs, line)) {
        if (line.rfind("VmHWM:", 0) == 0) {
            const char* p = line.c_str() + 6;
            while (*p == ' ' || *p == '\t') ++p;
            long kb = std::atol(p);
            if (kb >= 0) return static_cast<int64_t>(kb) * 1024;
        } else if (line.rfind("VmRSS:", 0) == 0) {
            const char* p = line.c_str() + 6;
            while (*p == ' ' || *p == '\t') ++p;
            vm_rss_kb = std::atol(p);
        }
    }
    return vm_rss_kb >= 0 ? vm_rss_kb * 1024 : -1;
#endif
}

std::vector<Validation> validate_per_rhs(const CscLower& A,
                                         std::span<const double> b,
                                         std::span<const double> x,
                                         int nrhs) {
    const double inf = std::numeric_limits<double>::infinity();
    if (A.n <= 0 || nrhs <= 0 || A.validate_shape() != ErrorCode::Success) {
        return {Validation{false, inf, inf, inf}};
    }
    const std::size_t n = static_cast<std::size_t>(A.n);
    const std::size_t total_len = n * static_cast<std::size_t>(nrhs);
    if (b.size() < total_len || x.size() < total_len) {
        return {Validation{false, inf, inf, inf}};
    }

    for (double v : A.values) {
        if (!std::isfinite(v)) {
            return std::vector<Validation>(static_cast<std::size_t>(nrhs), Validation{false, inf, inf, inf});
        }
    }

    // Build canonical merged lower-triangular entries so duplicate entries in A are summed before taking |A|.
    std::vector<CanonicalEntry> raw_entries;
    raw_entries.reserve(A.values.size());
    for (Int j = 0; j < A.n; ++j) {
        for (Int p = A.col_ptr[static_cast<std::size_t>(j)];
             p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
            Int i = A.row_idx[static_cast<std::size_t>(p)];
            double v = A.values[static_cast<std::size_t>(p)];
            if (i >= j) {
                raw_entries.push_back({i, j, static_cast<long double>(v)});
            } else {
                raw_entries.push_back({j, i, static_cast<long double>(v)});
            }
        }
    }
    std::sort(raw_entries.begin(), raw_entries.end(), [](const CanonicalEntry& a, const CanonicalEntry& b_e) {
        if (a.col != b_e.col) return a.col < b_e.col;
        return a.row < b_e.row;
    });

    std::vector<CanonicalEntry> entries;
    entries.reserve(raw_entries.size());
    for (const auto& e : raw_entries) {
        if (!entries.empty() && entries.back().row == e.row && entries.back().col == e.col) {
            entries.back().val += e.val;
        } else {
            entries.push_back(e);
        }
    }

    std::vector<NeumaierSum> row_abs_sum(n);
    NeumaierSum fro_sq_sum;
    for (const auto& e : entries) {
        const long double av = std::fabsl(e.val);
        const long double sq = e.val * e.val;
        row_abs_sum[static_cast<std::size_t>(e.row)].add(av);
        if (e.row != e.col) {
            row_abs_sum[static_cast<std::size_t>(e.col)].add(av);
            fro_sq_sum.add(2.0L * sq);
        } else {
            fro_sq_sum.add(sq);
        }
    }

    long double norm_A_inf = 0.0L;
    for (std::size_t i = 0; i < n; ++i) {
        norm_A_inf = std::max(norm_A_inf, row_abs_sum[i].result());
    }
    const long double norm_A_fro = std::sqrt(std::max(0.0L, fro_sq_sum.result()));

    std::vector<Validation> results(static_cast<std::size_t>(nrhs));
    for (int r = 0; r < nrhs; ++r) {
        const double* b_col = b.data() + static_cast<std::size_t>(r) * n;
        const double* x_col = x.data() + static_cast<std::size_t>(r) * n;

        bool col_finite = true;
        long double norm_b_inf = 0.0L;
        long double norm_x_inf = 0.0L;
        NeumaierSum norm_b_sq;
        NeumaierSum norm_x_sq;

        for (std::size_t i = 0; i < n; ++i) {
            if (!std::isfinite(b_col[i]) || !std::isfinite(x_col[i])) {
                col_finite = false;
                break;
            }
            const long double bi = static_cast<long double>(b_col[i]);
            const long double xi = static_cast<long double>(x_col[i]);
            norm_b_inf = std::max(norm_b_inf, std::fabsl(bi));
            norm_x_inf = std::max(norm_x_inf, std::fabsl(xi));
            norm_b_sq.add(bi * bi);
            norm_x_sq.add(xi * xi);
        }

        if (!col_finite) {
            results[static_cast<std::size_t>(r)] = Validation{false, inf, inf, inf};
            continue;
        }

        std::vector<NeumaierSum> ax_sum(n);
        std::vector<NeumaierSum> abs_ax_sum(n);
        for (const auto& e : entries) {
            const std::size_t row = static_cast<std::size_t>(e.row);
            const std::size_t col = static_cast<std::size_t>(e.col);
            const long double v = e.val;
            const long double av = std::fabsl(v);
            const long double x_c = static_cast<long double>(x_col[col]);
            const long double x_r = static_cast<long double>(x_col[row]);
            ax_sum[row].add(v * x_c);
            abs_ax_sum[row].add(av * std::fabsl(x_c));
            if (row != col) {
                ax_sum[col].add(v * x_r);
                abs_ax_sum[col].add(av * std::fabsl(x_r));
            }
        }

        long double norm_res_inf = 0.0L;
        long double max_omega = 0.0L;
        NeumaierSum norm_res_sq;
        bool metrics_finite = true;

        for (std::size_t i = 0; i < n; ++i) {
            const long double bi = static_cast<long double>(b_col[i]);
            const long double res_i = bi - ax_sum[i].result();
            const long double abs_res_i = std::fabsl(res_i);
            norm_res_inf = std::max(norm_res_inf, abs_res_i);
            norm_res_sq.add(res_i * res_i);

            const long double comp_denom = abs_ax_sum[i].result() + std::fabsl(bi);
            long double omega_i = 0.0L;
            if (comp_denom > 0.0L) {
                omega_i = abs_res_i / comp_denom;
            } else if (abs_res_i == 0.0L) {
                omega_i = 0.0L;
            } else {
                omega_i = std::numeric_limits<long double>::infinity();
                metrics_finite = false;
            }
            max_omega = std::max(max_omega, omega_i);
        }

        const long double eta_denom = norm_A_inf * norm_x_inf + norm_b_inf;
        long double eta_inf_val = 0.0L;
        if (eta_denom > 0.0L) {
            eta_inf_val = norm_res_inf / eta_denom;
        } else if (norm_res_inf == 0.0L) {
            eta_inf_val = 0.0L;
        } else {
            eta_inf_val = std::numeric_limits<long double>::infinity();
            metrics_finite = false;
        }

        const long double norm_res_2 = std::sqrt(std::max(0.0L, norm_res_sq.result()));
        const long double norm_x_2 = std::sqrt(std::max(0.0L, norm_x_sq.result()));
        const long double norm_b_2 = std::sqrt(std::max(0.0L, norm_b_sq.result()));
        const long double rel_denom = norm_A_fro * norm_x_2 + norm_b_2;
        long double rel_res_val = 0.0L;
        if (rel_denom > 0.0L) {
            rel_res_val = norm_res_2 / rel_denom;
        } else if (norm_res_2 == 0.0L) {
            rel_res_val = 0.0L;
        } else {
            rel_res_val = std::numeric_limits<long double>::infinity();
            metrics_finite = false;
        }

        Validation v_out;
        v_out.eta_inf = static_cast<double>(eta_inf_val);
        v_out.omega = static_cast<double>(max_omega);
        v_out.relative_residual = static_cast<double>(rel_res_val);
        v_out.finite = metrics_finite &&
                       std::isfinite(v_out.eta_inf) &&
                       std::isfinite(v_out.omega) &&
                       std::isfinite(v_out.relative_residual);
        results[static_cast<std::size_t>(r)] = v_out;
    }

    return results;
}

Validation validate(const CscLower& A,
                    std::span<const double> b,
                    std::span<const double> x,
                    int nrhs) {
    const double inf = std::numeric_limits<double>::infinity();
    auto per_rhs = validate_per_rhs(A, b, x, nrhs);
    if (per_rhs.empty()) {
        return Validation{false, inf, inf, inf};
    }

    Validation agg;
    agg.finite = true;
    agg.eta_inf = 0.0;
    agg.omega = 0.0;
    agg.relative_residual = 0.0;

    for (const auto& vr : per_rhs) {
        if (!vr.finite) {
            agg.finite = false;
        }
        agg.eta_inf = std::max(agg.eta_inf, vr.eta_inf);
        agg.omega = std::max(agg.omega, vr.omega);
        agg.relative_residual = std::max(agg.relative_residual, vr.relative_residual);
    }
    if (!agg.finite) {
        if ( std::isnan(agg.eta_inf) || agg.eta_inf == 0.0 ) agg.eta_inf = inf;
        if ( std::isnan(agg.omega) || agg.omega == 0.0 ) agg.omega = inf;
        if ( std::isnan(agg.relative_residual) || agg.relative_residual == 0.0 ) agg.relative_residual = inf;
    }
    return agg;
}

bool is_qualified_validation(const Validation& v,
                             double max_eta_inf,
                             double max_omega) noexcept {
    return v.finite &&
           std::isfinite(v.eta_inf) &&
           std::isfinite(v.omega) &&
           v.eta_inf <= max_eta_inf &&
           v.omega <= max_omega;
}

Outcome qualify_outcome(Outcome raw_outcome,
                        const Validation& v,
                        double max_eta_inf,
                        double max_omega) noexcept {
    if (raw_outcome != Outcome::Success) {
        return raw_outcome;
    }
    if (!is_qualified_validation(v, max_eta_inf, max_omega)) {
        return Outcome::Inaccurate;
    }
    return Outcome::Success;
}

} // namespace smf::bench
