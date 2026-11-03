#include "adapters/adapter.hpp"
#include "common/replay_refine.hpp"
#include "common/replay_steps.hpp"
#include "common/result.hpp"
#include "common/validation.hpp"
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/ipopt_plugin_control.hpp"
#include "smf/matrix_market.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <fstream>
#include <string>
#include <vector>

#if defined(SMF_USE_MKL) && SMF_USE_MKL
#include <mkl.h>
#endif
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

// RunRecord::order_hash: AdapterStats::order_hash as 16 hex digits.
std::string order_hash_hex(std::uint64_t h) {
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(h));
    return buf;
}

// Binary cache next to a .mtx (<path>.smfbin): large Matrix Market files
// take seconds to parse, and the harness reloads the matrix for every
// attempt. Parsing is outside the timed region either way.
constexpr char kBinMagic[8] = {'S', 'M', 'F', 'B', 'I', 'N', '0', '1'};

bool load_bin_cache(const std::string &mtx_path, smf::CscLower &A) {
    namespace fs = std::filesystem;
    const std::string bin = mtx_path + ".smfbin";
    std::error_code ec;
    if (!fs::exists(bin, ec) || fs::last_write_time(bin, ec) < fs::last_write_time(mtx_path, ec)) {
        return false;
    }
    std::ifstream in(bin, std::ios::binary);
    char magic[8];
    std::int64_t n = 0, nnz = 0;
    in.read(magic, 8);
    in.read(reinterpret_cast<char *>(&n), sizeof n);
    in.read(reinterpret_cast<char *>(&nnz), sizeof nnz);
    if (!in || std::memcmp(magic, kBinMagic, 8) != 0 || n < 0 || nnz < 0) return false;
    A.n = static_cast<smf::Int>(n);
    A.col_ptr.resize(static_cast<std::size_t>(n) + 1);
    A.row_idx.resize(static_cast<std::size_t>(nnz));
    A.values.resize(static_cast<std::size_t>(nnz));
    in.read(reinterpret_cast<char *>(A.col_ptr.data()), static_cast<std::streamsize>(A.col_ptr.size() * sizeof(smf::Int)));
    in.read(reinterpret_cast<char *>(A.row_idx.data()), static_cast<std::streamsize>(A.row_idx.size() * sizeof(smf::Int)));
    in.read(reinterpret_cast<char *>(A.values.data()), static_cast<std::streamsize>(A.values.size() * sizeof(double)));
    return static_cast<bool>(in);
}

void save_bin_cache(const std::string &mtx_path, const smf::CscLower &A) {
    const std::string tmp = mtx_path + ".smfbin.tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        const std::int64_t n = A.n;
        const std::int64_t nnz = static_cast<std::int64_t>(A.row_idx.size());
        out.write(kBinMagic, 8);
        out.write(reinterpret_cast<const char *>(&n), sizeof n);
        out.write(reinterpret_cast<const char *>(&nnz), sizeof nnz);
        out.write(reinterpret_cast<const char *>(A.col_ptr.data()), static_cast<std::streamsize>(A.col_ptr.size() * sizeof(smf::Int)));
        out.write(reinterpret_cast<const char *>(A.row_idx.data()), static_cast<std::streamsize>(A.row_idx.size() * sizeof(smf::Int)));
        out.write(reinterpret_cast<const char *>(A.values.data()), static_cast<std::streamsize>(A.values.size() * sizeof(double)));
        if (!out) return;
    }
    std::error_code ec;
    std::filesystem::rename(tmp, mtx_path + ".smfbin", ec);
}

smf::CscLower make_tridiag_spd(int n) {
    smf::CscLower A;
    A.n = n;
    A.col_ptr.resize(static_cast<std::size_t>(n) + 1, 0);
    A.row_idx.reserve(static_cast<std::size_t>(2 * n));
    A.values.reserve(static_cast<std::size_t>(2 * n));
    for (int j = 0; j < n; ++j) {
        A.row_idx.push_back(j);
        A.values.push_back(4.0);
        if (j + 1 < n) {
            A.row_idx.push_back(j + 1);
            A.values.push_back(-1.0);
        }
        A.col_ptr[static_cast<std::size_t>(j + 1)] =
            static_cast<smf::Int>(A.row_idx.size());
    }
    return A;
}

smf::CscLower make_banded_spd(int n, int bw) {
    smf::CscLower A;
    A.n = n;
    A.col_ptr.resize(static_cast<std::size_t>(n) + 1, 0);
    for (int j = 0; j < n; ++j) {
        A.row_idx.push_back(j);
        A.values.push_back(static_cast<double>(2 * bw + 4));
        for (int d = 1; d <= bw && j + d < n; ++d) {
            A.row_idx.push_back(j + d);
            A.values.push_back(-1.0 / static_cast<double>(d));
        }
        A.col_ptr[static_cast<std::size_t>(j + 1)] =
            static_cast<smf::Int>(A.row_idx.size());
    }
    return A;
}

smf::CscLower make_poisson2d_spd(int grid) {
    const int n = grid * grid;
    smf::CscLower A;
    A.n = n;
    A.col_ptr.resize(static_cast<std::size_t>(n) + 1, 0);
    A.row_idx.reserve(static_cast<std::size_t>(3 * n));
    A.values.reserve(static_cast<std::size_t>(3 * n));
    for (int r = 0; r < grid; ++r) {
        for (int c = 0; c < grid; ++c) {
            const int j = r * grid + c;
            A.row_idx.push_back(j);
            A.values.push_back(4.0 + 1e-2);
            if (c + 1 < grid) {
                A.row_idx.push_back(j + 1);
                A.values.push_back(-1.0);
            }
            if (r + 1 < grid) {
                A.row_idx.push_back(j + grid);
                A.values.push_back(-1.0);
            }
            A.col_ptr[static_cast<std::size_t>(j + 1)] =
                static_cast<smf::Int>(A.row_idx.size());
        }
    }
    return A;
}

// Build a stage-coupled KKT saddle-point matrix for optimal control / MPC:
// Each stage t in [0, horizon) has primal state/control block of size `n_prim`
// and equality constraint block of size `n_dual` coupling stage t to t+1.
// Lower-triangular CSC with positive primal curvature, off-diagonal Jacobian
// coupling, and negative dual regularization (-1e-3).
smf::CscLower make_kkt_saddle_stage(int horizon, int n_prim, int n_dual, int variant) {
    const int stage_sz = n_prim + n_dual;
    const int n = horizon * stage_sz;
    smf::CscLower A;
    A.n = n;
    A.col_ptr.resize(static_cast<std::size_t>(n) + 1, 0);
    const double base_h = 4.0 + 0.25 * static_cast<double>(variant % 5);
    const double reg_d = -1e-2 - 2e-3 * static_cast<double>(variant % 4);

    for (int t = 0; t < horizon; ++t) {
        const int base = t * stage_sz;
        // Primal columns within stage t
        for (int p = 0; p < n_prim; ++p) {
            const int col = base + p;
            A.row_idx.push_back(col);
            A.values.push_back(base_h + 0.1 * static_cast<double>((p + t) % 4));
            if (p + 1 < n_prim) {
                A.row_idx.push_back(col + 1);
                A.values.push_back(-0.35);
            }
            // Jacobian coupling J_t: connect primal var p to dual constraint rows in stage t
            const int d_row0 = base + n_prim + (p % n_dual);
            A.row_idx.push_back(d_row0);
            A.values.push_back(1.0 + 0.05 * static_cast<double>(p));
            if (n_dual > 1) {
                const int d_row1 = base + n_prim + ((p + 1) % n_dual);
                if (d_row1 > d_row0) {
                    A.row_idx.push_back(d_row1);
                    A.values.push_back(-0.4);
                }
            }
            A.col_ptr[static_cast<std::size_t>(col + 1)] =
                static_cast<smf::Int>(A.row_idx.size());
        }
        // Dual columns within stage t (couple to next stage's primal variables too)
        for (int d = 0; d < n_dual; ++d) {
            const int col = base + n_prim + d;
            A.row_idx.push_back(col);
            A.values.push_back(reg_d);
            if (t + 1 < horizon) {
                const int next_prim = (t + 1) * stage_sz + (d % n_prim);
                A.row_idx.push_back(next_prim);
                A.values.push_back(-1.0);
            }
            A.col_ptr[static_cast<std::size_t>(col + 1)] =
                static_cast<smf::Int>(A.row_idx.size());
        }
    }
    return A;
}

smf::CscLower make_indef_banded(int n, int bw) {
    smf::CscLower A;
    A.n = n;
    A.col_ptr.resize(static_cast<std::size_t>(n) + 1, 0);
    for (int j = 0; j < n; ++j) {
        A.row_idx.push_back(j);
        const double sign = (j % 3 == 2) ? -1.0 : 1.0;
        A.values.push_back(sign * static_cast<double>(2 * bw + 5));
        for (int d = 1; d <= bw && j + d < n; ++d) {
            A.row_idx.push_back(j + d);
            A.values.push_back((d % 2 == 0 ? 0.5 : -0.8) / static_cast<double>(d));
        }
        A.col_ptr[static_cast<std::size_t>(j + 1)] =
            static_cast<smf::Int>(A.row_idx.size());
    }
    return A;
}

smf::CscLower make_builtin_fixture(const std::string &case_id, bool &is_spd) {
    is_spd = false;
    if (case_id == "smoke_spd_tridiag_10") {
        is_spd = true;
        return make_tridiag_spd(10);
    }
    if (case_id == "tridiag_500") {
        is_spd = true;
        return make_tridiag_spd(500);
    }
    if (case_id == "tridiag_2000") {
        is_spd = true;
        return make_tridiag_spd(2000);
    }
    if (case_id == "tridiag_5000") {
        is_spd = true;
        return make_tridiag_spd(5000);
    }
    if (case_id == "banded_spd_500_8") {
        is_spd = true;
        return make_banded_spd(500, 8);
    }
    if (case_id == "banded_spd_2000_10") {
        is_spd = true;
        return make_banded_spd(2000, 10);
    }
    if (case_id == "poisson2d_20") {
        is_spd = true;
        return make_poisson2d_spd(20);
    }
    if (case_id == "poisson2d_31") {
        is_spd = true;
        return make_poisson2d_spd(31);
    }
    if (case_id == "poisson2d_50") {
        is_spd = true;
        return make_poisson2d_spd(50);
    }
    if (case_id == "smoke_indef_exchange_2") {
        smf::CscLower A;
        A.n = 2;
        A.col_ptr = {0, 2, 3};
        A.row_idx = {0, 1, 1};
        A.values = {0.0, 1.0, 0.0};
        return A;
    }
    if (case_id == "indef_banded_600_6") {
        return make_indef_banded(600, 6);
    }
    if (case_id == "indef_banded_1500_8") {
        return make_indef_banded(1500, 8);
    }
    if (case_id == "kkt_ocp_lqr") {
        return make_kkt_saddle_stage(40, 6, 3, 0);
    }
    if (case_id == "kkt_vtol_intercept") {
        return make_kkt_saddle_stage(35, 8, 4, 1);
    }
    if (case_id == "kkt_robot_arm_mpc") {
        return make_kkt_saddle_stage(30, 8, 4, 2);
    }
    if (case_id == "kkt_ac_opf") {
        return make_kkt_saddle_stage(45, 6, 3, 3);
    }
    if (case_id == "kkt_contact_qp") {
        return make_kkt_saddle_stage(35, 7, 3, 4);
    }
    if (case_id == "kkt_chem_process") {
        return make_kkt_saddle_stage(40, 5, 3, 5);
    }
    if (case_id == "kkt_autonomous_driving_nmpc") {
        return make_kkt_saddle_stage(36, 6, 3, 6);
    }
    if (case_id == "kkt_quadrotor_swing") {
        return make_kkt_saddle_stage(32, 8, 4, 7);
    }
    if (case_id == "kkt_fixed_wing_guidance") {
        return make_kkt_saddle_stage(40, 6, 3, 8);
    }
    if (case_id == "kkt_spacecraft_rendezvous") {
        return make_kkt_saddle_stage(38, 7, 3, 9);
    }
    // Default synthetic KKT saddle-point fixture (4x4 indefinite)
    smf::CscLower A;
    A.n = 4;
    A.col_ptr = {0, 3, 5, 7, 8};
    A.row_idx = {0, 2, 3, 1, 2, 2, 3, 3};
    A.values = {4.0, 1.0, -1.0, 3.0, 2.0, -1e-2, 0.5, -1e-2};
    return A;
}

// Matrix Market (lower triangle, symmetric) writer for --export-mtx / --export-step.
bool write_matrix_market(const std::string &path, const smf::CscLower &A) {
    std::FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::fprintf(f, "%%%%MatrixMarket matrix coordinate real symmetric\n%d %d %lld\n", static_cast<int>(A.n),
                 static_cast<int>(A.n), static_cast<long long>(A.nnz()));
    for (smf::Int j = 0; j < A.n; ++j) {
        for (smf::Int p = A.col_ptr[static_cast<std::size_t>(j)]; p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
            std::fprintf(f, "%d %d %.17g\n", static_cast<int>(A.row_idx[static_cast<std::size_t>(p)]) + 1,
                         static_cast<int>(j) + 1, A.values[static_cast<std::size_t>(p)]);
        }
    }
    return std::fclose(f) == 0;
}

// The values of the last factorization a warm_sequence replay runs (the matrix
// its record's inertia belongs to): the last of the --seq-steps picks of a
// recorded sequence, else the last of the 5 synthetic steps (nonzero leading
// diagonal entries x 1.2), mirroring the replay loop in main().
bool last_replayed_values(const smf::CscLower &A, const std::string &seq_values_path, int seq_steps,
                          std::vector<double> &out) {
    out = A.values;
    if (!seq_values_path.empty()) {
        const std::size_t step_bytes = A.values.size() * sizeof(double);
        std::ifstream in(seq_values_path, std::ios::binary | std::ios::ate);
        const std::size_t bytes = in ? static_cast<std::size_t>(in.tellg()) : 0;
        const std::size_t nsteps = step_bytes ? bytes / step_bytes : 0;
        if (nsteps == 0) return false;
        const auto picks = smf::bench::replay_step_picks(nsteps, seq_steps);
        in.seekg(static_cast<std::streamoff>(picks.back() * step_bytes));
        in.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(step_bytes));
        return static_cast<bool>(in);
    }
    const double delta = 1.0 + 0.05 * 4.0;
    for (smf::Int j = 0; j < A.n; ++j) {
        const smf::Int p0 = A.col_ptr[static_cast<std::size_t>(j)];
        if (p0 < A.col_ptr[static_cast<std::size_t>(j) + 1] && A.row_idx[static_cast<std::size_t>(p0)] == j) {
            const double v0 = A.values[static_cast<std::size_t>(p0)];
            if (v0 != 0.0) out[static_cast<std::size_t>(p0)] = v0 * delta;
        }
    }
    return true;
}

} // namespace

int main(int argc, char **argv) {
    std::string manifest_path;
    std::string case_id = "smoke_kkt_4";
    std::string sequence_id;
    std::string family_id;
    std::string solver_id = "smf";
    std::string mode = "cold";
    std::string track = "recommended";
    std::string jsonl_out;
    std::string mtx_path;
    std::string mtype;  // "spd" | "indefinite" override for .mtx cases
    std::string seq_values_path; // recorded value sequence (raw doubles, nnz per step)
    int trial_id = 0;
    int nrhs = 1;
    int threads = 1;
    bool is_warmup = false;
    // Control overrides for bisecting an accuracy failure (applied after the
    // track defaults): --pivot-u X, --scaling none|equil|matching|auto,
    // --ordering auto|auto_parallel|amd|metis|matching, --refine-steps N,
    // --matching-trigger predicted|realized
    // (harness refinement).
    double pivot_u_override = -1.0;
    std::string scaling_override;
    std::string ordering_override;
    std::string matching_trigger;  // predicted | realized (smf auto orderings)
    int refine_steps = 3;
    // --ipm-static off|fallback|regularized [--ipm-static-zero]: the smf
    // solver runs through smf::IpmSession with that static mode, as the MA97
    // plugin does inside IPOPT (make_smf_ipm_adapter); per-step lines go to
    // stderr. --seq-steps N|all: warm_sequence replays N evenly spaced
    // recorded factorizations (default 5) or every one. --print-steps: the
    // per-step stderr lines (inertia, delays) for any solver, e.g. MA27's
    // reference inertia of a recorded sequence.
    std::string ipm_static;
    bool ipm_static_zero = false;
    bool print_steps = false;
    int seq_steps = 5;
    // --export-mtx PATH: write the case's matrix (a built-in fixture or --mtx)
    // as Matrix Market, print "mtype=spd|indefinite", and exit. run_wsl.py uses
    // it to hand the built-in cases to the Linux SSIDS driver.
    std::string export_mtx;
    // --export-step PATH: write the matrix whose inertia the record reports
    // (cold: the case's matrix; warm_sequence: its last replayed step, see
    // last_replayed_values) as Matrix Market and exit. tools/bench/inertia_ref.py
    // builds its reference inertia from it.
    std::string export_step;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--manifest" && i + 1 < argc) manifest_path = argv[++i];
        else if (arg == "--case" && i + 1 < argc) case_id = argv[++i];
        else if (arg == "--sequence-id" && i + 1 < argc) sequence_id = argv[++i];
        else if (arg == "--family-id" && i + 1 < argc) family_id = argv[++i];
        else if (arg == "--solver" && i + 1 < argc) solver_id = argv[++i];
        else if (arg == "--mode" && i + 1 < argc) mode = argv[++i];
        else if (arg == "--track" && i + 1 < argc) track = argv[++i];
        else if (arg == "--trial" && i + 1 < argc) trial_id = std::atoi(argv[++i]);
        else if (arg == "--nrhs" && i + 1 < argc) nrhs = std::max(1, std::atoi(argv[++i]));
        else if (arg == "--threads" && i + 1 < argc) threads = std::max(1, std::atoi(argv[++i]));
        else if (arg == "--mtx" && i + 1 < argc) mtx_path = argv[++i];
        else if (arg == "--mtype" && i + 1 < argc) mtype = argv[++i];
        else if (arg == "--seq-values" && i + 1 < argc) seq_values_path = argv[++i];
        else if (arg == "--JsonlOut" && i + 1 < argc) jsonl_out = argv[++i];
        else if (arg == "--pivot-u" && i + 1 < argc) pivot_u_override = std::atof(argv[++i]);
        else if (arg == "--scaling" && i + 1 < argc) scaling_override = argv[++i];
        else if (arg == "--ordering" && i + 1 < argc) ordering_override = argv[++i];
        else if (arg == "--matching-trigger" && i + 1 < argc) matching_trigger = argv[++i];
        else if (arg == "--refine-steps" && i + 1 < argc) refine_steps = std::max(0, std::atoi(argv[++i]));
        else if (arg == "--ipm-static" && i + 1 < argc) ipm_static = argv[++i];
        else if (arg == "--ipm-static-zero") ipm_static_zero = true;
        else if (arg == "--print-steps") print_steps = true;
        else if (arg == "--export-mtx" && i + 1 < argc) export_mtx = argv[++i];
        else if (arg == "--export-step" && i + 1 < argc) export_step = argv[++i];
        else if (arg == "--seq-steps" && i + 1 < argc) {
            const std::string v = argv[++i];
            seq_steps = (v == "all") ? -1 : std::max(1, std::atoi(v.c_str()));
        }
        else if (arg == "--warmup") is_warmup = true;
    }

    // "ipopt-plugin" track: the thread count is the plugin's own,
    // smf::plugin_num_threads() (SMF_NUM_THREADS, else OMP_NUM_THREADS, else
    // the performance cores; ipopt_plugin_control.hpp), as inside IPOPT. run.py
    // exports OMP_NUM_THREADS = --threads, so the two agree unless
    // SMF_NUM_THREADS is set (or bench_replay runs without run.py): the run
    // then uses the plugin's count, and its record says so.
    const bool plugin_track = track == "ipopt-plugin";
    if (plugin_track) {
        const smf::PluginThreads pt = smf::plugin_threads();
        if (pt.threads != threads) {
            std::fprintf(stderr, "warning: ipopt-plugin track: the plugin runs %d thread(s) (%s), not --threads %d; "
                                 "this run uses %d\n",
                         pt.threads, smf::plugin_threads_source_name(pt.source), threads, pt.threads);
        }
        threads = pt.threads;
    }

    // One thread budget for every library in the process: smf's tree
    // parallelism (Control::num_threads), MKL (BLAS / PARDISO) and OpenMP.
    // run.py also exports OMP/MKL/OPENBLAS_NUM_THREADS so third-party
    // OpenMP runtimes we cannot reach from here obey the same cap.
#if defined(SMF_USE_MKL) && SMF_USE_MKL
    mkl_set_num_threads(threads);
#endif
#if defined(_OPENMP)
    // Only smf runs on this binary's own OpenMP runtime (libomp140, /openmp:llvm).
    // Touching it for another solver would initialise it next to that solver's
    // runtime (conda MUMPS loads libomp.dll) and abort with "OMP: Error #15".
    // Those solvers read OMP_NUM_THREADS, which run.py exports.
    if (solver_id == "smf") omp_set_num_threads(threads);
#endif

    if (sequence_id.empty()) sequence_id = case_id;
    if (family_id.empty()) family_id = sequence_id;

    // "ipopt-plugin" track: smf exactly as the MA97 plugin (libsmf_ma97) runs
    // it inside IPOPT under IPOPT's default ma97 options: the plugin's
    // Control (smf::ipopt_plugin_control(), the function the plugin maps every
    // MA97 control block through) and its IpmSession with the plugin's
    // IpmControl (smf::ipopt_plugin_ipm_control(), SMF_IPM_* environment);
    // the analysis is deferred to the first factorization, as IPOPT analyses
    // the pattern without values. --ipm-static, when given, overrides the
    // static mode. smf only: rivals inside IPOPT are not modelled yet.
    // IPOPT's dynamic scaling (MC64 whenever delays grow by > 0.05 n over
    // the last MC64 factor, reuse in between; common/ipopt_dynamic_scaling.hpp)
    // is emulated by the adapter over the replayed steps. With
    // SMF_IPM_REANALYSE=1 a re-analysis runs inside factor() and its time
    // lands in factor_ms, not analyse_ms.
    // The IpmSession path (--ipm-static or the ipopt-plugin track): a singular
    // factor is one IPOPT rejects and refactors with delta_c, so a replay
    // skips it, and the analysis runs inside the first factor().
    const bool ipm_path = !ipm_static.empty() || plugin_track;
    std::unique_ptr<smf::bench::SolverAdapter> adapter;
    if (plugin_track) {
        if (solver_id == "smf") {
            smf::IpmControl ipm = smf::ipopt_plugin_ipm_control();
            if (!ipm_static.empty()) {
                const auto ipm_mode = smf::bench::parse_ipm_static(ipm_static);
                if (!ipm_mode) {
                    std::fprintf(stderr, "--ipm-static needs off|fallback|regularized\n");
                    return 2;
                }
                ipm.static_mode = *ipm_mode;
                ipm.static_as_zero = ipm_static_zero;
            }
            adapter = smf::bench::make_smf_ipopt_plugin_adapter(ipm);
        }
    } else if (!ipm_static.empty()) {
        const auto ipm_mode = smf::bench::parse_ipm_static(ipm_static);
        if (!ipm_mode || solver_id != "smf") {
            std::fprintf(stderr, "--ipm-static needs --solver smf and off|fallback|regularized\n");
            return 2;
        }
        adapter = smf::bench::make_smf_ipm_adapter(*ipm_mode, ipm_static_zero);
    } else {
        adapter = smf::bench::make_solver_adapter(solver_id);
    }
    smf::bench::RunRecord rec;
    rec.schema_version = 1;
    rec.matrix_id = case_id;
    rec.sequence_id = sequence_id;
    rec.family_id = family_id;
    rec.solver_id = solver_id;
    rec.mode = mode;
    rec.track = track;
    rec.trial_id = trial_id;
    rec.warmup = is_warmup;
    rec.nrhs = nrhs;
    rec.threads = threads;
    rec.blas_threads = threads;
#ifdef SMF_BLAS_BACKEND_STR
    rec.blas_backend = SMF_BLAS_BACKEND_STR;
#else
    rec.blas_backend = "unknown";
#endif

    if (!adapter) {
        rec.outcome = smf::bench::Outcome::Unavailable;
        rec.reason = plugin_track ? "the ipopt-plugin track runs smf only (rivals inside IPOPT are not modelled)"
                                  : "unregistered solver adapter: " + solver_id;
        if (!jsonl_out.empty()) {
            smf::bench::append_run_record_jsonl(jsonl_out, rec);
        }
        return 1;
    }

    const auto ident = adapter->identity();
    rec.solver_version = ident.version;
    rec.library_path = ident.library_path;

    bool is_spd = false;
    smf::CscLower A;
    if (!mtx_path.empty()) {
        if (!load_bin_cache(mtx_path, A)) {
            auto mm = smf::read_matrix_market(mtx_path);
            if (!mm.ok) {
                rec.outcome = smf::bench::Outcome::InvalidInput;
                rec.reason = mm.error_message;
                if (!jsonl_out.empty()) smf::bench::append_run_record_jsonl(jsonl_out, rec);
                return 1;
            }
            A = std::move(mm.matrix);
            save_bin_cache(mtx_path, A);
        }
    } else {
        A = make_builtin_fixture(case_id, is_spd);
    }
    if (mtype == "spd") is_spd = true;
    else if (mtype == "indefinite") is_spd = false;

    if (!export_mtx.empty()) {
        if (!write_matrix_market(export_mtx, A)) {
            std::fprintf(stderr, "cannot write %s\n", export_mtx.c_str());
            return 2;
        }
        std::printf("mtype=%s\n", is_spd ? "spd" : "indefinite");
        return 0;
    }
    if (!export_step.empty()) {
        smf::CscLower S = A;
        if (mode == "warm_sequence" && !last_replayed_values(A, seq_values_path, seq_steps, S.values)) {
            std::fprintf(stderr, "unreadable --seq-values file %s\n", seq_values_path.c_str());
            return 2;
        }
        if (!write_matrix_market(export_step, S)) {
            std::fprintf(stderr, "cannot write %s\n", export_step.c_str());
            return 2;
        }
        std::printf("mtype=%s\n", is_spd ? "spd" : "indefinite");
        return 0;
    }

    rec.n = A.n;
    rec.nnz = A.nnz();

    if (!ident.available) {
        rec.outcome = smf::bench::Outcome::Unavailable;
        rec.reason = adapter->stats().reason;
        if (!jsonl_out.empty()) {
            smf::bench::append_run_record_jsonl(jsonl_out, rec);
        }
        return 1;
    }

    if ((is_spd && !ident.supports_spd) ||
        (!is_spd && !ident.supports_indefinite)) {
        rec.outcome = smf::bench::Outcome::Unavailable;
        rec.reason = is_spd ? "solver does not support SPD"
                            : "solver does not support indefinite matrices";
        if (!jsonl_out.empty()) {
            smf::bench::append_run_record_jsonl(jsonl_out, rec);
        }
        return 1;
    }

    smf::Control ctrl;
    ctrl.matrix_type = is_spd
                           ? smf::MatrixType::RealSymmetricPositiveDefinite
                           : smf::MatrixType::RealSymmetricIndefinite;
    ctrl.num_threads = threads;
    // "ipopt-plugin" track: the plugin's Control as is (IPOPT's KKT matrix
    // type, the plugin's num_threads included: `threads` above); the overrides below still
    // apply on top.
    if (plugin_track) {
        const auto plugin_ctrl = smf::ipopt_plugin_control();
        if (!plugin_ctrl) {
            std::fprintf(stderr, "smf::ipopt_plugin_control() rejected the default MA97 options\n");
            return 2;
        }
        ctrl = *plugin_ctrl;
    }
    // "ipopt" track: the pivot threshold IPOPT itself drives HSL solvers with
    // (ma27_pivtol / ma57_pivtol / ma97_u default 1e-8), so KKT sequences
    // captured from IPOPT are factorized the way IPOPT factorizes them.
    if (track == "ipopt") ctrl.pivot_u = 1e-8;
    if (pivot_u_override >= 0.0) ctrl.pivot_u = pivot_u_override;
    if (scaling_override == "none") ctrl.scaling = smf::ScalingMethod::None;
    else if (scaling_override == "equil") ctrl.scaling = smf::ScalingMethod::Equilibration;
    else if (scaling_override == "matching") ctrl.scaling = smf::ScalingMethod::Matching;
    else if (scaling_override == "auto") ctrl.scaling = smf::ScalingMethod::Auto;
    if (ordering_override == "amd") ctrl.ordering = smf::OrderingMethod::AMD;
    else if (ordering_override == "metis") ctrl.ordering = smf::OrderingMethod::METIS;
    else if (ordering_override == "matching") ctrl.ordering = smf::OrderingMethod::Matching;
    else if (ordering_override == "auto") ctrl.ordering = smf::OrderingMethod::AutoSerial;
    else if (ordering_override == "auto_parallel") ctrl.ordering = smf::OrderingMethod::AutoParallel;
    // run.py has no per-run flag for it: SMF_BENCH_MATCHING_TRIGGER
    // (predicted | realized) stands in when --matching-trigger is absent.
    if (matching_trigger.empty()) {
        if (const char *e = std::getenv("SMF_BENCH_MATCHING_TRIGGER")) matching_trigger = e;
    }
    if (matching_trigger == "realized") ctrl.matching_trigger = smf::MatchingTrigger::RealizedDelays;
    else if (matching_trigger == "predicted") ctrl.matching_trigger = smf::MatchingTrigger::Predicted;
    // Front-kernel tuning knobs (run.py has no per-run flags for them):
    // SMF_BENCH_FRONT_KERNEL = tpp | aptp | auto, SMF_BENCH_APTP_NB,
    // SMF_BENCH_APTP_IB, SMF_BENCH_APTP_MIN_ELIG (integers).
    if (const char *e = std::getenv("SMF_BENCH_FRONT_KERNEL")) {
        const std::string k = e;
        if (k == "tpp") ctrl.front_kernel = smf::FrontKernel::Tpp;
        else if (k == "aptp") ctrl.front_kernel = smf::FrontKernel::Aptp;
        else if (k == "auto") ctrl.front_kernel = smf::FrontKernel::Auto;
    }
    // SMF_BENCH_AMALGAMATION = smf | cholmod | cost (merge rule).
    if (const char *e = std::getenv("SMF_BENCH_AMALGAMATION")) {
        const std::string k = e;
        if (k == "smf") ctrl.amalgamation = smf::Amalgamation::Smf;
        else if (k == "cholmod") ctrl.amalgamation = smf::Amalgamation::SmfPlusCholmod;
        else if (k == "cost") ctrl.amalgamation = smf::Amalgamation::CostModel;
    }
    if (const char *e = std::getenv("SMF_BENCH_APTP_NB")) ctrl.aptp_nb = std::atoi(e);
    if (const char *e = std::getenv("SMF_BENCH_APTP_IB")) ctrl.aptp_ib = std::atoi(e);
    if (const char *e = std::getenv("SMF_BENCH_APTP_MIN_ELIG")) ctrl.aptp_min_elig = std::atoi(e);
    // Static pivoting / refinement knobs (no run.py flags either):
    // SMF_BENCH_STATIC_PIVOT_TOL (double), SMF_BENCH_DELAY_BUDGET (int),
    // SMF_BENCH_REFINE_METHOD = classical | fgmres | auto. The smf adapter
    // refines with smf::refine whenever its factor has static pivots.
    if (const char *e = std::getenv("SMF_BENCH_STATIC_PIVOT_TOL")) ctrl.static_pivot_tol = std::atof(e);
    if (const char *e = std::getenv("SMF_BENCH_DELAY_BUDGET")) ctrl.delay_budget = std::atoi(e);
    // SMF_BENCH_SCALE_ON_DELAY (double; negative = off): Control::scale_on_delay_ratio.
    if (const char *e = std::getenv("SMF_BENCH_SCALE_ON_DELAY")) ctrl.scale_on_delay_ratio = std::atof(e);
    // SMF_BENCH_SCALE_ON_DELAY_NEXT (double; negative = off): Control::scale_on_delay_next_ratio.
    if (const char *e = std::getenv("SMF_BENCH_SCALE_ON_DELAY_NEXT")) ctrl.scale_on_delay_next_ratio = std::atof(e);
    // SMF_BENCH_SCALE_ON_DELAY_EARLY (double; negative = off): Control::scale_on_delay_early_ratio.
    if (const char *e = std::getenv("SMF_BENCH_SCALE_ON_DELAY_EARLY")) ctrl.scale_on_delay_early_ratio = std::atof(e);
    // SMF_BENCH_SCALE_ON_DELAY_EARLY_PROJ (double): Control::scale_on_delay_early_proj.
    if (const char *e = std::getenv("SMF_BENCH_SCALE_ON_DELAY_EARLY_PROJ")) ctrl.scale_on_delay_early_proj = std::atof(e);
    // SMF_BENCH_MATCHER = ssp | hybrid: Control::matching_algorithm
    // (ShortestPath, Hybrid; the auto matching trial's matcher). Anything
    // else is an error (exit code 2), not a silent fallback to the default.
    if (const char *e = std::getenv("SMF_BENCH_MATCHER")) {
        const std::string k = e;
        if (k == "ssp") ctrl.matching_algorithm = smf::MatchingAlgorithm::ShortestPath;
        else if (k == "hybrid") ctrl.matching_algorithm = smf::MatchingAlgorithm::Hybrid;
        else {
            std::fprintf(stderr, "error: SMF_BENCH_MATCHER=\"%s\" is not ssp or hybrid\n", e);
            return 2;
        }
    }
    // SMF_BENCH_AUTO_THREADS = 0 | 1 (Control::auto_threads).
    if (const char *e = std::getenv("SMF_BENCH_AUTO_THREADS")) ctrl.auto_threads = std::atoi(e) != 0;
    // SMF_BENCH_THREAD_COSTS = "per_thread,per_top_front" or "per_thread,
    // per_top_front,per_split_front_thread,split_cap" (Control::
    // thread_cost_per_thread, thread_cost_per_top_front,
    // thread_cost_per_split_front_thread, thread_split_scaling_cap; work
    // units, the cap an integer). Anything else keeps the defaults.
    if (const char *e = std::getenv("SMF_BENCH_THREAD_COSTS")) {
        double v[4] = {0, 0, 0, 0};
        int n = 0;
        const char *p = e;
        bool ok = true;
        while (ok && n < 4) {
            char *end = nullptr;
            v[n] = std::strtod(p, &end);
            ok = end != p && (*end == ',' || *end == '\0');
            if (!ok) break;
            ++n;
            p = end;
            if (*p == '\0') break;
            ++p;  // the comma; a trailing one leaves p at the end and fails the next parse
        }
        ok = ok && *p == '\0' && (n == 2 || n == 4) &&
             (n < 4 || (v[3] == std::floor(v[3]) && v[3] >= 1 && v[3] < 1e6));
        if (ok) {
            ctrl.thread_cost_per_thread = v[0];
            ctrl.thread_cost_per_top_front = v[1];
            if (n == 4) {
                ctrl.thread_cost_per_split_front_thread = v[2];
                ctrl.thread_split_scaling_cap = static_cast<int>(v[3]);
            }
        } else {
            std::fprintf(stderr, "warning: SMF_BENCH_THREAD_COSTS=\"%s\" is not two or four comma-separated numbers (the last a positive integer); keeping the defaults\n", e);
        }
    }
    // SMF_BENCH_SOLVE_COSTS = "per_thread,per_top_step,per_top_row" (Control::
    // solve_cost_per_thread, solve_cost_per_top_step, solve_cost_per_top_row;
    // step-work units). Anything but exactly three numbers keeps the defaults.
    if (const char *e = std::getenv("SMF_BENCH_SOLVE_COSTS")) {
        double v[3];
        const char *p = e;
        bool ok = true;
        for (int i = 0; i < 3 && ok; ++i) {
            char *end = nullptr;
            v[i] = std::strtod(p, &end);
            ok = end != p && *end == (i < 2 ? ',' : '\0');
            p = end + (i < 2 ? 1 : 0);
        }
        if (ok) {
            ctrl.solve_cost_per_thread = v[0];
            ctrl.solve_cost_per_top_step = v[1];
            ctrl.solve_cost_per_top_row = v[2];
        } else {
            std::fprintf(stderr, "warning: SMF_BENCH_SOLVE_COSTS=\"%s\" is not three comma-separated numbers; keeping the defaults\n", e);
        }
    }
    // SMF_BENCH_SCALE_REUSE_TOL (double; negative = recompute): Control::scale_on_delay_reuse_tol.
    if (const char *e = std::getenv("SMF_BENCH_SCALE_REUSE_TOL")) ctrl.scale_on_delay_reuse_tol = std::atof(e);
    // SMF_BENCH_CACHE_INVERSE_DIAG = 0 | 1 (Control::cache_inverse_diag).
    if (const char *e = std::getenv("SMF_BENCH_CACHE_INVERSE_DIAG")) ctrl.cache_inverse_diag = std::atoi(e) != 0;
    if (const char *e = std::getenv("SMF_BENCH_REFINE_METHOD")) {
        const std::string k = e;
        if (k == "classical") ctrl.refine_method = smf::RefineMethod::Classical;
        else if (k == "fgmres") ctrl.refine_method = smf::RefineMethod::Fgmres;
        else if (k == "auto") ctrl.refine_method = smf::RefineMethod::Auto;
    }
    adapter->configure(ctrl, track);

    const std::size_t total_len =
        static_cast<std::size_t>(A.n) * static_cast<std::size_t>(nrhs);
    std::vector<double> b(total_len, 1.0);
    std::vector<double> x = b;

    // Untimed in-process warmup so one-time runtime/DLL initialization is excluded
    // from timed solver measurements.
    {
        std::vector<double> x_warm = b;
        if (adapter->analyse(A) == smf::bench::Outcome::Success &&
            adapter->factor(A.values) == smf::bench::Outcome::Success) {
            (void)adapter->solve(x_warm, nrhs);
        }
    }

    smf::bench::Outcome out_a = smf::bench::Outcome::Success;
    smf::bench::Outcome out_f = smf::bench::Outcome::Success;
    smf::bench::Outcome out_s = smf::bench::Outcome::Success;

    if (mode == "warm_sequence") {
        // Recorded sequences replay --seq-steps of their factorizations;
        // synthetic ones always 5.
        int kSeqSteps = 5;
        std::vector<std::size_t> picks;
        const std::size_t step_bytes = A.values.size() * sizeof(double);
        std::ifstream in;
        if (!seq_values_path.empty()) {
            in.open(seq_values_path, std::ios::binary | std::ios::ate);
            const std::size_t bytes = in ? static_cast<std::size_t>(in.tellg()) : 0;
            const std::size_t nsteps = step_bytes ? bytes / step_bytes : 0;
            if (nsteps == 0) {
                rec.outcome = smf::bench::Outcome::InvalidInput;
                rec.reason = "unreadable --seq-values file";
                if (!jsonl_out.empty()) smf::bench::append_run_record_jsonl(jsonl_out, rec);
                return 1;
            }
            picks = smf::bench::replay_step_picks(nsteps, seq_steps);
            kSeqSteps = static_cast<int>(picks.size());
        }
        rec.batch_count = kSeqSteps;
        rec.per_update_validation.clear();
        rec.per_update_validation.reserve(static_cast<std::size_t>(kSeqSteps));

        // Pre-build the numeric value updates outside the timed region:
        // a recorded sequence (e.g. IPOPT iterations) when given, otherwise
        // synthetic diagonal perturbations.
        std::vector<std::vector<double>> seq_values(static_cast<std::size_t>(kSeqSteps), A.values);
        bool recorded = false;
        if (!seq_values_path.empty()) {
            for (int step = 0; step < kSeqSteps; ++step) {
                const std::size_t pick = picks[static_cast<std::size_t>(step)];
                in.seekg(static_cast<std::streamoff>(pick * step_bytes));
                in.read(reinterpret_cast<char *>(seq_values[static_cast<std::size_t>(step)].data()),
                        static_cast<std::streamsize>(step_bytes));
            }
            recorded = static_cast<bool>(in);
        }
        for (int step = 0; step < kSeqSteps && !recorded; ++step) {
            const double delta = 1.0 + 0.05 * static_cast<double>(step);
            for (smf::Int j = 0; j < A.n; ++j) {
                const smf::Int p0 = A.col_ptr[static_cast<std::size_t>(j)];
                if (p0 < A.col_ptr[static_cast<std::size_t>(j) + 1] &&
                    A.row_idx[static_cast<std::size_t>(p0)] == j) {
                    const double v0 = A.values[static_cast<std::size_t>(p0)];
                    if (v0 != 0.0) {
                        seq_values[step][static_cast<std::size_t>(p0)] = v0 * delta;
                    }
                }
            }
        }

        std::vector<std::vector<double>> step_solutions(static_cast<std::size_t>(kSeqSteps), b);
        double total_factor_ms = 0.0;
        int seq_delay_rescales = 0;
        int seq_delay_early_stops = 0;
        double seq_delay_try_ms = 0.0;       // summed over the steps, like seq_delay_rescales
        double seq_delay_matching_ms = 0.0;
        std::vector<double> step_factor_ms;  // successful factors, in order
        double total_solve_ms = 0.0;
        double total_refine_ms = 0.0;
        const bool refine = smf::bench::refine_on_track(track, nrhs);
        // IpmSession path: a singular factor (zero pivots) is one IPOPT rejects
        // and refactors with delta_c; it is never solved with, so the replay
        // skips its solve and its validation instead of stopping.
        std::vector<char> skipped(static_cast<std::size_t>(kSeqSteps), 0);

        const auto t0 = std::chrono::steady_clock::now();
        out_a = adapter->analyse(A);
        double analyse_ms = adapter->stats().analyse_ms;
        bool has_phases = adapter->stats().has_analyse_phases;
        smf::AnalysePhaseSeconds phases = adapter->stats().analyse_phases;
        if (out_a == smf::bench::Outcome::Success) {
            for (int step = 0; step < kSeqSteps; ++step) {
                out_f = adapter->factor(seq_values[step]);
                total_factor_ms += adapter->stats().factor_ms;
                seq_delay_rescales += adapter->stats().delay_rescales;
                seq_delay_early_stops += adapter->stats().delay_early_stops;
                seq_delay_try_ms += adapter->stats().delay_try_ms;
                seq_delay_matching_ms += adapter->stats().delay_matching_ms;
                if (out_f == smf::bench::Outcome::Success) step_factor_ms.push_back(adapter->stats().factor_ms);
                // The IpmSession adapter runs the deferred analysis in factor()
                // and adds its time to analyse_ms (not to factor_ms).
                if (step == 0 && ipm_path) analyse_ms = adapter->stats().analyse_ms;
                if (step == 0 && !has_phases && adapter->stats().has_analyse_phases) {
                    has_phases = true;
                    phases = adapter->stats().analyse_phases;
                }
                if (ipm_path && out_f == smf::bench::Outcome::Singular) {
                    const auto fst = adapter->stats();
                    skipped[static_cast<std::size_t>(step)] = 1;
                    out_f = smf::bench::Outcome::Success;
                    std::fprintf(stderr, "step %d (stored %zu): singular, skipped: static=%d inertia=%d/%d/%d\n",
                                 step, recorded ? picks[static_cast<std::size_t>(step)] : static_cast<std::size_t>(step),
                                 fst.static_pivots, fst.inertia_pos, fst.inertia_neg, fst.inertia_zero);
                    continue;
                }
                // A failed factor fails the replay (out_s would otherwise stay Success
                // and the untouched right-hand sides would be validated).
                if (out_f != smf::bench::Outcome::Success) {
                    out_s = out_f;
                    break;
                }
                out_s = adapter->solve(step_solutions[step], nrhs);
                total_solve_ms += adapter->stats().solve_ms;
                if (out_s != smf::bench::Outcome::Success) break;
                if (refine) {
                    const auto tr0 = std::chrono::steady_clock::now();
                    smf::bench::refine_solution(
                        [&adapter](std::vector<double> &r) {
                            return adapter->solve(r, 1) == smf::bench::Outcome::Success;
                        },
                        A, seq_values[static_cast<std::size_t>(step)], b,
                        step_solutions[static_cast<std::size_t>(step)], refine_steps);
                    total_refine_ms += std::chrono::duration<double, std::milli>(
                                           std::chrono::steady_clock::now() - tr0).count();
                }
                if (!ipm_static.empty() || print_steps) {
                    const auto fst = adapter->stats();
                    std::fprintf(stderr, "step %d (stored %zu): static=%d delays=%d rescaled=%d early=%d inertia=%d/%d/%d factor_ms=%.3f threads=%d subtrees=%d top=%d pops=%d sub_ms=%.3f top_ms=%.3f gather_ms=%.3f solve_threads=%d try_ms=%.3f match_ms=%.3f\n", step,
                                 recorded ? picks[static_cast<std::size_t>(step)] : static_cast<std::size_t>(step),
                                 fst.static_pivots, fst.delayed_pivots, fst.delay_rescales, fst.delay_early_stops, fst.inertia_pos,
                                 fst.inertia_neg, fst.inertia_zero, fst.factor_ms, fst.threads_used,
                                 fst.parallel_subtrees, fst.top_part_fronts, fst.top_part_queue_pops, fst.subtree_ms, fst.top_ms,
                                 fst.gather_ms, fst.solve_threads, fst.delay_try_ms, fst.delay_matching_ms);
                }
            }
            if (!ipm_static.empty() && out_s != smf::bench::Outcome::Success) {
                const auto fst = adapter->stats();
                std::fprintf(stderr, "stopped: factor %s, solve %s (%s); static=%d inertia=%d/%d/%d\n",
                             smf::bench::outcome_to_string(out_f), smf::bench::outcome_to_string(out_s),
                             fst.reason.c_str(), fst.static_pivots, fst.inertia_pos, fst.inertia_neg,
                             fst.inertia_zero);
            }
        } else {
            out_f = out_a;
            out_s = out_a;
        }
        const auto t1 = std::chrono::steady_clock::now();

        const auto st = adapter->stats();
        rec.analyse_ms = analyse_ms;
        rec.has_analyse_phases = has_phases;
        rec.analyse_phases = phases;
        rec.factor_ms = total_factor_ms;
        if (!step_factor_ms.empty()) {
            rec.first_factor_ms = step_factor_ms.front();
            if (step_factor_ms.size() > 1) {
                std::vector<double> warm(step_factor_ms.begin() + 1, step_factor_ms.end());
                std::sort(warm.begin(), warm.end());
                const std::size_t m = warm.size();
                rec.refactor_ms = (m % 2 == 1) ? warm[m / 2] : 0.5 * (warm[m / 2 - 1] + warm[m / 2]);
            }
        }
        rec.solve_ms = total_solve_ms;
        rec.refine_ms = total_refine_ms;
        rec.wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        rec.peak_rss_bytes = smf::bench::read_peak_rss_bytes();
        rec.inertia_pos = st.inertia_pos;
        rec.inertia_neg = st.inertia_neg;
        rec.inertia_zero = st.inertia_zero;
        rec.delayed_pivots = st.delayed_pivots;
        if (solver_id == "smf") {
            rec.aptp_failed_columns = st.aptp_failed_columns;
            rec.static_pivots = st.static_pivots;
            rec.delay_rescales = seq_delay_rescales;
            rec.delay_early_stops = seq_delay_early_stops;
            rec.delay_try_ms = seq_delay_try_ms;
            rec.delay_matching_ms = seq_delay_matching_ms;
            rec.refine_iterations = st.refine_iterations;
            if (st.refine_iterations >= 0) rec.refine_method = st.refine_fgmres ? "fgmres" : "classical";
        }
        rec.factor_entries = st.factor_entries;
        rec.flops = st.flops;
        if (solver_id == "smf" && st.order_hash != 0) rec.order_hash = order_hash_hex(st.order_hash);

        int last_solved = kSeqSteps - 1;
        while (last_solved > 0 && skipped[static_cast<std::size_t>(last_solved)]) --last_solved;
        if (out_s == smf::bench::Outcome::Success && skipped[static_cast<std::size_t>(last_solved)]) {
            out_s = smf::bench::Outcome::Singular;  // every step singular: nothing was solved
            rec.reason = "every replayed factor was singular";
        }
        if (out_s == smf::bench::Outcome::Success) {
            for (int step = 0; step < kSeqSteps; ++step) {
                if (skipped[static_cast<std::size_t>(step)]) continue;
                smf::CscLower A_step = A;
                A_step.values = seq_values[step];
                rec.per_update_validation.push_back(
                    smf::bench::validate(A_step, b, step_solutions[step], nrhs));
            }
            smf::CscLower A_last = A;
            A_last.values = seq_values[static_cast<std::size_t>(last_solved)];
            x = step_solutions[static_cast<std::size_t>(last_solved)];
            rec.per_rhs_validation = smf::bench::validate_per_rhs(A_last, b, x, nrhs);
            rec.validation = smf::bench::validate(A_last, b, x, nrhs);
            rec.outcome = smf::bench::qualify_outcome(out_s, rec.validation);
            for (const auto &vu : rec.per_update_validation) {
                if (!smf::bench::is_qualified_validation(vu)) {
                    rec.outcome = smf::bench::Outcome::Inaccurate;
                }
            }
            if (rec.outcome != smf::bench::Outcome::Success) {
                rec.reason = "exceeded backward error threshold";
            }
        } else {
            rec.outcome = out_s;
            rec.reason = st.reason;
        }
    } else {
        const int batch_count = (A.n <= 64) ? 10 : 1;
        rec.batch_count = batch_count;
        double sum_a = 0.0, sum_f = 0.0, sum_s = 0.0, sum_r = 0.0;

        const auto t0 = std::chrono::steady_clock::now();
        for (int b_it = 0; b_it < batch_count; ++b_it) {
            x = b;
            out_a = adapter->analyse(A);
            double a_ms = adapter->stats().analyse_ms;
            out_f = (out_a == smf::bench::Outcome::Success)
                        ? adapter->factor(A.values)
                        : out_a;
            // The IpmSession adapter runs the deferred analysis in factor()
            // and adds its time to analyse_ms (not to factor_ms).
            if (ipm_path && out_a == smf::bench::Outcome::Success) a_ms = adapter->stats().analyse_ms;
            sum_a += a_ms;
            sum_f += adapter->stats().factor_ms;
            out_s = (out_f == smf::bench::Outcome::Success)
                        ? adapter->solve(x, nrhs)
                        : out_f;
            sum_s += adapter->stats().solve_ms;
            if (out_s != smf::bench::Outcome::Success) break;
            // IPOPT-style refinement on the ipopt track, identical for every
            // solver and to the warm_sequence path (R29). Not part of solve_ms.
            sum_r += smf::bench::refine_timed(
                track, nrhs,
                [&adapter](std::vector<double> &r) {
                    return adapter->solve(r, 1) == smf::bench::Outcome::Success;
                },
                A, A.values, b, x, refine_steps);
        }
        const auto t1 = std::chrono::steady_clock::now();

        const auto st = adapter->stats();
        const double inv_batch = 1.0 / static_cast<double>(batch_count);
        rec.analyse_ms = sum_a * inv_batch;
        rec.has_analyse_phases = st.has_analyse_phases;  // last repetition's phases
        rec.analyse_phases = st.analyse_phases;
        rec.factor_ms = sum_f * inv_batch;
        rec.solve_ms = sum_s * inv_batch;
        rec.refine_ms = sum_r * inv_batch;
        rec.wall_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count() * inv_batch;
        rec.peak_rss_bytes = smf::bench::read_peak_rss_bytes();
        rec.inertia_pos = st.inertia_pos;
        rec.inertia_neg = st.inertia_neg;
        rec.inertia_zero = st.inertia_zero;
        rec.delayed_pivots = st.delayed_pivots;
        if (solver_id == "smf") {
            rec.aptp_failed_columns = st.aptp_failed_columns;
            rec.static_pivots = st.static_pivots;
            rec.delay_rescales = st.delay_rescales;
            rec.delay_early_stops = st.delay_early_stops;
            rec.delay_try_ms = st.delay_try_ms;
            rec.delay_matching_ms = st.delay_matching_ms;
            rec.refine_iterations = st.refine_iterations;
            if (st.refine_iterations >= 0) rec.refine_method = st.refine_fgmres ? "fgmres" : "classical";
        }
        rec.factor_entries = st.factor_entries;
        rec.flops = st.flops;
        if (solver_id == "smf" && st.order_hash != 0) rec.order_hash = order_hash_hex(st.order_hash);

        if (out_s == smf::bench::Outcome::Success) {
            rec.per_rhs_validation = smf::bench::validate_per_rhs(A, b, x, nrhs);
            rec.validation = smf::bench::validate(A, b, x, nrhs);
            rec.outcome = smf::bench::qualify_outcome(out_s, rec.validation);
            if (rec.outcome != smf::bench::Outcome::Success) {
                rec.reason = "exceeded backward error threshold";
            }
        } else {
            rec.outcome = out_s;
            rec.reason = st.reason;
        }
    }

    if (!jsonl_out.empty()) {
        smf::bench::append_run_record_jsonl(jsonl_out, rec);
    }

    return rec.outcome == smf::bench::Outcome::Success ? 0 : 1;
}
