// SMF_BENCH_PHASE_TRACE=1: an indefinite factorization prints its phase marks
// (solver/src/phase_trace.hpp) to stderr, in order. The environment is read
// once per process, so this executable sets it before anything factors.

#include "kkt_test_matrices.hpp"
#include "phase_trace.hpp"
#include "smf/control.hpp"
#include "smf/info.hpp"
#include "smf/solver.hpp"
#include "smf/workspace.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <io.h>
#define SMF_DUP _dup
#define SMF_DUP2 _dup2
#define SMF_CLOSE _close
#else
#include <unistd.h>
#define SMF_DUP dup
#define SMF_DUP2 dup2
#define SMF_CLOSE close
#endif

using namespace smf;

namespace {
// stderr (fd 2) into a file while fn runs; returns what was written. The
// in-tree gtest fallback has no CaptureStderr.
template <class F> std::string capture_stderr(F &&fn) {
    const std::string path = "phase_trace_stderr.txt";
    std::fflush(stderr);
    const int saved = SMF_DUP(2);
    std::FILE *f = std::fopen(path.c_str(), "w");
    SMF_DUP2(fileno(f), 2);
    fn();
    std::fflush(stderr);
    SMF_DUP2(saved, 2);
    SMF_CLOSE(saved);
    std::fclose(f);
    std::ifstream in(path);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}
}  // namespace

TEST(PhaseTrace, MarksInOrderWhenOn) {
#ifdef _WIN32
    _putenv_s("SMF_BENCH_PHASE_TRACE", "2");  // level 2: marks plus kernel sums
#else
    setenv("SMF_BENCH_PHASE_TRACE", "2", 1);
#endif
    const CscLower A = opf_like_kkt(20000, 8000, 1);
    for (const int threads : {1, 4}) {
        Control ctrl;
        ctrl.num_threads = threads;
        ctrl.auto_threads = false;
        Solver solver;
        Info info;
        auto ak = solver.analyse(A, ctrl, info);
        ASSERT_NE(ak.get(), nullptr);
        NumericWorkspace nws;
        FactorKeep fk;
        FactorStatus st = FactorStatus::Success;
        const std::string err = capture_stderr([&] { st = solver.factor(*ak, A, ctrl, info, fk, nws); });
        ASSERT_EQ(st, FactorStatus::Success);
        if (threads > 1) ASSERT_GT(info.threads_used, 1) << "premise: the tree splits at 4 threads";
        std::vector<const char *> want = {"fi_entry", "fi_mapped", "fi_scaled", "fi_call", "mf_setup"};
        if (info.threads_used > 1)
            for (const char *m : {"mf_p2", "mf_p3", "mf_gathered", "mf_steps", "mf_diag", "mf_top"})
                want.push_back(m);
        for (const char *m : {"mf_ret", "fi_end"}) want.push_back(m);
        std::size_t at = 0;
        for (const char *m : want) {
            const std::size_t k = err.find(std::string("# seg ") + m + " t=", at);
            ASSERT_NE(k, std::string::npos) << threads << "T: mark " << m << " missing or out of order in\n" << err;
            at = k;
        }
    }
}

// SMF_BENCH_PHASE_TRACE=2: after mf_p3 the factorization prints the top
// fronts' kernel time, `# sum <slot> s=<seconds>` for top_asm, top_panel,
// top_defer and top_schur, and the large top fronts' assembly and TPP panels
// are counted in them.
TEST(PhaseTrace, KernelSumsAtLevelTwo) {
#ifdef _WIN32
    _putenv_s("SMF_BENCH_PHASE_TRACE", "2");
#else
    setenv("SMF_BENCH_PHASE_TRACE", "2", 1);
#endif
    const CscLower A = opf_like_kkt(20000, 8000, 1);
    Control ctrl;
    ctrl.num_threads = 4;
    ctrl.auto_threads = false;
    Solver solver;
    Info info;
    auto ak = solver.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);
    NumericWorkspace nws;
    FactorKeep fk;
    FactorStatus st = FactorStatus::Success;
    const std::string err = capture_stderr([&] { st = solver.factor(*ak, A, ctrl, info, fk, nws); });
    ASSERT_EQ(st, FactorStatus::Success);
    ASSERT_GT(info.top_part_fronts, 0) << "premise: a top part";
    const std::size_t p3 = err.find("# seg mf_p3 t=");
    ASSERT_NE(p3, std::string::npos);
    for (const char *slot : {"top_asm", "top_panel", "top_defer", "top_schur"}) {
        const std::string key = std::string("# sum ") + slot + " s=";
        const std::size_t k = err.find(key, p3);
        ASSERT_NE(k, std::string::npos) << "sum " << slot << " missing after mf_p3 in\n" << err;
        const double v = std::atof(err.c_str() + k + key.size());
        EXPECT_GE(v, 0.0) << slot;
        if (std::string(slot) == "top_asm" || std::string(slot) == "top_panel") EXPECT_GT(v, 0.0) << slot;
    }
}

// A factorization that stops in the top phase (an inertia or work limit)
// skips the flush after mf_p3; its sums must not be printed with the next
// factorization's. A stale 5 s left in top_asm is gone by the next flush.
TEST(PhaseTrace, TopPhaseStartDiscardsStaleSums) {
#ifdef _WIN32
    _putenv_s("SMF_BENCH_PHASE_TRACE", "2");
#else
    setenv("SMF_BENCH_PHASE_TRACE", "2", 1);
#endif
    ASSERT_EQ(phase_trace_level(), 2);
    const CscLower A = opf_like_kkt(20000, 8000, 1);
    Control ctrl;
    ctrl.num_threads = 4;
    ctrl.auto_threads = false;
    Solver solver;
    Info info;
    auto ak = solver.analyse(A, ctrl, info);
    ASSERT_NE(ak.get(), nullptr);
    NumericWorkspace nws;
    FactorKeep fk;
    phase_sum_add(PhaseSum::TopAsm, 5.0);  // what an aborted top phase leaves
    FactorStatus st = FactorStatus::Success;
    const std::string err = capture_stderr([&] { st = solver.factor(*ak, A, ctrl, info, fk, nws); });
    ASSERT_EQ(st, FactorStatus::Success);
    const std::string key = "# sum top_asm s=";
    const std::size_t k = err.find(key);
    ASSERT_NE(k, std::string::npos);
    EXPECT_LT(std::atof(err.c_str() + k + key.size()), 1.0) << "a stale sum was printed with this factorization";
}

