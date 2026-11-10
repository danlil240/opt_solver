// SMF_BENCH_PHASE_TRACE=1: an indefinite factorization prints its phase marks
// (solver/src/phase_trace.hpp) to stderr, in order. The environment is read
// once per process, so this executable sets it before anything factors.

#include "kkt_test_matrices.hpp"
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
    _putenv_s("SMF_BENCH_PHASE_TRACE", "1");
#else
    setenv("SMF_BENCH_PHASE_TRACE", "1", 1);
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
