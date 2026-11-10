// Phase marks of a factorization (see phase_trace.hpp).

#include "phase_trace.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace smf {
namespace {
const auto kStart = std::chrono::steady_clock::now();
} // namespace

bool phase_trace_on() noexcept {
  static const bool on = [] {
#ifdef _WIN32
    char *e = nullptr;
    std::size_t len = 0;
    if (_dupenv_s(&e, &len, "SMF_BENCH_PHASE_TRACE") != 0 || e == nullptr) return false;
    const bool set = std::atoi(e) != 0;
    std::free(e);
    return set;
#else
    const char *e = std::getenv("SMF_BENCH_PHASE_TRACE");
    return e != nullptr && std::atoi(e) != 0;
#endif
  }();
  return on;
}

void phase_mark(const char *name) noexcept {
  if (!phase_trace_on()) return;
  const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - kStart).count();
  std::fprintf(stderr, "# seg %s t=%.7f\n", name, t);
}

} // namespace smf
