// Phase marks of a factorization (see phase_trace.hpp).

#include "phase_trace.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace smf {
namespace {
const auto kStart = std::chrono::steady_clock::now();
constexpr int kSlots = static_cast<int>(PhaseSum::Count);
const char *const kSlotNames[kSlots] = {"top_asm", "top_panel", "top_defer", "top_schur"};
std::atomic<long long> g_sum_ns[kSlots] = {};
thread_local int t_top_front = 0;
} // namespace

int phase_trace_level() noexcept {
  static const int level = [] {
#ifdef _WIN32
    char *e = nullptr;
    std::size_t len = 0;
    if (_dupenv_s(&e, &len, "SMF_BENCH_PHASE_TRACE") != 0 || e == nullptr) return 0;
    const int v = std::atoi(e);
    std::free(e);
#else
    const char *e = std::getenv("SMF_BENCH_PHASE_TRACE");
    const int v = e != nullptr ? std::atoi(e) : 0;
#endif
    return v > 0 ? v : 0;
  }();
  return level;
}

bool phase_trace_on() noexcept { return phase_trace_level() > 0; }

void phase_mark(const char *name) noexcept {
  if (!phase_trace_on()) return;
  const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - kStart).count();
  std::fprintf(stderr, "# seg %s t=%.7f\n", name, t);
}

void phase_sum_add(PhaseSum slot, double seconds) noexcept {
  if (phase_trace_level() < 2) return;
  g_sum_ns[static_cast<int>(slot)].fetch_add(std::llround(seconds * 1e9), std::memory_order_relaxed);
}

void phase_sum_flush() noexcept {
  if (phase_trace_level() < 2) return;
  for (int s = 0; s < kSlots; ++s) {
    const long long ns = g_sum_ns[s].exchange(0, std::memory_order_relaxed);
    std::fprintf(stderr, "# sum %s s=%.9f\n", kSlotNames[s], static_cast<double>(ns) * 1e-9);
  }
}

void phase_sum_discard() noexcept {
  if (phase_trace_level() < 2) return;
  for (auto &g : g_sum_ns) g.store(0, std::memory_order_relaxed);
}

PhaseTopFrontScope::PhaseTopFrontScope() noexcept : on_(phase_trace_level() >= 2) {
  if (on_) ++t_top_front;
}
PhaseTopFrontScope::~PhaseTopFrontScope() noexcept {
  if (on_) --t_top_front;
}
bool phase_in_top_front() noexcept { return t_top_front > 0; }

} // namespace smf
