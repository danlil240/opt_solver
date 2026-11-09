#pragma once
// Phase marks of a factorization (SMF_BENCH_PHASE_TRACE=1): `# seg <name> t=<s>`
// on stderr at fixed points of factor_indef and the multifrontal engine, read
// by tools/bench/phase_split.py. Off (the default) a mark is one load of a
// cached flag. Marks, in order: fi_entry, fi_mapped, fi_scaled, fi_call,
// mf_setup, then on the tree-parallel engine mf_p2, mf_p3, mf_gathered,
// mf_steps, mf_diag, mf_top, mf_toprows, then mf_ret, fi_end.
//
// SMF_BENCH_PHASE_TRACE=2 adds the top fronts' kernel time, summed over a
// factorization and printed after mf_p3 as `# sum <slot> s=<seconds>`:
// top_asm (assembly in process_front), top_panel (the TPP kernel's pivot
// search and application, stall handling included), top_defer (its deferred
// panel updates) and top_schur (its Schur update). The sums are over every
// top front on every thread, not the critical path.

namespace smf {

// SMF_BENCH_PHASE_TRACE level (read once): 0 off, 1 marks, 2 marks and kernel sums.
int phase_trace_level() noexcept;
// Level >= 1.
bool phase_trace_on() noexcept;

// Prints the mark with the steady-clock time since process start when on.
void phase_mark(const char *name) noexcept;

enum class PhaseSum : int { TopAsm, TopPanel, TopDefer, TopSchur, Count };
// Adds seconds to this factorization's sum (atomic; no-op below level 2).
void phase_sum_add(PhaseSum slot, double seconds) noexcept;
// Prints `# sum <name> s=<seconds>` for every slot and zeroes them (level 2).
void phase_sum_flush() noexcept;
// Zeroes the sums without printing them: the top phase starts with it, so a
// factorization that stopped before its flush (an inertia or work limit)
// leaves nothing in the next one's sums.
void phase_sum_discard() noexcept;

// Level 2 only: marks this thread as running a top front while it lives, so
// that the TPP kernel's laps are summed (phase_in_top_front()).
class PhaseTopFrontScope {
public:
  PhaseTopFrontScope() noexcept;
  ~PhaseTopFrontScope() noexcept;
  PhaseTopFrontScope(const PhaseTopFrontScope &) = delete;
  PhaseTopFrontScope &operator=(const PhaseTopFrontScope &) = delete;

private:
  bool on_ = false;
};
bool phase_in_top_front() noexcept;

} // namespace smf
