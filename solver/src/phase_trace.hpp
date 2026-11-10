#pragma once
// Phase marks of a factorization (SMF_BENCH_PHASE_TRACE=1): `# seg <name> t=<s>`
// on stderr at fixed points of factor_indef and the multifrontal engine, read
// by tools/bench/phase_split.py. Off (the default) a mark is one load of a
// cached flag. Marks, in order: fi_entry, fi_mapped, fi_scaled, fi_call,
// mf_setup, then on the tree-parallel engine mf_p2, mf_p3, mf_gathered,
// mf_steps, mf_diag, mf_top, mf_toprows, then mf_ret, fi_end.

namespace smf {

// SMF_BENCH_PHASE_TRACE=1 in the environment (read once).
bool phase_trace_on() noexcept;

// Prints the mark with the steady-clock time since process start when on.
void phase_mark(const char *name) noexcept;

} // namespace smf
