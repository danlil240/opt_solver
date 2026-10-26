#pragma once

// Which stored factorizations of a recorded value sequence bench_replay
// replays (--seq-steps N|all).

#include <cstddef>
#include <vector>

namespace smf::bench {

// Indices into a sequence of `nsteps` stored factorizations: `want` evenly
// spaced ones from the first to the last (repeats when nsteps < want; the
// default 5 is the harness's warm_sequence), or every one when want < 0.
inline std::vector<std::size_t> replay_step_picks(std::size_t nsteps, int want) {
    std::vector<std::size_t> picks;
    if (nsteps == 0) return picks;
    if (want < 0) {
        for (std::size_t i = 0; i < nsteps; ++i) picks.push_back(i);
        return picks;
    }
    for (int step = 0; step < want; ++step) {
        picks.push_back((nsteps == 1 || want == 1)
                            ? 0
                            : static_cast<std::size_t>(step) * (nsteps - 1) / static_cast<std::size_t>(want - 1));
    }
    return picks;
}

} // namespace smf::bench
