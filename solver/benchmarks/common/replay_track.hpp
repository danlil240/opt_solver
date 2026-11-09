#pragma once
// replay_track.hpp — the Control settings a bench_replay track (--track) adds
// to smf's defaults. Header-only: bench_replay and the tests share it.

#include "smf/control.hpp"
#include "smf/ipopt_plugin_control.hpp"

#include <string_view>

namespace smf::bench {

// "ipopt": KKT sequences captured from IPOPT are factorized the way IPOPT
// drives its linear solvers: IPOPT's pivot threshold (ma27_pivtol /
// ma57_pivtol / ma97_u default 1e-8), and analysed for as many
// refactorizations as the IPOPT plugin expects (kIpoptExpectedRefactors).
// Other tracks keep the defaults.
inline void apply_replay_track(Control &ctrl, std::string_view track) {
    if (track == "ipopt") {
        ctrl.pivot_u = 1e-8;
        ctrl.expected_refactors = kIpoptExpectedRefactors;
    }
}

} // namespace smf::bench
