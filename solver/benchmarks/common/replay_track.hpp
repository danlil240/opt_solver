#pragma once
// replay_track.hpp — the Control settings a bench_replay track (--track) adds
// to smf's defaults. Header-only: bench_replay and the tests share it.

#include "smf/control.hpp"
#include "smf/ipopt_plugin_control.hpp"

#include <string_view>

namespace smf::bench {

// "ipopt": KKT matrices captured from IPOPT are factorized the way IPOPT
// drives its linear solvers: IPOPT's pivot threshold (ma27_pivtol /
// ma57_pivtol / ma97_u default 1e-8). A replayed sequence (`sequence`:
// --mode warm_sequence) stands for an IPOPT run, so it is also analysed for
// as many refactorizations as the IPOPT plugin expects
// (kIpoptExpectedRefactors); one cold factorization is analysed for one.
// Other tracks keep the defaults.
inline void apply_replay_track(Control &ctrl, std::string_view track, bool sequence) {
    if (track == "ipopt") {
        ctrl.pivot_u = 1e-8;
        if (sequence) ctrl.expected_refactors = kIpoptExpectedRefactors;
    }
}

} // namespace smf::bench
