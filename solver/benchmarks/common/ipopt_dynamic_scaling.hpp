#pragma once
// IPOPT's default MA97 scaling strategy, as the ipopt-plugin track replays
// it around the plugin (bench_replay --track ipopt-plugin).
//
// IPOPT 3.14 (IpMa97SolverInterface.cpp) with its defaults, ma97_scaling =
// dynamic, ma97_switch1 = od_hd_reuse and ma97_switch2/3 = never:
//  - it factors unscaled (MA97 scaling = 0) until a factorization returns
//    successfully with num_delay - numdelay_ > 0.05 n, where numdelay_ is the
//    delay count of the last rescaled (MC64) factorization (0 before any);
//  - it then requests MC64 (scaling = 1) on the next factorization, keeps the
//    scale the solver hands back, records that factorization's delays as
//    numdelay_, and passes the scale back (scaling = 0, "reuse") on every
//    later factorization, until the rule fires again;
//  - a singular factorization returns before that check, so a pending MC64
//    request stays pending.
// The on-demand ("od") part, IPOPT's requests for higher solution quality,
// is not modelled.

#include "smf/ipm_session.hpp"
#include "smf/ipopt_plugin_control.hpp"

#include <cstddef>
#include <vector>

namespace smf::bench {

inline constexpr double kIpoptHighDelayFraction = 0.05;

class IpoptDynamicScaling {
public:
    void reset() {
        request_ = false;
        numdelay_ = 0;
        scale_.clear();
    }

    /// IPOPT asks for MC64 (MA97 scaling = 1) on this factorization.
    bool mc64_requested() const { return request_; }
    /// The scale IPOPT holds and passes back; null while it holds none.
    const double *scale() const { return scale_.empty() ? nullptr : scale_.data(); }
    int numdelay() const { return numdelay_; }

    /// After one plugin factorization: `call` is the plugin's scaling decision
    /// for it (computed possibly set by the stale-scale retry), `fk_scale` the
    /// factor's scale, `delays` its delayed pivots.
    void after_factor(IpmFactorStatus st, const Ma97ScalingCall &call, const std::vector<double> &fk_scale,
                      int delays, int n) {
        // The plugin fails the call (flag -5) without handing a scale back.
        if (st == IpmFactorStatus::Error) return;
        if (call.computed && !fk_scale.empty()) scale_ = fk_scale;
        if (call.unit_scale) scale_.assign(static_cast<std::size_t>(n), 1.0);
        // IPOPT returns SINGULAR before its delay check: the request stays.
        // (WrongInertia only arises with SMF_IPM_ABORT=1; treated the same.)
        if (st != IpmFactorStatus::Ok) return;
        if (request_) {
            numdelay_ = delays;
            request_ = false;
        }
        if (static_cast<double>(delays - numdelay_) > kIpoptHighDelayFraction * static_cast<double>(n))
            request_ = true;
    }

private:
    bool request_ = false;
    int numdelay_ = 0;
    std::vector<double> scale_;
};

} // namespace smf::bench
