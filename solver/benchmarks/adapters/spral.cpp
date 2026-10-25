#include "adapter.hpp"

namespace smf::bench {

namespace {

class SpralSsidsAdapter final : public SolverAdapter {
public:
    AdapterIdentity identity() const override {
        AdapterIdentity id;
        id.solver_id = "ssids";
        id.version = "not-linked";
        id.library_path = "unavailable";
        id.integer_abi = "int32_offsets64";
        id.available = false;
        id.supports_spd = true;
        id.supports_indefinite = true;
        return id;
    }
    void configure(const Control &, const std::string &) override {}
    Outcome analyse(const CscLower &) override { return Outcome::Unavailable; }
    Outcome factor(std::span<const double>) override { return Outcome::Unavailable; }
    Outcome solve(std::span<double>, int) override { return Outcome::Unavailable; }
    AdapterStats stats() const override {
        return AdapterStats{Outcome::Unavailable, "SPRAL SSIDS not installed"};
    }
};

} // namespace

std::unique_ptr<SolverAdapter> make_spral_ssids_adapter() {
    return std::make_unique<SpralSsidsAdapter>();
}

} // namespace smf::bench
