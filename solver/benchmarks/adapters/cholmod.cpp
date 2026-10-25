#include "adapter.hpp"

namespace smf::bench {

namespace {

class CholmodSolverAdapter final : public SolverAdapter {
public:
    AdapterIdentity identity() const override {
        AdapterIdentity id;
        id.solver_id = "cholmod";
        id.version = "not-linked";
        id.library_path = "unavailable";
        id.integer_abi = "int32";
        id.available = false;
        id.supports_spd = true;
        id.supports_indefinite = false;
        return id;
    }
    void configure(const Control &, const std::string &) override {}
    Outcome analyse(const CscLower &) override { return Outcome::Unavailable; }
    Outcome factor(std::span<const double>) override { return Outcome::Unavailable; }
    Outcome solve(std::span<double>, int) override { return Outcome::Unavailable; }
    AdapterStats stats() const override {
        return AdapterStats{Outcome::Unavailable, "CHOLMOD not installed"};
    }
};

} // namespace

std::unique_ptr<SolverAdapter> make_cholmod_adapter() {
    return std::make_unique<CholmodSolverAdapter>();
}

} // namespace smf::bench
