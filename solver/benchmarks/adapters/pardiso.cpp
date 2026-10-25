#include "adapter.hpp"

namespace smf::bench {

namespace {

class PardisoSolverAdapter final : public SolverAdapter {
public:
    explicit PardisoSolverAdapter(std::string variant)
        : variant_(std::move(variant)) {}

    AdapterIdentity identity() const override {
        AdapterIdentity id;
        id.solver_id = variant_;
        id.version = "not-linked";
        id.library_path = "unavailable";
        id.integer_abi = "int32";
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
        return AdapterStats{Outcome::Unavailable, variant_ + " not installed"};
    }

private:
    std::string variant_;
};

} // namespace

std::unique_ptr<SolverAdapter> make_pardiso_adapter(const std::string &variant) {
    return std::make_unique<PardisoSolverAdapter>(variant);
}

} // namespace smf::bench
