#pragma once

#include "../common/result.hpp"
#include "../common/validation.hpp"
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"

#include <memory>
#include <span>
#include <string>
#include <vector>

namespace smf::bench {

struct AdapterIdentity {
    std::string solver_id;
    std::string version;
    std::string library_path;
    std::string integer_abi = "int32";
    bool available = false;
    bool supports_spd = true;
    bool supports_indefinite = true;
    bool is_external_memory = false;
};

struct AdapterStats {
    Outcome outcome = Outcome::Failed;
    std::string reason;
    double analyse_ms = 0.0;
    double factor_ms = 0.0;
    double solve_ms = 0.0;
    int inertia_pos = 0;
    int inertia_neg = 0;
    int inertia_zero = 0;
    int delayed_pivots = 0;
    int64_t factor_entries = 0;
    double flops = 0.0;
};

class SolverAdapter {
public:
    virtual ~SolverAdapter() = default;
    virtual AdapterIdentity identity() const = 0;
    virtual void configure(const Control &ctrl, const std::string &track = "recommended") = 0;
    virtual Outcome analyse(const CscLower &A) = 0;
    virtual Outcome factor(std::span<const double> values) = 0;
    virtual Outcome solve(std::span<double> b_and_x, int nrhs = 1) = 0;
    virtual AdapterStats stats() const = 0;
};

std::unique_ptr<SolverAdapter> make_smf_adapter();
std::unique_ptr<SolverAdapter> make_hsl_adapter(const std::string &hsl_id);
std::unique_ptr<SolverAdapter> make_spral_ssids_adapter();
std::unique_ptr<SolverAdapter> make_mumps_adapter();
std::unique_ptr<SolverAdapter> make_pardiso_adapter(const std::string &variant = "mkl_pardiso");
std::unique_ptr<SolverAdapter> make_cholmod_adapter();
std::unique_ptr<SolverAdapter> make_solver_adapter(const std::string &solver_id);
std::vector<std::string> all_registered_adapter_ids();

} // namespace smf::bench
