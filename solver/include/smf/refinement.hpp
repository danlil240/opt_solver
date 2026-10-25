#pragma once

#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/error.hpp"
#include "smf/factor_posdef.hpp"
#include "smf/workspace.hpp"

#include <span>

namespace smf {

struct RefinementResult {
    ErrorCode status = ErrorCode::Success;
    int iterations = 0;
    double eta_inf = 0.0;
    double omega = 0.0;
};

RefinementResult refine(const CscLower &original,
                        const FactorKeep &fkeep,
                        const Control &ctrl,
                        std::span<const double> b,
                        std::span<double> x,
                        int nrhs,
                        SolveWorkspace &ws);

} // namespace smf
