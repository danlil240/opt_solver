#pragma once

#include "smf/control.hpp"
#include "smf/numeric_front.hpp"
#include "smf/workspace.hpp"

namespace smf {

/// Blocked indefinite frontal factorization using threshold pivoting,
/// panel W = L21 * D formation, and tiled BLAS-3 Schur complement updates
/// A22 -= L21 * Wᵀ.
FrontResult factor_front_blocked(NumericFront &front,
                                 const Control &ctrl,
                                 NumericWorkspace &ws,
                                 double effective_small = -1.0,
                                 bool is_root = false);

} // namespace smf
