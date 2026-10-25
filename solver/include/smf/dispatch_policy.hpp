#pragma once

#include "smf/analysis.hpp"
#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"

#include <string>

namespace smf {

enum class NativePath {
    TinyDense = 0,
    BandedSPD,
    SparseSPD,
    GeneralIndefinite,
    CertifiedQuasiDefinite
};

struct DispatchDecision {
    NativePath path = NativePath::GeneralIndefinite;
    std::string reason;
    int lower_bandwidth = 0;
    double density = 0.0;
};

const char *native_path_to_string(NativePath path) noexcept;

DispatchDecision select_native_path(const AnalysisKeep &ak,
                                    const Control &ctrl,
                                    int nrhs = 1,
                                    int expected_refactors = 1);

bool certify_quasi_definite(const CscLower &A, double min_abs_diag = 1e-12) noexcept;

} // namespace smf
