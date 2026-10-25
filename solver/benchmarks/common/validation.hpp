#pragma once

#include "result.hpp"
#if __has_include("smf/csc_matrix.hpp")
#include "smf/csc_matrix.hpp"
#else
#include "../../include/smf/csc_matrix.hpp"
#endif

#include <span>
#include <vector>

namespace smf::bench {

inline constexpr double kMaxQualifiedEtaInf = 1e-10;
inline constexpr double kMaxQualifiedOmega = 1e-8;

Validation validate(const CscLower& A,
                    std::span<const double> b,
                    std::span<const double> x,
                    int nrhs = 1);

std::vector<Validation> validate_per_rhs(const CscLower& A,
                                         std::span<const double> b,
                                         std::span<const double> x,
                                         int nrhs = 1);

bool is_qualified_validation(const Validation& v,
                             double max_eta_inf = kMaxQualifiedEtaInf,
                             double max_omega = kMaxQualifiedOmega) noexcept;

Outcome qualify_outcome(Outcome raw_outcome,
                        const Validation& v,
                        double max_eta_inf = kMaxQualifiedEtaInf,
                        double max_omega = kMaxQualifiedOmega) noexcept;

} // namespace smf::bench
