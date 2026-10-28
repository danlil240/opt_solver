#pragma once

#include "smf/control.hpp"
#include "smf/csc_matrix.hpp"
#include "smf/dense_indef.hpp"
#include "smf/error.hpp"
#include "smf/factor_posdef.hpp"
#include "smf/types.hpp"

#include <vector>

namespace smf {

struct FrontDiagBlock {
    int8_t tag = 0;   // 1 = 1x1, 2 = 2x2
    Int row0 = 0;
    Int row1 = 0;
    double d00 = 0.0;
    double d10 = 0.0;
    double d11 = 0.0;
};

class NumericFront {
public:
    NumericFront() = default;
    NumericFront(std::vector<Int> row_ids, int num_eligible);

    int num_rows() const noexcept { return static_cast<int>(row_ids_.size()); }
    int num_eligible() const noexcept { return num_eligible_; }

    std::vector<Int>& row_ids() noexcept { return row_ids_; }
    const std::vector<Int>& row_ids() const noexcept { return row_ids_; }

    double* data() noexcept { return data_.data(); }
    const double* data() const noexcept { return data_.data(); }

    double& at(int row, int col) noexcept {
        const int f = num_rows();
        return data_[static_cast<std::size_t>(col) * static_cast<std::size_t>(f) +
                     static_cast<std::size_t>(row)];
    }
    const double& at(int row, int col) const noexcept {
        const int f = num_rows();
        return data_[static_cast<std::size_t>(col) * static_cast<std::size_t>(f) +
                     static_cast<std::size_t>(row)];
    }

    void sym_swap(int p, int q) noexcept;

    void scatter_original_columns(const CscLower& Ap, Int col_start, Int col_end);
    void assemble_contribution(const std::vector<Int>& child_rows,
                               const std::vector<double>& child_contrib);

private:
    std::vector<Int> row_ids_;
    int num_eligible_ = 0;
    std::vector<double> data_;
};

struct FrontResult {
    ErrorCode status = ErrorCode::Success;
    std::vector<Int> accepted_row_ids;
    std::vector<Int> l_row_ids;
    int num_accepted = 0;
    int front_size = 0;
    std::vector<int8_t> pivot_tags;
    std::vector<double> l_panel;
    std::vector<FrontDiagBlock> d_blocks;
    std::vector<Int> delayed_col_ids;
    std::vector<Int> contrib_row_ids;
    std::vector<double> contrib;
    InertiaCounts inertia{};
    int delayed_count = 0;
    double flops = 0.0;
};

FrontResult factor_front_reference(NumericFront& front,
                                   const Control& ctrl,
                                   double effective_small = -1.0,
                                   bool is_root = false);

} // namespace smf
