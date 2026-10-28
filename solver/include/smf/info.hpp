#pragma once
#include "smf/types.hpp"
#include "smf/error.hpp"

namespace smf {

/// Wall time of each analyse phase (seconds). The phases partition
/// Info::analyse_seconds: matching covers the maximum-product matching
/// ordering/scaling work (including the auto-matching trial), ordering the
/// rest of the ordering choice (AMD/METIS/identity), etree_counts the
/// elimination tree and column counts, amalgamate the postorder relabelling
/// and relaxed amalgamation, assembly_tree the permuted pattern plus
/// assembly tree, keep_build the AnalysisKeep assembly.
struct AnalysePhaseSeconds {
    double check         = 0.0;
    double ordering      = 0.0;
    double matching      = 0.0;
    double etree_counts  = 0.0;
    double amalgamate    = 0.0;
    double assembly_tree = 0.0;
    double keep_build    = 0.0;
};

struct Info {
    // Top-level status
    ErrorCode    status              = ErrorCode::Success;
    FactorStatus factor_status       = FactorStatus::Success;

    // Matrix input diagnostics
    int          matrix_duplicates   = 0;
    int          matrix_missing_diag = 0;
    int          matrix_out_of_range = 0;

    // Rank information
    int          structural_rank     = 0;
    int          numerical_rank      = 0;

    // Ordering used (value matches OrderingMethod enum)
    int          ordering_used       = 0;
    /// OrderingMethod::UserSupplied: user pairs that could not be kept in
    /// one supernode because the second is not the etree parent of the first.
    int          pairs_split         = 0;
    /// 1 when this factor() re-analysed with the matching ordering
    /// (Control::matching_trigger = RealizedDelays: the first factorization
    /// delayed more than 5% of n pivots), else 0.
    int          matching_reanalyses = 0;
    /// How analyse() decided the auto matching trial.
    MatchingTrial matching_trial     = MatchingTrial::None;
    /// The trial's predicted factor flops (sum of squared column counts) of
    /// the matching ordering over the base (AMD/METIS) ordering's, built
    /// from the matching's initial phase (only when the searches exceeded
    /// the budget) and from the exact matching. 0 when not computed.
    /// matching_initial_ratio is set as soon as the initial ordering is
    /// compared, so it stays set when the resumed exact matching then
    /// fails (matching_trial None, matching_exact_ratio 0); a trial
    /// rejected on the initial ordering (RejectedInitial) has no exact
    /// ratio, and one whose searches stayed within the budget (Exact) no
    /// initial ratio.
    double       matching_initial_ratio = 0.0;
    double       matching_exact_ratio   = 0.0;

    // Inertia
    int          num_negative        = 0;   // negative eigenvalues
    int          num_zero            = 0;   // zero eigenvalues
    int          num_positive        = 0;   // positive eigenvalues

    // Factorization counts
    LongInt      actual_factor_entries  = 0;
    LongInt      predicted_factor_entries = 0;
    double       predicted_flops        = 0.0;
    double       actual_flops           = 0.0;
    int          delayed_pivots         = 0;
    int          static_pivots          = 0;  ///< pivots perturbed by static pivoting
    /// False iff this factorization perturbed a pivot by static pivoting
    /// (static_pivots > 0: Control::delay_budget, static_pivot_tol,
    /// fallback_static_pivot_tol): the inertia counts are then those of a
    /// perturbed matrix, not necessarily of A.
    bool         inertia_exact          = true;
    /// Control::delay_budget: fronts factored a second time, with the static
    /// path, because their delays would have taken the count past the budget.
    int          delay_budget_retries   = 0;
    /// Fully summed columns the APTP front kernel failed (summed over
    /// fronts, before the TPP fallback; Control::front_kernel, and the tile
    /// fronts of Control::top_tiles whatever the kernel). Those the
    /// fallback could not pivot either are also in delayed_pivots.
    int          aptp_failed_columns    = 0;
    int          parallel_subtrees      = 0;  ///< subtrees factored concurrently (0 = serial)
    /// Fronts factored as 2D tile tasks at the top of the tree
    /// (Control::top_tiles).
    int          top_tile_fronts        = 0;
    /// Contribution blocks of small fronts handed to their parent in a
    /// small-front buffer instead of the stack (Control::small_front_kernels).
    int          small_front_handovers  = 0;
    /// After FactorStatus::InertiaExceeded: predicted flops (sum of p f^2,
    /// as predicted_flops) of the fronts that were not factored.
    double       aborted_flops_skipped  = 0.0;

    // Assembly tree stats
    int          max_front_size         = 0;
    int          max_supernode_size     = 0;
    int          max_tree_depth         = 0;

    // Timings (seconds)
    double       analyse_seconds        = 0.0;
    AnalysePhaseSeconds analyse_phases;
    double       factor_seconds         = 0.0;
    /// Control::cache_inverse_diag: fronts whose L11^-1 the last factor()
    /// cached, and the time that took (included in factor_seconds).
    int          inverse_diag_fronts    = 0;
    double       inverse_diag_seconds   = 0.0;
    double       solve_seconds          = 0.0;
    double       refinement_seconds     = 0.0;
    int          refinement_iterations  = 0;

    // Arena memory diagnostics
    long         arena_peak_bytes       = 0;
    int          arena_growths          = 0;
};

} // namespace smf
