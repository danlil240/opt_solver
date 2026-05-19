#pragma once
#include <cstddef>
#include <cstring>
#include <cassert>

namespace smf {

/// Two-buffer alternating stack for contribution (Schur-complement) blocks.
///
/// The "active" buffer receives alloc()/free_top() calls.  flip() switches
/// to the other buffer (resetting its top to 0) so a parent front can keep
/// reading the old buffer while new contributions are pushed on the fresh one.
class FactorStack {
public:
    /// Construct with a per-buffer initial capacity (bytes).  Default 16 MiB.
    explicit FactorStack(std::size_t initial_bytes = 1u << 24);
    ~FactorStack() = default;

    FactorStack(const FactorStack&)            = delete;
    FactorStack& operator=(const FactorStack&) = delete;
    FactorStack(FactorStack&&)                 = delete;
    FactorStack& operator=(FactorStack&&)      = delete;

    /// Allocate n_doubles on the active buffer.
    /// Grows the active buffer by 1.5× (preserving live data) when needed.
    double* alloc(std::size_t n_doubles);

    /// Release the top n_doubles from the active buffer (LIFO).
    void free_top(std::size_t n_doubles);

    /// Switch active buffer; the new active starts at top=0.
    /// The previous buffer remains intact for external readers.
    void flip();

    /// Lifetime high-water mark across both buffers (bytes).
    std::size_t high_water_bytes() const noexcept { return hwm_total_; }

    /// Number of times any buffer was grown.
    int growth_count() const noexcept { return growth_count_; }

    /// Total bytes currently live (top_bytes of both buffers combined).
    std::size_t live_bytes() const noexcept {
        return buf_[0].top_bytes + buf_[1].top_bytes;
    }

private:
    struct Buffer {
        double*     data          = nullptr;
        std::size_t capacity_bytes = 0;
        std::size_t top_bytes      = 0;

        Buffer() = default;
        ~Buffer() { delete[] data; }

        Buffer(const Buffer&)            = delete;
        Buffer& operator=(const Buffer&) = delete;

        void init(std::size_t bytes);
        /// Grow so that capacity_bytes >= min_bytes, preserving live data.
        void grow(std::size_t min_bytes);
    };

    Buffer      buf_[2];
    int         active_       = 0;
    std::size_t hwm_total_    = 0;
    int         growth_count_ = 0;
};

} // namespace smf
