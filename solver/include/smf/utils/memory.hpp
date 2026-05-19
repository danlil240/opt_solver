#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace smf {

/// Arena allocator for hot-path numeric memory.
/// - All allocations are aligned to at least `align` bytes (default 64).
/// - Not thread-safe.
/// - Supports O(1) reset to a saved marker.
class AlignedArena {
public:
    explicit AlignedArena(std::size_t capacity_bytes);
    ~AlignedArena();

    AlignedArena(const AlignedArena&)            = delete;
    AlignedArena& operator=(const AlignedArena&) = delete;

    /// Allocate `bytes` with alignment `align` (must be power of 2, default 64).
    /// Returns nullptr if capacity is exceeded (grows the arena by 1.5× and retries once).
    void* allocate(std::size_t bytes, std::size_t align = 64);

    /// Marker (byte offset from arena start).
    using Marker = std::size_t;

    /// Save current allocation position.
    Marker save() const noexcept { return offset_; }

    /// Reset allocation pointer to a previous marker (does not zero memory).
    void reset_to(Marker m) noexcept;

    /// Reset the entire arena (all allocations invalidated).
    void clear() noexcept;

    // Diagnostics
    std::size_t peak_bytes()   const noexcept { return peak_bytes_; }
    int         grow_count()   const noexcept { return grow_count_; }
    std::size_t capacity()     const noexcept { return capacity_; }
    std::size_t used()         const noexcept { return offset_; }

private:
    void grow(std::size_t min_new_capacity);

    uint8_t*    buf_          = nullptr;
    std::size_t capacity_     = 0;
    std::size_t offset_       = 0;
    std::size_t peak_bytes_   = 0;
    int         grow_count_   = 0;
    std::vector<void*> old_bufs_;
};

} // namespace smf
