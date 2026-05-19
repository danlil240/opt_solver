#include "smf/utils/memory.hpp"

#include <cstdlib>    // std::free
#include <cstring>    // std::memcpy
#include <algorithm>  // std::max
#include <cassert>
#include <new>        // std::bad_alloc

namespace smf {

namespace {
// Round x up to the nearest multiple of align (align must be power of 2).
inline std::size_t align_up(std::size_t x, std::size_t align) noexcept {
    return (x + align - 1u) & ~(align - 1u);
}
} // anonymous namespace

AlignedArena::AlignedArena(std::size_t capacity_bytes) {
    // Round capacity up to a multiple of 64 bytes.
    capacity_ = align_up(capacity_bytes == 0 ? 64u : capacity_bytes, 64u);
    void* ptr = nullptr;
    if (::posix_memalign(&ptr, 64u, capacity_) != 0) {
        throw std::bad_alloc{};
    }
    buf_ = static_cast<uint8_t*>(ptr);
}

AlignedArena::~AlignedArena() {
    for (void* p : old_bufs_) std::free(p);
    std::free(buf_);
}

void* AlignedArena::allocate(std::size_t bytes, std::size_t align) {
    // Round current offset up to alignment boundary.
    std::size_t aligned_offset = align_up(offset_, align);
    if (aligned_offset + bytes > capacity_) {
        // Try to grow and retry once.
        grow(aligned_offset + bytes);
        aligned_offset = align_up(offset_, align);
    }
    offset_ = aligned_offset + bytes;
    if (offset_ > peak_bytes_) {
        peak_bytes_ = offset_;
    }
    return buf_ + aligned_offset;
}

void AlignedArena::reset_to(Marker m) noexcept {
    // Marker must be a previously saved offset.
    offset_ = m;
}

void AlignedArena::clear() noexcept {
    for (void* p : old_bufs_) { std::free(p); }
    old_bufs_.clear();
    offset_ = 0;
}

void AlignedArena::grow(std::size_t min_new_capacity) {
    std::size_t new_cap = std::max(min_new_capacity,
                                   capacity_ * 3u / 2u);
    new_cap = align_up(new_cap, 64u);

    void* new_ptr = nullptr;
    if (::posix_memalign(&new_ptr, 64u, new_cap) != 0) {
        throw std::bad_alloc{};
    }
    // Copy existing data.
    if (offset_ > 0) {
        std::memcpy(new_ptr, buf_, offset_);
    }
    old_bufs_.push_back(buf_);  // keep old buf alive
    buf_       = static_cast<uint8_t*>(new_ptr);
    capacity_  = new_cap;
    ++grow_count_;
}

} // namespace smf
