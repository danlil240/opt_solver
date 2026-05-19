#include <gtest/gtest.h>
#include "smf/utils/memory.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Test 1: AlignmentCheck
// ---------------------------------------------------------------------------
TEST(Arena, AlignmentCheck) {
    smf::AlignedArena arena(64u * 1024u);

    // Vary sizes between 1 and 256 bytes.
    for (int i = 0; i < 200; ++i) {
        std::size_t sz = static_cast<std::size_t>((i % 256) + 1);
        void* p = arena.allocate(sz);
        ASSERT_NE(p, nullptr);
        auto addr = reinterpret_cast<std::uintptr_t>(p);
        EXPECT_EQ(addr % 64u, 0u) << "Allocation " << i << " not 64-byte aligned";
    }
}

// ---------------------------------------------------------------------------
// Test 2: GrowthTracking
// ---------------------------------------------------------------------------
TEST(Arena, GrowthTracking) {
    constexpr std::size_t INITIAL = 256u;
    smf::AlignedArena arena(INITIAL);

    // Write to first allocation, then grow past initial capacity.
    void* first = arena.allocate(64u);
    ASSERT_NE(first, nullptr);
    // Write a sentinel value.
    *static_cast<uint64_t*>(first) = 0xDEADBEEFCAFEBABEULL;

    // Force growth: allocate more than initial capacity.
    for (int i = 0; i < 10; ++i) {
        void* p = arena.allocate(64u);
        EXPECT_NE(p, nullptr);
    }

    EXPECT_GE(arena.grow_count(), 1) << "Arena should have grown at least once";
    // Data at first allocation should be preserved after grow.
    EXPECT_EQ(*static_cast<uint64_t*>(first), 0xDEADBEEFCAFEBABEULL);
}

// ---------------------------------------------------------------------------
// Test 3: MarkerReset
// ---------------------------------------------------------------------------
TEST(Arena, MarkerReset) {
    smf::AlignedArena arena(4096u);

    // Allocate some memory.
    void* p1 = arena.allocate(128u);
    ASSERT_NE(p1, nullptr);

    // Save marker.
    auto mark = arena.save();
    std::size_t used_at_mark = arena.used();

    // Allocate more.
    void* p2 = arena.allocate(128u);
    (void)p2;
    EXPECT_GT(arena.used(), used_at_mark);

    // Reset to marker.
    arena.reset_to(mark);
    EXPECT_EQ(arena.used(), used_at_mark);

    // Allocate again from same position — should get same pointer as p2.
    void* p3 = arena.allocate(128u);
    EXPECT_EQ(p3, p2) << "After reset_to, allocation should restart from marker";
}

// ---------------------------------------------------------------------------
// Test 4: ClearReset
// ---------------------------------------------------------------------------
TEST(Arena, ClearReset) {
    smf::AlignedArena arena(1024u);

    void* p = arena.allocate(256u);
    EXPECT_NE(p, nullptr);
    EXPECT_GT(arena.used(), 0u);

    arena.clear();
    EXPECT_EQ(arena.save(), 0u) << "After clear(), save() should return 0";
    EXPECT_EQ(arena.used(), 0u);
}

// ---------------------------------------------------------------------------
// Test 5: PeakTracking
// ---------------------------------------------------------------------------
TEST(Arena, PeakTracking) {
    smf::AlignedArena arena(4096u);

    std::size_t last_peak = arena.peak_bytes();
    EXPECT_EQ(last_peak, 0u);

    // Each allocation should increase peak.
    for (int i = 0; i < 5; ++i) {
        arena.allocate(64u);
        EXPECT_GE(arena.peak_bytes(), last_peak);
        last_peak = arena.peak_bytes();
    }

    // After reset_to(), peak should not decrease.
    auto mark = arena.save();
    arena.allocate(256u);
    std::size_t peak_before_reset = arena.peak_bytes();

    arena.reset_to(mark);
    EXPECT_EQ(arena.peak_bytes(), peak_before_reset)
        << "peak_bytes() must not decrease after reset_to()";

    // clear() also must not decrease peak.
    arena.clear();
    EXPECT_EQ(arena.peak_bytes(), peak_before_reset)
        << "peak_bytes() must not decrease after clear()";
}

// ---------------------------------------------------------------------------
// Test 6: AliasingSafety
// ---------------------------------------------------------------------------
TEST(Arena, AliasingSafety) {
    smf::AlignedArena arena(4096u);

    // Allocate first block and write known pattern.
    auto* p1 = static_cast<uint8_t*>(arena.allocate(128u));
    ASSERT_NE(p1, nullptr);
    std::memset(p1, 0xAB, 128u);

    // Allocate second block.
    auto* p2 = static_cast<uint8_t*>(arena.allocate(128u));
    ASSERT_NE(p2, nullptr);
    std::memset(p2, 0xCD, 128u);

    // Verify first block is unchanged.
    for (std::size_t i = 0; i < 128u; ++i) {
        ASSERT_EQ(p1[i], static_cast<uint8_t>(0xAB))
            << "Byte " << i << " of first allocation corrupted";
    }

    // Pointers must not overlap.
    EXPECT_NE(static_cast<void*>(p1), static_cast<void*>(p2));
    // They should be at least 128 bytes apart.
    auto diff = reinterpret_cast<std::uintptr_t>(p2) -
                reinterpret_cast<std::uintptr_t>(p1);
    EXPECT_GE(diff, 128u);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
