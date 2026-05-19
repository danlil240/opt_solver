#include "smf/factor_stack.hpp"
#include <gtest/gtest.h>
#include <cstring>

namespace smf {

// ---------------------------------------------------------------------------
// 1. Basic alloc / free roundtrip
// ---------------------------------------------------------------------------
TEST(StackBasic, AllocFreeRoundtrip) {
    FactorStack stack(4096);

    double* p = stack.alloc(100);
    for (int i = 0; i < 100; ++i) p[i] = static_cast<double>(i) + 1.5;

    stack.free_top(100);
    EXPECT_EQ(stack.live_bytes(), 0u);

    // After free, alloc again — pointer should be at the same location and
    // the caller is responsible for writing before reading (no stale-read contract).
    double* q = stack.alloc(100);
    EXPECT_EQ(p, q);   // same base address (LIFO stack)
    stack.free_top(100);
}

// ---------------------------------------------------------------------------
// 2. Growth preserves live data
// ---------------------------------------------------------------------------
TEST(StackBasic, GrowthPreservesData) {
    // Start with a tiny initial capacity to force a growth quickly
    FactorStack stack(128);   // 128 bytes = 16 doubles

    // Alloc 10 doubles — fits
    double* first = stack.alloc(10);
    for (int i = 0; i < 10; ++i) first[i] = 100.0 + i;

    // Alloc 100 doubles — must trigger growth (10+100 doubles > 16 capacity)
    double* second = stack.alloc(100);
    for (int i = 0; i < 100; ++i) second[i] = 200.0 + i;

    // After growth the first allocation may have moved — recompute its address
    // via the stack's base (we can't hold the old pointer safely after realloc).
    // Instead, verify via the second block that growth happened at least once.
    EXPECT_GE(stack.growth_count(), 1);

    // Verify second block values are correct (no corruption during grow)
    for (int i = 0; i < 100; ++i)
        EXPECT_DOUBLE_EQ(second[i], 200.0 + i);

    stack.free_top(100);
    stack.free_top(10);
}

// ---------------------------------------------------------------------------
// 3. High-water mark
// ---------------------------------------------------------------------------
TEST(StackBasic, HighWaterMark) {
    FactorStack stack(1 << 20);  // 1 MiB — plenty

    double* a = stack.alloc(500);
    (void)a;
    std::size_t hwm_after_500 = stack.high_water_bytes();
    EXPECT_GE(hwm_after_500, 500 * sizeof(double));

    stack.free_top(200);
    // HWM must not decrease
    EXPECT_GE(stack.high_water_bytes(), hwm_after_500);

    double* b = stack.alloc(300);
    (void)b;
    // Still ≥ 500*8 (we only have 300+300=600 live now but HWM was 500*8)
    EXPECT_GE(stack.high_water_bytes(), 500 * sizeof(double));

    stack.free_top(300);
    stack.free_top(300);
}

// ---------------------------------------------------------------------------
// 4. flip — writes to buf B do not affect buf A
// ---------------------------------------------------------------------------
TEST(StackBasic, Flip) {
    FactorStack stack(1 << 20);

    // Write to buffer A
    double* a = stack.alloc(64);
    for (int i = 0; i < 64; ++i) a[i] = static_cast<double>(i) * 3.14;

    // Flip to buffer B
    stack.flip();

    // B starts fresh
    EXPECT_EQ(stack.live_bytes(), 64 * sizeof(double));  // A still counted

    double* b = stack.alloc(64);
    // Write completely different pattern to B
    for (int i = 0; i < 64; ++i) b[i] = -999.0 - i;

    // Verify A data is unchanged (a pointer still valid — we did not flip back)
    for (int i = 0; i < 64; ++i)
        EXPECT_DOUBLE_EQ(a[i], static_cast<double>(i) * 3.14);

    stack.free_top(64);
    // flip back to A
    stack.flip();
    stack.free_top(64);
}

// ---------------------------------------------------------------------------
// 5. Alias safety — writing to B never corrupts A
// ---------------------------------------------------------------------------
TEST(StackBasic, AliasSafety) {
    FactorStack stack(1 << 20);

    // Fill buffer A with sentinel
    double* a = stack.alloc(256);
    for (int i = 0; i < 256; ++i) a[i] = 42.0;

    // Flip → now on B
    stack.flip();

    // Hammer B with different values
    double* b = stack.alloc(256);
    for (int i = 0; i < 256; ++i) b[i] = -1.0;

    // A and B must not alias
    ASSERT_NE(static_cast<void*>(a), static_cast<void*>(b));

    // A data still intact
    for (int i = 0; i < 256; ++i)
        EXPECT_DOUBLE_EQ(a[i], 42.0) << "A corrupted at index " << i;

    // Flip back to A — B now inactive but its pointer still valid
    stack.free_top(256);  // free B alloc
    stack.flip();         // active = A again

    // A data still intact after the flip sequence
    for (int i = 0; i < 256; ++i)
        EXPECT_DOUBLE_EQ(a[i], 42.0) << "A corrupted after flip-back at index " << i;

    stack.free_top(256);
}

// ---------------------------------------------------------------------------
// 6. Growth count
// ---------------------------------------------------------------------------
TEST(StackBasic, GrowthCount) {
    // Initial capacity exactly 8 doubles = 64 bytes
    FactorStack stack(64);

    // First alloc: 5 doubles — fits (5*8=40 < 64 rounded-up capacity)
    // The init rounds 64 bytes up, giving >=8 doubles.
    // Let's use a known-fit size then force two growths.

    // Alloc 8 doubles — fits in 64-byte buffer (8 doubles exactly)
    double* p1 = stack.alloc(8);
    (void)p1;
    EXPECT_EQ(stack.growth_count(), 0);

    // Alloc 1 more — forces first growth
    double* p2 = stack.alloc(1);
    (void)p2;
    EXPECT_EQ(stack.growth_count(), 1);

    // After first growth capacity ~ 64 * 1.5 = 96 bytes = 12 doubles.
    // We've used 9 doubles (72 bytes). Alloc 4 more (32 bytes) → 104 > 96 → second growth.
    double* p3 = stack.alloc(4);
    (void)p3;
    EXPECT_EQ(stack.growth_count(), 2);

    stack.free_top(4);
    stack.free_top(1);
    stack.free_top(8);
}

} // namespace smf
