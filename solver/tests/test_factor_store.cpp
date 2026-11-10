// FactorStore: per-thread slices of one reserved address range (the factor
// written in place by the tree-parallel engine, Control::factor_in_place).

#include "smf/factor_store.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <utility>

using namespace smf;

namespace {
constexpr std::size_t kMiB = std::size_t{1} << 20;
}

TEST(FactorStore, SlicesAreDisjointAndAddressedFromBase) {
    FactorStore<double> s;
    ASSERT_TRUE(s.reserve_slices(4, std::size_t{1} << 20));
    ASSERT_EQ(s.slices(), 4);
    for (int t = 0; t < 4; ++t) {
        ASSERT_TRUE(s.slice(t).grow(1000 + 10 * static_cast<std::size_t>(t)));
        for (std::size_t k = 0; k < s.slice(t).size(); ++k) s.slice(t).data()[k] = t + 1;
    }
    EXPECT_TRUE(s.all_attached());
    for (int t = 0; t < 4; ++t) {
        EXPECT_EQ(s.slice(t).data(), s.data() + s.slice_offset(t));
        for (std::size_t k = 0; k < s.slice(t).size(); ++k)
            ASSERT_EQ(s.data()[s.slice_offset(t) + k], t + 1) << t << " " << k;
        if (t > 0) EXPECT_GE(s.slice_offset(t), s.slice_offset(t - 1) + (std::size_t{1} << 20));
    }
    EXPECT_EQ(s.used(), 4000u + 60u);
    EXPECT_EQ(s.size(), s.slice_offset(3) + 1030);
    EXPECT_FALSE(s.empty());
}

TEST(FactorStore, GrowBeyondReservationDetachesAndRegathers) {
    FactorStore<double> s;
    ASSERT_TRUE(s.reserve_slices(2, 1000));
    ASSERT_TRUE(s.slice(0).grow(600));
    for (std::size_t k = 0; k < 600; ++k) s.slice(0).data()[k] = static_cast<double>(k);
    ASSERT_TRUE(s.slice(1).grow(10));
    for (std::size_t k = 0; k < 10; ++k) s.slice(1).data()[k] = -1.0 - static_cast<double>(k);
    ASSERT_TRUE(s.slice(0).grow(5000));
    EXPECT_TRUE(s.slice(0).detached());
    EXPECT_FALSE(s.all_attached());
    for (std::size_t k = 0; k < 600; ++k) ASSERT_EQ(s.slice(0).data()[k], static_cast<double>(k));
    for (std::size_t k = 600; k < 5000; ++k) s.slice(0).data()[k] = static_cast<double>(k);
    s.regather();
    EXPECT_EQ(s.slices(), 1);
    EXPECT_EQ(s.slice_offset(0), 0u);
    EXPECT_TRUE(s.all_attached());
    ASSERT_EQ(s.used(), 5010u);
    for (std::size_t k = 0; k < 5000; ++k) ASSERT_EQ(s.data()[k], static_cast<double>(k));
    for (std::size_t k = 0; k < 10; ++k) ASSERT_EQ(s.data()[5000 + k], -1.0 - static_cast<double>(k));
}

TEST(FactorStore, ClearKeepsReservationAndTrimReleases) {
    FactorStore<double> s;
    ASSERT_TRUE(s.reserve_slices(1, 256 * kMiB / sizeof(double)));
    ASSERT_TRUE(s.slice(0).grow(64 * kMiB / sizeof(double)));
    std::memset(s.data(), 1, 64 * kMiB);
    const double *base = s.data();
    EXPECT_GE(s.committed_bytes(), 64 * kMiB);
    s.clear();
    EXPECT_EQ(s.used(), 0u);
    EXPECT_TRUE(s.empty());
    ASSERT_TRUE(s.reserve_slices(1, 256 * kMiB / sizeof(double)));  // unchanged: kept
    EXPECT_EQ(s.data(), base);
    ASSERT_TRUE(s.slice(0).grow(kMiB / sizeof(double)));
    s.trim();
    EXPECT_LE(s.committed_bytes(), 16 * kMiB + 2 * kMiB);
    std::memset(s.data(), 2, kMiB);  // still writable
}

TEST(FactorStore, MoveAndCopyKeepOffsets) {
    FactorStore<int> s;
    ASSERT_TRUE(s.reserve_slices(3, 4096));
    for (int t = 0; t < 3; ++t) {
        ASSERT_TRUE(s.slice(t).grow(100));
        for (int k = 0; k < 100; ++k) s.slice(t).data()[k] = 1000 * t + k;
    }
    const std::size_t off2 = s.slice_offset(2);
    FactorStore<int> c(s);
    EXPECT_EQ(c.slice_offset(2), off2);
    EXPECT_EQ(c.data()[off2 + 7], 2007);
    FactorStore<int> m(std::move(s));
    EXPECT_EQ(m.slice_offset(2), off2);
    EXPECT_EQ(m.data()[off2 + 99], 2099);
    FactorStore<int> a;
    a = c;
    EXPECT_EQ(a.data()[a.slice_offset(1) + 5], 1005);
    std::size_t ranges = 0, total = 0;
    a.for_each_used([&](const int *, std::size_t n) { ++ranges; total += n; });
    EXPECT_EQ(ranges, 3u);
    EXPECT_EQ(total, 300u);
}

TEST(FactorStore, ZeroAndTinySlices) {
    FactorStore<double> s;
    ASSERT_TRUE(s.reserve_slices(8, 0));
    EXPECT_TRUE(s.slice(3).grow(0));
    EXPECT_TRUE(s.empty());
    EXPECT_EQ(s.size(), 0u);
    ASSERT_TRUE(s.reserve_slices(8, 1));
    ASSERT_TRUE(s.slice(7).grow(1));
    s.slice(7).data()[0] = 3.5;
    EXPECT_EQ(s.data()[s.slice_offset(7)], 3.5);
    FactorStore<double> empty;
    EXPECT_TRUE(empty.empty());
    EXPECT_EQ(empty.size(), 0u);
    s.assign(5, 2.0);
    EXPECT_EQ(s.slices(), 1);
    EXPECT_EQ(s.size(), 5u);
    EXPECT_EQ(s.data()[4], 2.0);
}

TEST(FactorStore, ReservationFailureFallsBackToDetached) {
#ifdef _WIN32
    _putenv_s("SMF_TEST_STORE_FAIL_RESERVE", "1");
#else
    setenv("SMF_TEST_STORE_FAIL_RESERVE", "1", 1);
#endif
    FactorStore<double> s;
    EXPECT_FALSE(s.reserve_slices(2, 1000));  // reported, still usable
    ASSERT_TRUE(s.slice(0).grow(50));
    ASSERT_TRUE(s.slice(1).grow(20));
    EXPECT_TRUE(s.slice(0).detached());
    for (std::size_t k = 0; k < 50; ++k) s.slice(0).data()[k] = static_cast<double>(k);
    for (std::size_t k = 0; k < 20; ++k) s.slice(1).data()[k] = 100.0 + static_cast<double>(k);
    s.regather();
    ASSERT_EQ(s.used(), 70u);
    EXPECT_EQ(s.data()[49], 49.0);
    EXPECT_EQ(s.data()[50], 100.0);
#ifdef _WIN32
    _putenv_s("SMF_TEST_STORE_FAIL_RESERVE", "");
#else
    unsetenv("SMF_TEST_STORE_FAIL_RESERVE");
#endif
}

// A one-slice store whose slice outgrows its reservation (the serial engine's
// factor when delays exceed the reserved 4x prediction): data() is the
// slice's new memory, so offsets from data() stay valid without a regather.
TEST(FactorStore, SingleSliceDetachKeepsDataValid) {
    FactorStore<double> s;
    ASSERT_TRUE(s.reserve_slices(1, 100));
    ASSERT_TRUE(s.slice(0).grow(100));
    for (std::size_t k = 0; k < 100; ++k) s.slice(0).data()[k] = static_cast<double>(k);
    ASSERT_TRUE(s.slice(0).grow(100000));
    ASSERT_TRUE(s.slice(0).detached());
    EXPECT_EQ(s.data(), s.slice(0).data());
    EXPECT_EQ(s.size(), 100000u);
    EXPECT_EQ(s.data()[99], 99.0);
    s.data()[99999] = 1.0;  // writable through data()
}

// A copy made when no range can be reserved keeps every slice at its original
// offset from data() (the factor's offsets were taken there), in one block.
TEST(FactorStore, CopyWithoutReservationKeepsOffsets) {
    FactorStore<double> s;
    ASSERT_TRUE(s.reserve_slices(2, 1000));
    ASSERT_TRUE(s.slice(0).grow(10));
    ASSERT_TRUE(s.slice(1).grow(20));
    for (std::size_t k = 0; k < 10; ++k) s.slice(0).data()[k] = static_cast<double>(k);
    for (std::size_t k = 0; k < 20; ++k) s.slice(1).data()[k] = 100.0 + static_cast<double>(k);
    const std::size_t off1 = s.slice_offset(1);
#ifdef _WIN32
    _putenv_s("SMF_TEST_STORE_FAIL_RESERVE", "1");
#else
    setenv("SMF_TEST_STORE_FAIL_RESERVE", "1", 1);
#endif
    const FactorStore<double> c(s);
#ifdef _WIN32
    _putenv_s("SMF_TEST_STORE_FAIL_RESERVE", "");
#else
    unsetenv("SMF_TEST_STORE_FAIL_RESERVE");
#endif
    ASSERT_GE(c.size(), off1 + 20);
    EXPECT_EQ(c.data()[3], 3.0);
    EXPECT_EQ(c.data()[off1 + 19], 119.0);
}

// trim() keeps what a slice reached in any of its last kTrimWindow
// factorizations (reserve_slices .. trim): a large front that moves between
// the engine's lanes from one factorization to the next does not make its
// room be released and faulted in again. Older room is released.
TEST(FactorStore, TrimKeepsRoomUsedWithinTheWindow) {
    FactorStore<double> s;
    const std::size_t cap = 256 * kMiB / sizeof(double);
    ASSERT_TRUE(s.reserve_slices(1, cap));
    ASSERT_TRUE(s.slice(0).grow(64 * kMiB / sizeof(double)));  // a big front's room
    std::memset(s.data(), 1, 64 * kMiB);
    ASSERT_TRUE(s.slice(0).grow(kMiB / sizeof(double)));        // ends at a small fill
    s.trim();
    EXPECT_GE(s.committed_bytes(), 64 * kMiB) << "the room just used is kept";
    for (int k = 1; k < FactorSlice<double>::kTrimWindow; ++k) {
        ASSERT_TRUE(s.reserve_slices(1, cap));
        ASSERT_TRUE(s.slice(0).grow(kMiB / sizeof(double)));
        s.trim();
        EXPECT_GE(s.committed_bytes(), 64 * kMiB) << "factorization " << k << " after the big one";
    }
    ASSERT_TRUE(s.reserve_slices(1, cap));
    ASSERT_TRUE(s.slice(0).grow(kMiB / sizeof(double)));
    s.trim();
    EXPECT_LE(s.committed_bytes(), 16 * kMiB + 2 * kMiB) << "room older than the window is released";
}
