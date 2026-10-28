// IndexedMinHeap (solver/src/indexed_min_heap.hpp): the addressable heap
// of the matching's shortest-path searches. Its pop order must be exactly
// the (key, id) lexicographic order, the order in which a std::push_heap /
// pop_heap heap of pair<double, int> with lazy duplicates (the searches'
// former queue) yields its live entries.

#include "../src/indexed_min_heap.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace smf {
namespace {

std::vector<std::pair<double, int>> drain(IndexedMinHeap &h) {
    std::vector<std::pair<double, int>> out;
    while (!h.empty()) {
        out.emplace_back(h.top_key(), h.top_id());
        h.pop();
    }
    return out;
}

TEST(IndexedMinHeap, EmptyAfterReset) {
    IndexedMinHeap h;
    h.reset(4);
    EXPECT_TRUE(h.empty());
    EXPECT_EQ(h.size(), 0u);
    for (int i = 0; i < 4; ++i) EXPECT_FALSE(h.contains(i));
}

TEST(IndexedMinHeap, EqualKeysPopInIncreasingId) {
    IndexedMinHeap h;
    h.reset(10);
    for (int i = 9; i >= 0; --i) h.push_or_decrease(i, 1.5);
    const auto out = drain(h);
    ASSERT_EQ(out.size(), 10u);
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(out[static_cast<std::size_t>(i)].first, 1.5);
        EXPECT_EQ(out[static_cast<std::size_t>(i)].second, i);
    }
}

TEST(IndexedMinHeap, KeyThenIdOrder) {
    IndexedMinHeap h;
    h.reset(6);
    h.push_or_decrease(5, 2.0);
    h.push_or_decrease(0, 3.0);
    h.push_or_decrease(3, 2.0);
    h.push_or_decrease(1, 1.0);
    h.push_or_decrease(4, 3.0);
    h.push_or_decrease(2, 0.0);
    const std::vector<std::pair<double, int>> want = {{0.0, 2}, {1.0, 1}, {2.0, 3}, {2.0, 5}, {3.0, 0}, {3.0, 4}};
    EXPECT_EQ(drain(h), want);
}

TEST(IndexedMinHeap, DecreaseKeyMovesEntryAndKeepsOneEntryPerId) {
    IndexedMinHeap h;
    h.reset(5);
    for (int i = 0; i < 5; ++i) h.push_or_decrease(i, 10.0 + i);
    h.push_or_decrease(4, 1.0);  // to the front
    h.push_or_decrease(2, 10.0); // ties with id 0, larger id
    h.push_or_decrease(3, 0.5);
    EXPECT_EQ(h.size(), 5u);
    EXPECT_TRUE(h.contains(2));
    const std::vector<std::pair<double, int>> want = {{0.5, 3}, {1.0, 4}, {10.0, 0}, {10.0, 2}, {11.0, 1}};
    EXPECT_EQ(drain(h), want);
    for (int i = 0; i < 5; ++i) EXPECT_FALSE(h.contains(i));
}

TEST(IndexedMinHeap, ClearForgetsEntriesAndIsReusable) {
    IndexedMinHeap h;
    h.reset(8);
    for (int i = 0; i < 8; ++i) h.push_or_decrease(i, static_cast<double>(8 - i));
    h.pop();
    h.clear();
    EXPECT_TRUE(h.empty());
    for (int i = 0; i < 8; ++i) EXPECT_FALSE(h.contains(i));
    h.push_or_decrease(6, 4.0);
    h.push_or_decrease(1, 4.0);
    const std::vector<std::pair<double, int>> want = {{4.0, 1}, {4.0, 6}};
    EXPECT_EQ(drain(h), want);
}

// Random push / decrease / pop sequences (keys from a small set, so ties
// are frequent) against the lazy-duplicate std heap the searches used:
// identical pop sequences.
TEST(IndexedMinHeap, MatchesLazyStdHeapOnRandomSequences) {
    std::mt19937 rng(12345);
    const auto inf = std::numeric_limits<double>::infinity();
    for (int round = 0; round < 200; ++round) {
        const int n = 1 + static_cast<int>(rng() % 60);
        IndexedMinHeap h;
        h.reset(n);
        std::vector<std::pair<double, int>> lazy;
        const auto cmp = std::greater<std::pair<double, int>>();
        std::vector<double> key(static_cast<std::size_t>(n), inf);
        std::vector<char> popped(static_cast<std::size_t>(n), 0);
        // Pops the lazy heap's next live entry, or returns false.
        const auto lazy_pop = [&](std::pair<double, int> &out) {
            while (!lazy.empty()) {
                std::pop_heap(lazy.begin(), lazy.end(), cmp);
                const auto e = lazy.back();
                lazy.pop_back();
                if (popped[static_cast<std::size_t>(e.second)] || e.first > key[static_cast<std::size_t>(e.second)])
                    continue;
                out = e;
                return true;
            }
            return false;
        };
        for (int step = 0; step < 400; ++step) {
            const unsigned op = rng() % 3;
            if (op < 2) {
                const int id = static_cast<int>(rng() % static_cast<unsigned>(n));
                if (popped[static_cast<std::size_t>(id)]) continue;
                const double k = static_cast<double>(rng() % 8) * 0.25;
                if (!(k < key[static_cast<std::size_t>(id)])) continue;  // the searches only decrease
                key[static_cast<std::size_t>(id)] = k;
                lazy.emplace_back(k, id);
                std::push_heap(lazy.begin(), lazy.end(), cmp);
                h.push_or_decrease(id, k);
            } else {
                std::pair<double, int> want;
                const bool any = lazy_pop(want);
                ASSERT_EQ(any, !h.empty());
                if (!any) continue;
                EXPECT_EQ(h.top_key(), want.first);
                EXPECT_EQ(h.top_id(), want.second);
                h.pop();
                popped[static_cast<std::size_t>(want.second)] = 1;
            }
        }
        std::pair<double, int> want;
        while (lazy_pop(want)) {
            ASSERT_FALSE(h.empty());
            EXPECT_EQ(h.top_key(), want.first);
            EXPECT_EQ(h.top_id(), want.second);
            h.pop();
            popped[static_cast<std::size_t>(want.second)] = 1;
        }
        EXPECT_TRUE(h.empty());
    }
}

}  // namespace
}  // namespace smf
