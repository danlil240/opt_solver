#pragma once
// Addressable 4-ary min-heap of (key, id) pairs, ids in [0, n), at most one
// entry per id, with decrease-key. Entries are ordered lexicographically by
// (key, id), a strict total order on distinct ids, so the pop sequence is
// fully determined by the pushed keys: exactly the order in which a
// std::push_heap / pop_heap heap of std::pair<double, int> under
// std::greater (with superseded duplicates skipped) yields its live
// entries. Used by the matching's shortest augmenting path searches.

#include <cstddef>
#include <vector>

namespace smf {

class IndexedMinHeap {
public:
    /// Empty heap over ids [0, n).
    void reset(int n) {
        pos_.assign(static_cast<std::size_t>(n), -1);
        heap_.clear();
    }

    bool empty() const { return heap_.empty(); }
    std::size_t size() const { return heap_.size(); }
    bool contains(int id) const { return pos_[static_cast<std::size_t>(id)] >= 0; }

    double top_key() const { return heap_.front().key; }
    int top_id() const { return heap_.front().id; }

    /// Inserts id with `key`, or lowers its key to `key` if present (the key
    /// must not grow).
    void push_or_decrease(int id, double key) {
        int &p = pos_[static_cast<std::size_t>(id)];
        if (p < 0) {
            heap_.push_back(Entry{key, id});
            sift_up(heap_.size() - 1);
        } else {
            heap_[static_cast<std::size_t>(p)].key = key;
            sift_up(static_cast<std::size_t>(p));
        }
    }

    /// Removes the top entry.
    void pop() {
        pos_[static_cast<std::size_t>(heap_.front().id)] = -1;
        const Entry last = heap_.back();
        heap_.pop_back();
        if (!heap_.empty()) sift_down(last);
    }

    /// Removes every entry, in O(size).
    void clear() {
        for (const Entry &e : heap_) pos_[static_cast<std::size_t>(e.id)] = -1;
        heap_.clear();
    }

private:
    struct Entry {
        double key;
        int id;
    };
    static constexpr std::size_t kArity = 4;

    static bool before(const Entry &a, const Entry &b) {
        return a.key < b.key || (a.key == b.key && a.id < b.id);
    }

    void place(std::size_t k, const Entry &e) {
        heap_[k] = e;
        pos_[static_cast<std::size_t>(e.id)] = static_cast<int>(k);
    }

    void sift_up(std::size_t k) {
        const Entry x = heap_[k];
        while (k > 0) {
            const std::size_t up = (k - 1) / kArity;
            if (!before(x, heap_[up])) break;
            place(k, heap_[up]);
            k = up;
        }
        place(k, x);
    }

    // Fills the hole at the root with x, moving it down.
    void sift_down(const Entry &x) {
        const std::size_t n = heap_.size();
        std::size_t k = 0;
        while (true) {
            const std::size_t c0 = kArity * k + 1;
            if (c0 >= n) break;
            const std::size_t c1 = c0 + kArity < n ? c0 + kArity : n;
            std::size_t best = c0;
            for (std::size_t c = c0 + 1; c < c1; ++c)
                if (before(heap_[c], heap_[best])) best = c;
            if (!before(heap_[best], x)) break;
            place(k, heap_[best]);
            k = best;
        }
        place(k, x);
    }

    std::vector<Entry> heap_;
    std::vector<int> pos_;
};

}  // namespace smf
