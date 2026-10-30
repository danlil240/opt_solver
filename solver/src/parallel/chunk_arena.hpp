#pragma once
// Bump allocation of blocks that live for one factorization, from chunks
// kept across factorizations.

#include <algorithm>
#include <cstddef>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace smf {

// Each factorization resets the arena and takes its blocks from the first
// chunk on, so the arena grows (by one chunk, whose pages are touched only
// as they are used) only when a factorization needs more in total than any
// before, whichever threads take the blocks and in whatever order: unlike
// per-thread buffers, which grow whenever a factorization deals the work
// differently. A block never spans two chunks. Thread-safe (one lock per
// block).
template <class T> class ChunkArena {
public:
  // At least `chunk` elements per chunk (a larger block gets a chunk of its
  // own size).
  explicit ChunkArena(std::size_t chunk = (std::size_t{16} << 20) / sizeof(T))
      : chunk_(std::max<std::size_t>(1, chunk)) {}
  ChunkArena(const ChunkArena &) = delete;
  ChunkArena &operator=(const ChunkArena &) = delete;

  // Every block taken so far is dead: the next take starts over.
  void reset() {
    std::lock_guard<std::mutex> lk(mtx_);
    cur_ = 0;
    used_ = 0;
  }
  // n uninitialised elements, valid until the next reset.
  T *take(std::size_t n) {
    std::lock_guard<std::mutex> lk(mtx_);
    for (;; ++cur_, used_ = 0) {
      if (cur_ == chunks_.size()) {
        const std::size_t cap = std::max(chunk_, n);
        chunks_.emplace_back(std::unique_ptr<T[]>(new T[cap]), cap);
      }
      if (used_ + n <= chunks_[cur_].second) {
        T *const at = chunks_[cur_].first.get() + used_;
        used_ += n;
        return at;
      }
    }
  }
  // Chunks allocated so far (they are never freed before the arena).
  std::size_t chunks() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return chunks_.size();
  }

private:
  const std::size_t chunk_;
  mutable std::mutex mtx_;
  std::vector<std::pair<std::unique_ptr<T[]>, std::size_t>> chunks_;  // (chunk, elements)
  std::size_t cur_ = 0, used_ = 0;  // the chunk taken from, and its elements taken
};

} // namespace smf
