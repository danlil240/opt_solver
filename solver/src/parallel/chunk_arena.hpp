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
// differently. A block never spans two chunks. A block larger than a chunk
// stays out of the bump chain (it would skip the chunks left, whose room
// then depends on where in the order it came): it gets a chunk of its own,
// kept and handed again, best fit, to such blocks after a reset, so the same
// large blocks take no new chunk in any order either. Thread-safe (one lock
// per block).
template <class T> class ChunkArena {
public:
  // `chunk` elements per chunk of the bump chain.
  explicit ChunkArena(std::size_t chunk = (std::size_t{16} << 20) / sizeof(T))
      : chunk_(std::max<std::size_t>(1, chunk)) {}
  ChunkArena(const ChunkArena &) = delete;
  ChunkArena &operator=(const ChunkArena &) = delete;

  // Every block taken so far is dead: the next take starts over.
  void reset() {
    std::lock_guard<std::mutex> lk(mtx_);
    cur_ = 0;
    used_ = 0;
    for (Big &b : big_) b.taken = false;
  }
  // n uninitialised elements, valid until the next reset.
  T *take(std::size_t n) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (n > chunk_) return take_big(n);
    for (;; ++cur_, used_ = 0) {
      if (cur_ == chunks_.size()) chunks_.emplace_back(new T[chunk_]);
      if (used_ + n <= chunk_) {
        T *const at = chunks_[cur_].get() + used_;
        used_ += n;
        return at;
      }
    }
  }
  // Chunks allocated so far, own-size ones included (they are never freed
  // before the arena).
  std::size_t chunks() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return chunks_.size() + big_.size();
  }

private:
  struct Big {
    std::unique_ptr<T[]> data;
    std::size_t size = 0;
    bool taken = false;
  };

  // Under mtx_: the smallest free own-size chunk that holds n, else a new one.
  T *take_big(std::size_t n) {
    Big *best = nullptr;
    for (Big &b : big_)
      if (!b.taken && b.size >= n && (best == nullptr || b.size < best->size)) best = &b;
    if (best == nullptr) {
      big_.push_back(Big{std::unique_ptr<T[]>(new T[n]), n, false});
      best = &big_.back();
    }
    best->taken = true;
    return best->data.get();
  }

  const std::size_t chunk_;
  mutable std::mutex mtx_;
  std::vector<std::unique_ptr<T[]>> chunks_;  // the bump chain, chunk_ elements each
  std::size_t cur_ = 0, used_ = 0;  // the chunk taken from, and its elements taken
  std::vector<Big> big_;             // own-size chunks of blocks larger than chunk_
};

} // namespace smf
