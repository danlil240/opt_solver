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
// as they are used) only when a factorization needs more than any before,
// whichever threads take the blocks: about the total of its blocks, give or
// take a chunk tail per chunk (a block never spans two chunks, so where a
// chunk ends unused depends on the order). Per-thread buffers, by contrast,
// grow whenever a factorization deals the work differently. A block
// larger than a chunk stays out of the bump chain (it would skip the chunks
// left, whose room then depends on where in the order it came): it gets a
// chunk of its own, kept and handed again, best fit, to such blocks after a
// reset, so the same large blocks take no new chunk in any order either.
// One that no free own-size chunk holds reallocates the largest free one
// (else takes a new one): there are never more own-size chunks than
// oversized blocks live at once, however their sizes drift over the
// factorizations. Thread-safe (one lock per block).
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
      if (cur_ == chunks_.size()) {
        chunks_.emplace_back(new T[chunk_]);
        ++growths_;
      }
      if (used_ + n <= chunk_) {
        T *const at = chunks_[cur_].get() + used_;
        used_ += n;
        return at;
      }
    }
  }
  // Chunks held, own-size ones included.
  std::size_t chunks() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return chunks_.size() + big_.size();
  }
  // Chunk allocations so far: new chunks and reallocated own-size ones.
  std::size_t growths() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return growths_;
  }

private:
  struct Big {
    std::unique_ptr<T[]> data;
    std::size_t size = 0;
    bool taken = false;
  };

  // Under mtx_: the smallest free own-size chunk that holds n, else the
  // largest free one reallocated to n, else a new one.
  T *take_big(std::size_t n) {
    Big *best = nullptr;     // smallest free that holds n
    Big *largest = nullptr;  // largest free (too small, if best is null)
    for (Big &b : big_) {
      if (b.taken) continue;
      if (b.size >= n && (best == nullptr || b.size < best->size)) best = &b;
      if (largest == nullptr || b.size > largest->size) largest = &b;
    }
    if (best == nullptr) {
      if (largest == nullptr) {
        big_.emplace_back();
        largest = &big_.back();
      }
      largest->size = 0;      // (an empty chunk if new throws)
      largest->data.reset();  // the old one goes before the new one comes
      largest->data.reset(new T[n]);
      largest->size = n;
      ++growths_;
      best = largest;
    }
    best->taken = true;
    return best->data.get();
  }

  const std::size_t chunk_;
  mutable std::mutex mtx_;
  std::vector<std::unique_ptr<T[]>> chunks_;  // the bump chain, chunk_ elements each
  std::size_t cur_ = 0, used_ = 0;  // the chunk taken from, and its elements taken
  std::vector<Big> big_;             // own-size chunks of blocks larger than chunk_
  std::size_t growths_ = 0;          // chunk allocations (growths())
};

} // namespace smf
