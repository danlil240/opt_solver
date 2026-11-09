#pragma once
// FactorStore: a factor array made of per-thread slices of one reserved
// address range. The tree-parallel engine gives each thread a slice that it
// appends its fronts to, so the factor needs no final gather into one array:
// offsets from data() are final as they are written (Control::factor_in_place).
// The range is reserved, not allocated: only pages a slice grows into are
// committed, so the gaps between slices cost address space only.
//
// A slice that outgrows its share of the range, or a store whose reservation
// failed, keeps going in private heap memory ("detached"); regather() then
// copies every slice into one contiguous range, in slice order (the gather the
// engine always did). Values are never zero-filled: a slice's size() elements
// are what its owner wrote.
//
// T is trivially copyable (double, Int, std::int8_t).

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

namespace smf {

// Reserved address space committed on demand (src/vm_reserve.cpp).
// vm_reserve: nullptr on failure, or when SMF_TEST_STORE_FAIL_RESERVE=1 (read
// on every call; a test hook for the fallback path).
void *vm_reserve(std::size_t bytes) noexcept;
bool vm_commit(void *p, std::size_t bytes) noexcept;  // p, bytes: multiples of vm_page()
void vm_decommit(void *p, std::size_t bytes) noexcept;
void vm_release(void *p, std::size_t bytes) noexcept;
std::size_t vm_granule() noexcept;  // reservation granularity
std::size_t vm_page() noexcept;     // commit granularity

template <class T> class FactorStore;

template <class T> class FactorSlice {
  static_assert(std::is_trivially_copyable_v<T>);

public:
  // Commits grow in chunks of this many bytes (within the slice's share).
  static constexpr std::size_t kCommitChunk = std::size_t{2} << 20;

  T *data() noexcept { return base_; }
  const T *data() const noexcept { return base_; }
  std::size_t size() const noexcept { return size_; }
  bool detached() const noexcept { return detached_; }

  // size() = n, keeping the first min(n, size()) values. Within the slice's
  // reservation it commits pages; beyond it (or with no reservation) it moves
  // to private heap memory of 2n elements. false: out of memory.
  bool grow(std::size_t n) {
    if (n <= size_) {
      size_ = n;
      return true;
    }
    if (!detached_ && n <= cap_) {
      if (n > committed_) {
        const std::size_t page = vm_page();
        std::size_t want = (n * sizeof(T) + kCommitChunk - 1) / kCommitChunk * kCommitChunk;
        want = std::min(want, (cap_ * sizeof(T) + page - 1) / page * page);
        const std::size_t have = committed_ * sizeof(T);
        const std::size_t have_pages = (have + page - 1) / page * page;
        if (want > have_pages &&
            !vm_commit(reinterpret_cast<char *>(base_) + have_pages, want - have_pages))
          return detach(n);
        committed_ = want / sizeof(T);
      }
      size_ = n;
      return true;
    }
    if (detached_ && n <= heap_cap_) {
      size_ = n;
      return true;
    }
    return detach(n);
  }

private:
  friend class FactorStore<T>;

  bool detach(std::size_t n) {
    const std::size_t cap = std::max<std::size_t>(2 * n, 64);
    T *p = static_cast<T *>(std::malloc(cap * sizeof(T)));
    if (p == nullptr) return false;
    if (size_ > 0) std::memcpy(p, base_, size_ * sizeof(T));
    if (detached_) std::free(base_);
    base_ = p;
    heap_cap_ = cap;
    detached_ = true;
    size_ = n;
    return true;
  }
  void free_heap() noexcept {
    if (detached_) std::free(base_);
    base_ = nullptr;
    detached_ = false;
    heap_cap_ = 0;
  }

  T *base_ = nullptr;
  std::size_t size_ = 0;
  std::size_t cap_ = 0;        // elements of this slice's share of the range
  std::size_t committed_ = 0;  // elements committed in the share
  std::size_t heap_cap_ = 0;   // detached: elements of the private block
  bool detached_ = false;
};

template <class T> class FactorStore {
public:
  FactorStore() = default;
  ~FactorStore() { reset(); }
  FactorStore(const FactorStore &o) { copy_from(o); }
  FactorStore &operator=(const FactorStore &o) {
    if (this != &o) {
      FactorStore tmp(o);
      swap(tmp);
    }
    return *this;
  }
  FactorStore(FactorStore &&o) noexcept { swap(o); }
  FactorStore &operator=(FactorStore &&o) noexcept {
    if (this != &o) {
      reset();
      swap(o);
    }
    return *this;
  }
  void swap(FactorStore &o) noexcept {
    std::swap(base_, o.base_);
    std::swap(range_bytes_, o.range_bytes_);
    std::swap(stride_, o.stride_);
    std::swap(cap_, o.cap_);
    slices_.swap(o.slices_);
  }

  // `slices` slices of `slice_capacity` elements each, every size() 0. An
  // unchanged layout keeps its reservation and committed pages. false: the
  // reservation failed; the slices then grow in private memory (regather()
  // makes them one array).
  bool reserve_slices(int slices, std::size_t slice_capacity) {
    const int k = std::max(1, slices);
    if (base_ != nullptr && k == static_cast<int>(slices_.size()) && slice_capacity == cap_) {
      for (std::size_t t = 0; t < slices_.size(); ++t) {
        auto &s = slices_[t];
        if (s.detached_) {
          s.free_heap();
          s.base_ = base_ + stride_ * t;
        }
        s.size_ = 0;
      }
      return true;
    }
    reset();
    cap_ = slice_capacity;
    const std::size_t g = vm_granule();
    const std::size_t share = (std::max<std::size_t>(slice_capacity, 1) * sizeof(T) + g - 1) / g * g;
    stride_ = share / sizeof(T);
    slices_.assign(static_cast<std::size_t>(k), FactorSlice<T>{});
    range_bytes_ = share * static_cast<std::size_t>(k);
    void *p = vm_reserve(range_bytes_);
    if (p == nullptr) {
      range_bytes_ = 0;
      for (auto &s : slices_) s.cap_ = 0;  // every grow detaches
      return false;
    }
    base_ = static_cast<T *>(p);
    for (std::size_t t = 0; t < slices_.size(); ++t) {
      slices_[t].base_ = base_ + stride_ * t;
      slices_[t].cap_ = cap_;
    }
    return true;
  }

  int slices() const noexcept { return static_cast<int>(slices_.size()); }
  FactorSlice<T> &slice(int t) { return slices_[static_cast<std::size_t>(t)]; }
  const FactorSlice<T> &slice(int t) const { return slices_[static_cast<std::size_t>(t)]; }
  // Elements from data(); meaningful while the slice is attached.
  std::size_t slice_offset(int t) const noexcept { return stride_ * static_cast<std::size_t>(t); }
  bool all_attached() const noexcept {
    if (base_ == nullptr) return slices_.size() <= 1 && (slices_.empty() || !slices_[0].detached_ ||
                                                         slices_[0].size_ == 0);
    return std::none_of(slices_.begin(), slices_.end(), [](const auto &s) { return s.detached_; });
  }

  // One contiguous slice holding every slice's values in slice order.
  void regather() {
    const std::size_t total = used();
    FactorStore out;
    if (!out.reserve_slices(1, total)) { /* private memory: still one slice */ }
    out.slice(0).grow(total);
    std::size_t at = 0;
    for (const auto &s : slices_) {
      if (s.size_ > 0) std::memcpy(out.slice(0).data() + at, s.base_, s.size_ * sizeof(T));
      at += s.size_;
    }
    swap(out);
  }

  T *data() noexcept { return base_ != nullptr ? base_ : (slices_.empty() ? nullptr : slices_[0].base_); }
  const T *data() const noexcept {
    return base_ != nullptr ? base_ : (slices_.empty() ? nullptr : slices_[0].base_);
  }
  // Extent: one past the highest used element, from data().
  std::size_t size() const noexcept {
    std::size_t e = 0;
    for (std::size_t t = 0; t < slices_.size(); ++t)
      if (slices_[t].size_ > 0)
        e = std::max(e, (base_ != nullptr && !slices_[t].detached_ ? stride_ * t : 0) + slices_[t].size_);
    return e;
  }
  std::size_t used() const noexcept {
    std::size_t u = 0;
    for (const auto &s : slices_) u += s.size_;
    return u;
  }
  bool empty() const noexcept { return used() == 0; }
  void clear() noexcept {
    for (auto &s : slices_) s.size_ = 0;
  }
  // One slice of n copies of v (the paths that build the factor in one piece).
  void assign(std::size_t n, const T &v) {
    if (!(base_ != nullptr && slices_.size() == 1 && cap_ >= n)) reserve_slices(1, n);
    else slices_[0].size_ = 0;
    slices_[0].grow(n);
    std::fill(slices_[0].base_, slices_[0].base_ + n, v);
  }
  template <class F> void for_each_used(F &&f) const {
    for (const auto &s : slices_)
      if (s.size_ > 0) f(static_cast<const T *>(s.base_), s.size_);
  }
  // Per attached slice: committed memory beyond its use by more than
  // max(use / 4, 16 MiB) is released down to the use (rounded up to a chunk).
  void trim() noexcept {
    const std::size_t page = vm_page();
    for (auto &s : slices_) {
      if (s.detached_ || base_ == nullptr) continue;
      const std::size_t used_b = s.size_ * sizeof(T);
      const std::size_t comm_b = s.committed_ * sizeof(T);
      if (comm_b <= used_b || comm_b - used_b <= std::max(used_b / 4, std::size_t{16} << 20)) continue;
      std::size_t keep = (used_b + FactorSlice<T>::kCommitChunk - 1) / FactorSlice<T>::kCommitChunk *
                         FactorSlice<T>::kCommitChunk;
      keep = (keep + page - 1) / page * page;
      const std::size_t comm_pages = (comm_b + page - 1) / page * page;
      if (keep >= comm_pages) continue;
      vm_decommit(reinterpret_cast<char *>(s.base_) + keep, comm_pages - keep);
      s.committed_ = keep / sizeof(T);
    }
  }
  // Committed bytes of the range plus private blocks (the trim rule, tests).
  std::size_t committed_bytes() const noexcept {
    std::size_t b = 0;
    for (const auto &s : slices_) b += s.detached_ ? s.heap_cap_ * sizeof(T) : s.committed_ * sizeof(T);
    return b;
  }

private:
  void reset() noexcept {
    for (auto &s : slices_) s.free_heap();
    slices_.clear();
    if (base_ != nullptr) vm_release(base_, range_bytes_);
    base_ = nullptr;
    range_bytes_ = 0;
    stride_ = 0;
    cap_ = 0;
  }
  void copy_from(const FactorStore &o) {
    if (o.slices_.empty()) return;
    if (o.base_ != nullptr && reserve_slices(o.slices(), o.cap_)) {
      for (std::size_t t = 0; t < o.slices_.size(); ++t) {
        const auto &src = o.slices_[t];
        if (src.detached_) {  // keep it private: its offset is not the range's
          slices_[t].cap_ = 0;
        }
        slices_[t].grow(src.size_);
        if (src.size_ > 0) std::memcpy(slices_[t].base_, src.base_, src.size_ * sizeof(T));
      }
      return;
    }
    reserve_slices(o.slices(), 0);
    for (std::size_t t = 0; t < o.slices_.size(); ++t) {
      slices_[t].cap_ = 0;
      slices_[t].grow(o.slices_[t].size_);
      if (o.slices_[t].size_ > 0)
        std::memcpy(slices_[t].base_, o.slices_[t].base_, o.slices_[t].size_ * sizeof(T));
    }
  }

  T *base_ = nullptr;
  std::size_t range_bytes_ = 0;
  std::size_t stride_ = 0;  // elements between slice starts
  std::size_t cap_ = 0;     // elements per slice (as asked)
  std::vector<FactorSlice<T>> slices_;
};

} // namespace smf
