#include "smf/factor_stack.hpp"
#include <cassert>
#include <cstring>
#include <stdexcept>

namespace smf {

// ---------------------------------------------------------------------------
// Buffer helpers
// ---------------------------------------------------------------------------

void FactorStack::Buffer::init(std::size_t bytes) {
  data = new double[bytes / sizeof(double)];
  capacity_bytes = bytes;
  top_bytes = 0;
}

void FactorStack::Buffer::grow(std::size_t min_bytes) {
  // New capacity: max(current * 1.5, min_bytes), rounded up to double boundary
  std::size_t new_cap = capacity_bytes + capacity_bytes / 2;
  if (new_cap < min_bytes)
    new_cap = min_bytes;
  // Round up to multiple of sizeof(double)
  new_cap = ((new_cap + sizeof(double) - 1) / sizeof(double)) * sizeof(double);

  double *new_data = new double[new_cap / sizeof(double)];
  if (top_bytes > 0)
    std::memcpy(new_data, data, top_bytes);

  delete[] data;
  data = new_data;
  capacity_bytes = new_cap;
}

// ---------------------------------------------------------------------------
// FactorStack
// ---------------------------------------------------------------------------

FactorStack::FactorStack(std::size_t initial_bytes) {
  // Round up to a double-aligned size
  std::size_t bytes =
      ((initial_bytes + sizeof(double) - 1) / sizeof(double)) * sizeof(double);
  buf_[0].init(bytes);
  buf_[1].init(bytes);
}

double *FactorStack::alloc(std::size_t n_doubles) {
  const std::size_t n_bytes = n_doubles * sizeof(double);
  Buffer &b = buf_[active_];

  if (b.top_bytes + n_bytes > b.capacity_bytes) {
    b.grow(b.top_bytes + n_bytes);
    ++growth_count_;
  }

  double *ptr = reinterpret_cast<double *>(reinterpret_cast<char *>(b.data) +
                                           b.top_bytes);
  b.top_bytes += n_bytes;

  // Update total high-water mark
  const std::size_t total_live = buf_[0].top_bytes + buf_[1].top_bytes;
  if (total_live > hwm_total_)
    hwm_total_ = total_live;

  return ptr;
}

void FactorStack::free_top(std::size_t n_doubles) {
  const std::size_t n_bytes = n_doubles * sizeof(double);
  Buffer &b = buf_[active_];
  assert(b.top_bytes >= n_bytes && "free_top: underflow");
  b.top_bytes -= n_bytes;
}

void FactorStack::flip() {
  active_ ^= 1;
  // Do NOT reset top_bytes — callers manage each buffer's top independently
  // via alloc/free_top. The inactive buffer's allocations remain valid.
}

} // namespace smf
