// Reserved address space, committed on demand (see vm_reserve.hpp).

#include "vm_reserve.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace smf {
namespace {
std::atomic<long long> g_decommit_bytes{0};
bool fail_reserve_hook() noexcept {
#ifdef _WIN32
  char *e = nullptr;
  std::size_t len = 0;
  if (_dupenv_s(&e, &len, "SMF_TEST_STORE_FAIL_RESERVE") != 0 || e == nullptr) return false;
  const bool on = std::atoi(e) != 0;
  std::free(e);
  return on;
#else
  const char *e = std::getenv("SMF_TEST_STORE_FAIL_RESERVE");
  return e != nullptr && std::atoi(e) != 0;
#endif
}
} // namespace

#ifdef _WIN32
void *vm_reserve(std::size_t bytes) noexcept {
  if (bytes == 0 || fail_reserve_hook()) return nullptr;
  return VirtualAlloc(nullptr, bytes, MEM_RESERVE, PAGE_NOACCESS);
}
bool vm_commit(void *p, std::size_t bytes) noexcept {
  return bytes == 0 || VirtualAlloc(p, bytes, MEM_COMMIT, PAGE_READWRITE) != nullptr;
}
void vm_decommit(void *p, std::size_t bytes) noexcept {
  if (bytes == 0) return;
  VirtualFree(p, bytes, MEM_DECOMMIT);
  g_decommit_bytes.fetch_add(static_cast<long long>(bytes), std::memory_order_relaxed);
}
void vm_release(void *p, std::size_t) noexcept {
  if (p != nullptr) VirtualFree(p, 0, MEM_RELEASE);
}
std::size_t vm_granule() noexcept {
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  return si.dwAllocationGranularity;
}
std::size_t vm_page() noexcept {
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  return si.dwPageSize;
}
#else
void *vm_reserve(std::size_t bytes) noexcept {
  if (bytes == 0 || fail_reserve_hook()) return nullptr;
  void *p = mmap(nullptr, bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  return p == MAP_FAILED ? nullptr : p;
}
bool vm_commit(void *p, std::size_t bytes) noexcept {
  return bytes == 0 || mprotect(p, bytes, PROT_READ | PROT_WRITE) == 0;
}
void vm_decommit(void *p, std::size_t bytes) noexcept {
  if (bytes == 0) return;
  madvise(p, bytes, MADV_DONTNEED);
  mprotect(p, bytes, PROT_NONE);
  g_decommit_bytes.fetch_add(static_cast<long long>(bytes), std::memory_order_relaxed);
}
void vm_release(void *p, std::size_t bytes) noexcept {
  if (p != nullptr) munmap(p, bytes);
}
std::size_t vm_granule() noexcept { return static_cast<std::size_t>(sysconf(_SC_PAGESIZE)); }
std::size_t vm_page() noexcept { return static_cast<std::size_t>(sysconf(_SC_PAGESIZE)); }
#endif

long long vm_decommit_bytes() noexcept { return g_decommit_bytes.load(std::memory_order_relaxed); }

} // namespace smf
