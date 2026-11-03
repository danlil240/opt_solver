# Vendored third-party code

smf builds METIS from these sources by default (`SMF_VENDORED_METIS=ON`) as one
static library, `smf_metis` (`CMakeLists.txt` here; upstream's CMake is not used).
`-DSMF_VENDORED_METIS=OFF` links an external METIS through `find_package(METIS)`
instead, whose calls smf then serialises with a process-wide mutex.

## Why vendored

The conda-forge `metis` 5.2.1 package on Windows (MSVC) is built with GKlib's
`__thread` defined to nothing: upstream's CMake thread-local-storage check
(`GKlibSystem.cmake`, `conf/gkbuild.cmake`) runs `try_compile` without the
`-D__thread=__declspec(thread)` it then adds, fails, and appends `-D__thread=`,
which MSVC applies last. GKlib's per-call state (the `gkmcore` memory core,
the `gk_jbufs` error jump buffers) becomes process-global, and two concurrent
METIS calls abort or corrupt the heap ("Could not find pointer ... in mcore").
Built here with real thread-local storage, 16 threads x 200 concurrent
`METIS_NodeND` / `METIS_ComputeVertexSeparator` calls give results bitwise
equal to the serial ones (`ParallelNd.RawConcurrentMetisIsSafe`), and the
serial orderings are bitwise those of the conda-forge DLL
(`ParallelNd.VendoredMetisMatchesCondaOrdering`).

## Sources

| dir | upstream | version | tarball | SHA-256 of the tarball |
|---|---|---|---|---|
| `metis/` | https://github.com/KarypisLab/METIS | tag `v5.2.1` | https://github.com/KarypisLab/METIS/archive/v5.2.1.tar.gz | `1a4665b2cd07edc2f734e30d7460afb19c1217c2547c2ac7bf6e1848d50aff7a` |
| `gklib/` | https://github.com/KarypisLab/GKlib | commit `8bd6bad750b2b0d90800c632cf18e8ee93ad72d7` | https://github.com/KarypisLab/GKlib/archive/8bd6bad750b2b0d90800c632cf18e8ee93ad72d7.tar.gz | `e1d59de12336731e6dde8465f05de7b907fae5d72c921cffcde217a89eaab654` |

These are the versions and tarballs conda-forge's `metis` 5.2.1 recipe builds
(its MSVC DLL links GKlib statically). Copied: `metis/include/metis.h`,
`metis/libmetis/*.{c,h}`, `gklib/*.{c,h}`, `gklib/win32/adapt.{c,h}` and the
licence files; upstream's programs, tests, graphs, manuals and build files are
left out. The unmodified copy is commit `third_party: METIS 5.2.1 and GKlib
8bd6bad7 sources, unmodified` in smf's history.

## Licences

Both are under the Apache License 2.0 (`LICENSE-Apache-2.0.txt`):

- METIS: Copyright 1995-2013, Regents of the University of Minnesota (`metis/LICENSE`).
- GKlib: Copyright 1995-2018, Regents of the University of Minnesota (`gklib/LICENSE.txt`).

Each file smf changed says so where it changed (`smf patch N` comments), as
section 4(b) of the licence asks.

## smf's patches

Three, all in `patches/smf-metis.patch` (a `git diff` against the unmodified
copy, paths from the repository root; `git apply` it to a fresh copy of the
sources above):

1. **`metis/include/metis.h`: index and real widths.** `IDXTYPEWIDTH 32` and
   `REALTYPEWIDTH 32` defined, as conda-forge's
   `0004-uncomment-width-variables.patch` does (smf asserts
   `sizeof(idx_t) == sizeof(smf::Int)`); plus `SMF_VENDORED_METIS_H`, which
   `ordering_metis.cpp` checks so that a thread-safe build cannot pick up
   another `metis.h`.
2. **`gklib/random.c`: thread-local `rand()`.** Without `USE_GKRAND`, GKlib's
   random numbers come from the C runtime's `rand()`, whose state is per
   thread with MSVC but one process-wide stream with glibc: concurrent METIS
   calls on Linux would interleave it and give run-dependent orderings.
   `gk_randinit` / `gk_randint32` / `gk_randint64` now use a `__thread` copy of
   MSVC's generator (state 1 until seeded, `state = state * 214013 + 2531011`,
   returns `(state >> 16) & 0x7fff`). Windows orderings stay bit-identical,
   and every platform gets the same per-thread sequence. (`gk_macros.h`'s
   `RandomInRange` still calls `rand()`; nothing METIS calls uses it.)
3. **`gklib/error.c`: no process-wide signal handlers.** Every METIS API call
   brackets its work with `gk_sigtrap()` / `gk_siguntrap()`, which installed
   and restored `SIGABRT` / `SIGTERM` handlers with `signal()`. Handlers are
   process-wide: concurrent calls race on them and can leave GKlib's handler
   installed, or drop the host's, after every call has returned. Now
   `gk_sigtrap` / `gk_siguntrap` only push and pop the thread's jump-buffer
   level, and `gk_errexit()` jumps straight to that buffer when a METIS call is
   active on the thread (METIS then returns its error code, as before).

## Build flags (`CMakeLists.txt`)

- MSVC: `WIN32 MSC _CRT_SECURE_NO_DEPRECATE USE_GKREGEX
  __thread=__declspec(thread)`, upstream's set with real TLS.
- GCC / Clang: `LINUX _FILE_OFFSET_BITS=64`, `HAVE_EXECINFO_H` /
  `HAVE_GETLINE` when present, `-std=c99 -fno-strict-aliasing`; native
  `__thread`. Symbols get hidden visibility, and `libsmf_ma97` links with
  `--exclude-libs,ALL`, so a process that also loads a system `libmetis` keeps
  the two copies apart.
- Always `NDEBUG NDEBUG2` (no METIS/GKlib assertions), warnings off, and none
  of smf's own flags (`/arch:AVX2`, `-march=native`, ...): the compiler's
  defaults for the build type, like the conda-forge build.
