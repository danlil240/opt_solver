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

Removed from that copy since, so that nothing under LGPL is built or shipped
(see patches 4 and 5):

| file | what it was | why it can go |
|---|---|---|
| `gklib/getopt.c` | glibc's getopt (LGPL-2.1) | used only by GKlib's command-line programs |
| `gklib/gk_getopt.h` | declarations for `getopt.c` | nothing else uses them |
| `gklib/gkregex.c`, `gklib/gkregex.h` | glibc's regex (LGPL-2.1), built on MSVC under `USE_GKREGEX` | its only caller, `gk_strstr_replace()`, is removed and never called by METIS |
| `gklib/gk_mksort.h` (content) | GKlib's sort macro, adapted from glibc's qsort (LGPL-2.1) | replaced by smf's own implementation (patch 4) |

## Licences

smf itself is MIT. The vendored code is linked statically into smf's
libraries (`smf_metis`, `libsmf`, `libsmf_ma97`); `install` copies these
notices to `share/doc/smf/third_party`.

- **Apache License 2.0** (`LICENSE-Apache-2.0.txt`), all of it except the
  headers below:
  - METIS: Copyright 1995-2013, Regents of the University of Minnesota (`metis/LICENSE`).
  - GKlib: Copyright 1995-2018, Regents of the University of Minnesota (`gklib/LICENSE.txt`).
  - `gklib/gk_mksort.h` is smf's own code (patch 4), Apache-2.0 as well.
  - Each file smf changed says so where it changed (`smf patch N` comments),
    as section 4(b) asks.
- **BSD-style**, notices in the files: `gklib/gk_ms_stdint.h`,
  `gklib/gk_ms_inttypes.h` (Alexander Chemeris; MSVC headers) and the
  MT19937-64 generator in `gklib/random.c` (Matsumoto and Nishimura; only with
  `USE_GKRAND`, which smf does not define).

Nothing under the LGPL or GPL is left in this directory.

## smf's patches

Five, all in `patches/smf-metis.patch`: `git diff -B -D` against the
unmodified copy, paths from the repository root. Deleted and rewritten files
appear without their old content, so no LGPL text is in the patch either; to
redo it on fresh sources, delete the files in the table above, then apply the
remaining hunks (for example `git apply --exclude` the deleted paths).

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
   active on the thread (METIS then returns its error code, as before, in
   every API function that sets a trap; see the gaps below).

4. **`gklib/gk_mksort.h`: smf's own `GK_MKQSORT`.** The original macro was
   adapted from glibc's qsort (LGPL-2.1). smf's replacement is written from a
   description of that algorithm, not from its code: median-of-three
   quicksort with an explicit stack, the smaller side first, ranges of at
   most 9 elements (the original's threshold of 8) left for one final
   insertion-sort pass that starts with the minimum of the first 9 elements
   moved to the front. METIS sorts key-value pairs by key only, so equal-key
   order feeds the orderings; the replacement gives the original's element
   order exactly. Checked against the original on 38,928 random arrays with
   many duplicate keys (ascending and descending, int and float keys, n up
   to 200,000) before the original was removed; the gtest
   `ParallelNd.GkSortPinnedOrder` pins its outputs.
5. **No regex, no getopt.** `gk_strstr_replace()` (`string.c`, the only
   regex user; METIS never calls it) and its prototype are removed, `GKlib.h`
   no longer includes a regex header or `gk_getopt.h`, and `getopt.c`,
   `gk_getopt.h`, `gkregex.c` and `gkregex.h` are deleted (table above).

Known gaps, unchanged from upstream and not reached by smf in normal use:
`METIS_ComputeVertexSeparator` sets no trap (`libmetis/parmetis.c`), so an
out-of-memory error inside it raises `SIGABRT` and ends the process, as it
did with conda-forge's DLL; `METIS_PartMeshNodal/Dual` (`meshpart.c`, unused
by smf) call `raise(SIGERR)` directly, which now terminates instead of being
caught.

## Build flags (`CMakeLists.txt`)

- MSVC: `WIN32 MSC _CRT_SECURE_NO_DEPRECATE __thread=__declspec(thread)`,
  upstream's set with real TLS and without `USE_GKREGEX` (patch 5).
- GCC / Clang: `LINUX _FILE_OFFSET_BITS=64`, `HAVE_EXECINFO_H` /
  `HAVE_GETLINE` when present, `-std=c99 -fno-strict-aliasing`; native
  `__thread`. Symbols get hidden visibility, and `libsmf_ma97` links with
  `--exclude-libs,ALL`, so a process that also loads a system `libmetis` keeps
  the two copies apart.
- Always `NDEBUG NDEBUG2` (no METIS/GKlib assertions), warnings off, and none
  of smf's own flags (`/arch:AVX2`, `-march=native`, ...): the compiler's
  defaults for the build type, like the conda-forge build.
