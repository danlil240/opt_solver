# smf_metis: the vendored METIS 5.2.1 + GKlib 8bd6bad7 as one static library
# (see README.md for versions, licences and smf's patches).
#
# Upstream's CMake is not used: its MSVC thread-local-storage check fails and
# expands GKlib's `__thread` to nothing, which makes GKlib's per-call state
# (memory core, error jump buffers) process-global, so concurrent METIS calls
# crash. Here `__thread` is GKlib's real thread-local storage on every
# compiler, which makes METIS_NodeND / METIS_ComputeVertexSeparator safe to
# call from several threads at once (SMF_METIS_THREADSAFE).

# The parent directory's flags (/W4, /arch:AVX2, -march=native, -Wpedantic,
# ...) are smf's, not METIS's: start from the compiler's defaults for the
# build type, as the conda-forge build did, so the orderings stay bit for bit
# what they were.
set_property(DIRECTORY PROPERTY COMPILE_OPTIONS "")

file(GLOB _smf_gklib_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/gklib/*.c")
file(GLOB _smf_metis_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/metis/libmetis/*.c")
if(MSVC)
    list(APPEND _smf_gklib_sources "${CMAKE_CURRENT_SOURCE_DIR}/gklib/win32/adapt.c")
endif()

add_library(smf_metis STATIC ${_smf_gklib_sources} ${_smf_metis_sources})
set_target_properties(smf_metis PROPERTIES
    POSITION_INDEPENDENT_CODE ON   # linked into libsmf_ma97
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF)
target_include_directories(smf_metis
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/metis/include>
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/gklib"
        "${CMAKE_CURRENT_SOURCE_DIR}/metis/libmetis")
# NDEBUG / NDEBUG2: GKlib's and METIS's assertions off, as in their release
# builds. Index and real widths (32 / 32) are fixed in metis.h (patch 1).
target_compile_definitions(smf_metis PRIVATE NDEBUG NDEBUG2)

if(MSVC)
    target_include_directories(smf_metis PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/gklib/win32")
    # Upstream's MSVC defines, with __thread as real TLS (the fix).
    target_compile_definitions(smf_metis PRIVATE
        WIN32 MSC _CRT_SECURE_NO_DEPRECATE USE_GKREGEX "__thread=__declspec(thread)")
    # /W0: upstream code, not ours to lint; C4005 (GKlib redefines INFINITY
    # before <math.h>) is reported even then, so it is named too.
    target_compile_options(smf_metis PRIVATE /W0 /wd4005)
else()
    # Upstream's GCC/Clang defines; __thread is native there.
    include(CheckIncludeFile)
    include(CheckSymbolExists)
    check_include_file(execinfo.h SMF_METIS_HAVE_EXECINFO_H)
    check_symbol_exists(getline "stdio.h" SMF_METIS_HAVE_GETLINE)
    target_compile_definitions(smf_metis PRIVATE
        LINUX _FILE_OFFSET_BITS=64
        $<$<BOOL:${SMF_METIS_HAVE_EXECINFO_H}>:HAVE_EXECINFO_H>
        $<$<BOOL:${SMF_METIS_HAVE_GETLINE}>:HAVE_GETLINE>)
    target_compile_options(smf_metis PRIVATE -fno-strict-aliasing -w)
    # The directory reset above also drops smf's -O3; without a build type
    # (single-config generators, no CMAKE_BUILD_TYPE) METIS would be built
    # at -O0. Optimisation does not change the orderings here (ISO C99:
    # -ffp-contract=off; no -march).
    if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
        target_compile_options(smf_metis PRIVATE -O2)
    endif()
    target_link_libraries(smf_metis PUBLIC m)
    # Keep METIS_* and gk_* out of the dynamic symbol table of any shared
    # library smf is linked into (libsmf_ma97): a process that also loads a
    # system libmetis must not have its calls bound to this copy, or ours to
    # its (non-thread-local) one.
    set_target_properties(smf_metis PROPERTIES C_VISIBILITY_PRESET hidden)
endif()
