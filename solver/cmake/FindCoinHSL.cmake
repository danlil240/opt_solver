# FindCoinHSL.cmake — locate the CoinHSL library (HSL MA27 / MA57 / MA97)
#
# USAGE
#   find_package(CoinHSL [REQUIRED] [QUIET])
#
# HINTS (set on the cmake command line or in your environment)
#   -DCOINHSL_ROOT=/path/to/coinhsl-prefix
#   -DCOINHSL_LIBRARY=/path/to/libcoinhsl.so   (overrides all searches)
#   env COINHSL_ROOT=/path/to/coinhsl-prefix
#
# RESULT VARIABLES
#   CoinHSL_FOUND          — TRUE if a usable CoinHSL library was found
#   COINHSL_LIBRARY        — full path to the library
#   COINHSL_INCLUDE_DIR    — include directory (may be empty; MA27 needs no public header)
#   COINHSL_LIBRARIES      — convenience alias for COINHSL_LIBRARY
#
# IMPORTED TARGET
#   CoinHSL::coinhsl       — UNKNOWN IMPORTED target (link with target_link_libraries)
#
# ---- LICENSING NOTE -------------------------------------------------------
# CoinHSL is licensed by STFC Rutherford Appleton Laboratory and is NOT
# freely redistributable.  It cannot be installed via apt-get on standard
# Debian / Ubuntu systems.
#
# Fastest path to obtain it:
#
#   1. Register and download CoinHSL source (free academic or commercial):
#        https://www.hsl.rl.ac.uk/
#
#   2. Build and install via ThirdParty-HSL (recommended for IPOPT users):
#        git clone https://github.com/coin-or-tools/ThirdParty-HSL
#        cd ThirdParty-HSL
#        # copy or symlink your coinhsl/ source directory here, then:
#        ./configure --prefix=/usr/local
#        make -j$(nproc)
#        sudo make install
#        # installs /usr/local/lib/libcoinhsl.so and /usr/local/lib/pkgconfig/coinhsl.pc
#
#   3. Re-configure smf pointing at that prefix:
#        cmake -S solver -B solver/build -DCOINHSL_ROOT=/usr/local
#      or point directly at the library:
#        cmake -S solver -B solver/build \
#            -DCOINHSL_LIBRARY=/usr/local/lib/libcoinhsl.so
#      or require CoinHSL (configure fails with this message when absent):
#        cmake -S solver -B solver/build -DSMF_REQUIRE_MA27=ON
# ---------------------------------------------------------------------------

include(FindPackageHandleStandardArgs)

# ---- 0. Accept a user-supplied COINHSL_LIBRARY path directly ---------------
if(COINHSL_LIBRARY AND EXISTS "${COINHSL_LIBRARY}")
    set(_coinhsl_found_lib "${COINHSL_LIBRARY}")
else()
    # ---- 1. Try pkg-config (ThirdParty-HSL installs coinhsl.pc) ------------
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND OR PKG_CONFIG_FOUND)
        pkg_check_modules(_COINHSL_PC QUIET coinhsl CoinHSL)
    endif()
    if(_COINHSL_PC_FOUND)
        set(_coinhsl_pc_lib_dirs "${_COINHSL_PC_LIBRARY_DIRS}")
    endif()

    # ---- 2. Build search hint list -----------------------------------------
    set(_coinhsl_lib_hints)
    foreach(_r
            "${COINHSL_ROOT}"
            "$ENV{COINHSL_ROOT}"
            "${_coinhsl_pc_lib_dirs}"
            /usr/local
            /usr/local/lib/x86_64-linux-gnu
            /usr/lib/x86_64-linux-gnu
            /usr/lib
            /opt/coinhsl
            /opt/local)
        if(_r)
            list(APPEND _coinhsl_lib_hints "${_r}/lib" "${_r}")
        endif()
    endforeach()

    # ---- 3. Library search -------------------------------------------------
    find_library(_coinhsl_found_lib
        NAMES coinhsl coinHSL CoinHSL hsl ma27 hsl_ma27
        HINTS ${_coinhsl_lib_hints}
        DOC "CoinHSL library containing MA27 / MA57 / MA97")
endif()

# ---- 4. Include directory (optional; MA27 has no required public header) ----
set(_coinhsl_inc_hints)
foreach(_r
        "${COINHSL_ROOT}"
        "$ENV{COINHSL_ROOT}"
        /usr/local
        /usr
        /opt/coinhsl
        /opt/local)
    if(_r)
        list(APPEND _coinhsl_inc_hints
            "${_r}/include/coin-or/hsl"
            "${_r}/include/coin-or"
            "${_r}/include/hsl"
            "${_r}/include")
    endif()
endforeach()

find_path(COINHSL_INCLUDE_DIR
    NAMES CoinHslConfig.h coin/CoinHslConfig.h hsl_ma27.h CoinHSL.h
    HINTS ${_coinhsl_inc_hints}
    DOC "CoinHSL include directory (may be empty; MA27 has no required public header)")

# ---- 5. Promote to persistent cache variable -------------------------------
set(COINHSL_LIBRARY "${_coinhsl_found_lib}" CACHE FILEPATH
    "CoinHSL library path (set to override auto-detection)" FORCE)
set(COINHSL_LIBRARIES "${COINHSL_LIBRARY}")

# ---- 6. Standard-args check ------------------------------------------------
# Build the failure message as a single concatenated string; FAIL_MESSAGE must
# receive exactly one argument (extra strings are mis-parsed as keywords by
# find_package_handle_standard_args).
string(CONCAT _coinhsl_fail_msg
    "CoinHSL (libcoinhsl) was not found.\n"
    "CoinHSL is licensed by STFC Rutherford Appleton Laboratory and is NOT\n"
    "available via apt-get on standard Debian/Ubuntu systems.\n"
    "\n"
    "Fastest path:\n"
    "  1. Register and download source (free academic/commercial licence):\n"
    "       https://www.hsl.rl.ac.uk/\n"
    "  2. Build via ThirdParty-HSL:\n"
    "       git clone https://github.com/coin-or-tools/ThirdParty-HSL\n"
    "       cd ThirdParty-HSL\n"
    "       ./configure --prefix=/usr/local && make -j\$(nproc) && sudo make install\n"
    "  3. Re-run cmake with one of:\n"
    "       -DCOINHSL_ROOT=/usr/local\n"
    "       -DCOINHSL_LIBRARY=/usr/local/lib/libcoinhsl.so\n"
    "  To make a missing CoinHSL a hard configure error in future:\n"
    "       -DSMF_REQUIRE_MA27=ON\n")
find_package_handle_standard_args(CoinHSL
    REQUIRED_VARS COINHSL_LIBRARY
    FAIL_MESSAGE "${_coinhsl_fail_msg}")

# ---- 7. Create imported target ---------------------------------------------
if(CoinHSL_FOUND AND NOT TARGET CoinHSL::coinhsl)
    add_library(CoinHSL::coinhsl UNKNOWN IMPORTED)
    set_target_properties(CoinHSL::coinhsl PROPERTIES
        IMPORTED_LOCATION "${COINHSL_LIBRARY}")
    if(COINHSL_INCLUDE_DIR)
        set_target_properties(CoinHSL::coinhsl PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${COINHSL_INCLUDE_DIR}")
    endif()
endif()

mark_as_advanced(COINHSL_LIBRARY COINHSL_INCLUDE_DIR)
