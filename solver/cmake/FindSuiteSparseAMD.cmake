# FindSuiteSparseAMD.cmake
# Locates the SuiteSparse AMD library (libamd) and its header.
#
# Result variables:
#   SuiteSparseAMD_FOUND          - TRUE if found
#   SuiteSparseAMD_INCLUDE_DIRS   - Include directory containing amd.h
#   SuiteSparseAMD_LIBRARIES      - Full path to libamd
#
# Imported target:
#   SuiteSparseAMD::amd

include(FindPackageHandleStandardArgs)

find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(PC_AMD QUIET amd)
endif()

find_path(SuiteSparseAMD_INCLUDE_DIR
    NAMES suitesparse/amd.h amd.h
    HINTS ${PC_AMD_INCLUDE_DIRS}
    PATHS
        /usr/include
        /usr/local/include
        /usr/include/suitesparse
        /usr/local/include/suitesparse
)

find_library(SuiteSparseAMD_LIBRARY
    NAMES amd
    HINTS ${PC_AMD_LIBRARY_DIRS}
    PATHS
        /usr/lib
        /usr/local/lib
        /usr/lib/x86_64-linux-gnu
)

find_package_handle_standard_args(SuiteSparseAMD
    REQUIRED_VARS SuiteSparseAMD_LIBRARY SuiteSparseAMD_INCLUDE_DIR
)

if(SuiteSparseAMD_FOUND)
    set(SuiteSparseAMD_INCLUDE_DIRS "${SuiteSparseAMD_INCLUDE_DIR}")
    set(SuiteSparseAMD_LIBRARIES    "${SuiteSparseAMD_LIBRARY}")

    if(NOT TARGET SuiteSparseAMD::amd)
        add_library(SuiteSparseAMD::amd UNKNOWN IMPORTED)
        set_target_properties(SuiteSparseAMD::amd PROPERTIES
            IMPORTED_LOCATION             "${SuiteSparseAMD_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${SuiteSparseAMD_INCLUDE_DIR}"
        )
    endif()
endif()

mark_as_advanced(SuiteSparseAMD_INCLUDE_DIR SuiteSparseAMD_LIBRARY)
