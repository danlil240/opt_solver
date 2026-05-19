# FindCHOLMOD.cmake
# Finds the CHOLMOD library (part of SuiteSparse).
#
# Defines:
#   CHOLMOD_FOUND          - TRUE if header and library found
#   CHOLMOD_INCLUDE_DIRS   - include directories
#   CHOLMOD_LIBRARIES      - libraries to link
#   CHOLMOD::CHOLMOD       - imported target (if found)

find_path(CHOLMOD_INCLUDE_DIR
    NAMES cholmod.h
    PATHS
        /usr/include/suitesparse
        /usr/local/include/suitesparse
        /usr/local/include
        /opt/local/include/ufsparse
        /opt/homebrew/include/suitesparse
)

find_library(CHOLMOD_LIBRARY
    NAMES cholmod
    PATHS
        /usr/lib/x86_64-linux-gnu
        /usr/lib
        /usr/local/lib
        /opt/local/lib
        /opt/homebrew/lib
)

# Handle standard arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CHOLMOD
    REQUIRED_VARS CHOLMOD_INCLUDE_DIR CHOLMOD_LIBRARY
)

if(CHOLMOD_FOUND)
    set(CHOLMOD_INCLUDE_DIRS "${CHOLMOD_INCLUDE_DIR}")
    set(CHOLMOD_LIBRARIES    "${CHOLMOD_LIBRARY}")

    if(NOT TARGET CHOLMOD::CHOLMOD)
        add_library(CHOLMOD::CHOLMOD UNKNOWN IMPORTED)
        set_target_properties(CHOLMOD::CHOLMOD PROPERTIES
            IMPORTED_LOCATION             "${CHOLMOD_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${CHOLMOD_INCLUDE_DIR}"
        )
    endif()
endif()

mark_as_advanced(CHOLMOD_INCLUDE_DIR CHOLMOD_LIBRARY)
