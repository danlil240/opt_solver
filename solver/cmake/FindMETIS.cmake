# FindMETIS.cmake — finds system METIS 5 (NOT coin-or bundled METIS)
# Prefers /usr/include/metis.h (libmetis-dev) over coin-or bundled headers.

include(FindPackageHandleStandardArgs)

# Explicitly search system paths first (coin-or metis is NOT the METIS 5 API)
find_path(METIS_INCLUDE_DIR
    NAMES metis.h
    PATHS
        /usr/include
        /usr/local/include
    NO_DEFAULT_PATH
)
if(NOT METIS_INCLUDE_DIR)
    find_path(METIS_INCLUDE_DIR NAMES metis.h)
endif()

find_library(METIS_LIBRARY
    NAMES metis
    PATHS
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
    NO_DEFAULT_PATH
)
if(NOT METIS_LIBRARY)
    find_library(METIS_LIBRARY NAMES metis)
endif()

find_package_handle_standard_args(METIS
    REQUIRED_VARS METIS_LIBRARY METIS_INCLUDE_DIR
)

if(METIS_FOUND)
    set(METIS_INCLUDE_DIRS "${METIS_INCLUDE_DIR}")
    set(METIS_LIBRARIES    "${METIS_LIBRARY}")

    if(NOT TARGET METIS::metis)
        add_library(METIS::metis UNKNOWN IMPORTED)
        set_target_properties(METIS::metis PROPERTIES
            IMPORTED_LOCATION             "${METIS_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${METIS_INCLUDE_DIR}"
        )
    endif()
endif()

mark_as_advanced(METIS_INCLUDE_DIR METIS_LIBRARY)
