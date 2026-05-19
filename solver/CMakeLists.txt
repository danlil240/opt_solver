cmake_minimum_required(VERSION 3.20)
project(smf VERSION 0.1.0 LANGUAGES CXX)

# ---- C++ standard -------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- CMake module path --------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ---- Options ------------------------------------------------------------
option(SMF_ENABLE_OPENMP       "Enable OpenMP parallelism"              ON)
option(SMF_USE_MKL             "Use Intel MKL instead of OpenBLAS"      OFF)
option(SMF_USE_METIS           "Enable METIS 5 fill-reducing ordering"  ON)
option(SMF_USE_SUITESPARSE_AMD "Enable SuiteSparse AMD ordering"        ON)
option(SMF_DETERMINISTIC       "Enable deterministic (reproducible) mode" OFF)
option(SMF_BUILD_TESTS         "Build unit tests"                       ON)
option(SMF_BUILD_BENCHMARKS    "Build benchmarks"                       ON)
option(SMF_SANITIZE            "Enable -fsanitize=address,undefined"    OFF)
option(SMF_BUILD_IPOPT_ADAPTER "Build smf IPOPT TSymLinearSolver adapter" OFF)
option(SMF_BUILD_C_API         "Build C ABI wrapper for smf"              OFF)
option(SMF_BUILD_CHOLMOD_COMPARE "Build CHOLMOD comparison tests (ON if CHOLMOD found)" OFF)

# ---- Compiler flags -----------------------------------------------------
add_compile_options(-Wall -Wextra -Wpedantic -O2 -g)

if(SMF_SANITIZE)
    add_compile_options(-fsanitize=address,undefined)
    add_link_options(-fsanitize=address,undefined)
endif()

if(SMF_DETERMINISTIC)
    add_compile_definitions(SMF_DETERMINISTIC=1)
endif()

# ---- Required: BLAS & LAPACK --------------------------------------------
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# ---- Optional: OpenMP ---------------------------------------------------
if(SMF_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

# ---- Optional: SuiteSparse AMD (soft-fail with warning) -----------------
# Decision D-001: AMD and METIS are non-fatal; warn and set flags so the
# build succeeds even in constrained environments.
set(SMF_HAS_AMD OFF)
if(SMF_USE_SUITESPARSE_AMD)
    find_package(SuiteSparseAMD QUIET)
    if(SuiteSparseAMD_FOUND)
        set(SMF_HAS_AMD ON)
        message(STATUS "smf: SuiteSparse AMD found -- SMF_HAS_AMD=ON")
    else()
        message(WARNING "smf: SuiteSparse AMD NOT found -- SMF_HAS_AMD=OFF; "
                        "AMD ordering will be unavailable. "
                        "Install libsuitesparse-dev or libsuitesparse-amd-dev.")
    endif()
endif()

# ---- Optional: METIS 5 (soft-fail with warning) -------------------------
set(SMF_HAS_METIS OFF)
if(SMF_USE_METIS)
    find_package(METIS QUIET)
    if(METIS_FOUND)
        set(SMF_HAS_METIS ON)
        message(STATUS "smf: METIS found -- SMF_HAS_METIS=ON")
    else()
        message(WARNING "smf: METIS NOT found -- SMF_HAS_METIS=OFF; "
                        "METIS ordering will be unavailable. "
                        "Install libmetis-dev.")
    endif()
endif()

# ---- Optional: CHOLMOD (for comparison tests) ----------------------------
set(SMF_HAS_CHOLMOD OFF)
find_package(CHOLMOD QUIET)
if(CHOLMOD_FOUND)
    set(SMF_HAS_CHOLMOD ON)
    message(STATUS "smf: CHOLMOD found -- SMF_HAS_CHOLMOD=ON")
    if(NOT SMF_BUILD_CHOLMOD_COMPARE)
        set(SMF_BUILD_CHOLMOD_COMPARE ON CACHE BOOL
            "Build CHOLMOD comparison tests (auto-enabled: CHOLMOD found)" FORCE)
        message(STATUS "smf: SMF_BUILD_CHOLMOD_COMPARE auto-enabled (CHOLMOD found)")
    endif()
else()
    message(STATUS "smf: CHOLMOD NOT found -- cholmod compare test will be skipped")
    if(SMF_BUILD_CHOLMOD_COMPARE)
        message(WARNING "smf: SMF_BUILD_CHOLMOD_COMPARE=ON but CHOLMOD not found; disabling.")
        set(SMF_BUILD_CHOLMOD_COMPARE OFF CACHE BOOL "" FORCE)
    endif()
endif()

# ---- Optional: Eigen3 (test-only reference oracle) ----
if(SMF_BUILD_TESTS)
    find_package(Eigen3 QUIET CONFIG)
    if(NOT Eigen3_FOUND)
        find_package(Eigen3 QUIET)
    endif()
endif()

# ---- GTest (required when building tests) --------------------------------
if(SMF_BUILD_TESTS)
    find_package(GTest REQUIRED)
    enable_testing()
    include(CTest)
endif()

# ---- smf placeholder static library ------------------------------------
add_library(smf STATIC
    src/smf_stub.cpp
    src/csc_matrix.cpp
    src/check_matrix.cpp
    src/dense_kernel_lapack.cpp
    src/utils/memory.cpp
    src/dense_indef.cpp
    src/ordering_amd.cpp
    src/ordering_metis.cpp
    src/sym_graph.cpp
    src/etree.cpp
    src/supernode_detection.cpp
    src/assembly_tree.cpp
    src/symbolic_analysis.cpp
    src/frontal_matrix.cpp
    src/factor_stack.cpp
    src/pivoting.cpp
    src/factor_posdef.cpp
    src/factor_indef.cpp
    src/diagnostics.cpp
    src/solve_forward.cpp
    src/solve_backward.cpp
    src/solve_diag.cpp
    src/solve.cpp
    src/scaling.cpp
    src/scaling_matching.cpp
    src/solver.cpp
    src/parallel/task_tree.cpp
    src/parallel/blas_thread_guard.cpp
    src/parallel/determinism.cpp
    src/coo_to_csc.cpp
    src/solve_sparse_fwd.cpp
    src/matrix_market.cpp
)

target_include_directories(smf PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(smf PUBLIC
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

if(SMF_ENABLE_OPENMP AND TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(smf PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(smf PUBLIC SMF_PARALLEL=1)
endif()

if(SMF_HAS_AMD)
    target_link_libraries(smf PUBLIC SuiteSparseAMD::amd)
    target_include_directories(smf PUBLIC /usr/include/suitesparse)
endif()

if(SMF_HAS_METIS)
    target_link_libraries(smf PUBLIC METIS::metis)
endif()

# Propagate feature flags as compile definitions to downstream targets
target_compile_definitions(smf PUBLIC
    $<$<BOOL:${SMF_HAS_AMD}>:SMF_HAS_AMD=1>
    $<$<BOOL:${SMF_HAS_METIS}>:SMF_HAS_METIS=1>
    $<$<BOOL:${SMF_DETERMINISTIC}>:SMF_DETERMINISTIC=1>
    $<$<BOOL:${SMF_HAS_OPENBLAS}>:SMF_HAS_OPENBLAS=1>
    $<$<BOOL:${SMF_USE_MKL}>:SMF_USE_MKL=1>
)

# ---- Optional: detect OpenBLAS thread API --------------------------------
set(SMF_HAS_OPENBLAS OFF)
if(NOT SMF_USE_MKL)
    find_package(OpenBLAS QUIET CONFIG)
    if(NOT OpenBLAS_FOUND)
        # Fallback: look for the cblas header that ships with the pthread variant.
        find_path(OPENBLAS_CBLAS_INCLUDE_DIR cblas.h
            PATHS
                /usr/include/x86_64-linux-gnu/openblas-pthread
                /usr/include/openblas-pthread
                /usr/include/openblas
                /usr/include/x86_64-linux-gnu
                /usr/local/include
        )
        if(OPENBLAS_CBLAS_INCLUDE_DIR)
            set(SMF_HAS_OPENBLAS ON)
            message(STATUS "smf: OpenBLAS cblas.h found at ${OPENBLAS_CBLAS_INCLUDE_DIR} -- SMF_HAS_OPENBLAS=ON")
            target_include_directories(smf PUBLIC ${OPENBLAS_CBLAS_INCLUDE_DIR})
        endif()
    else()
        set(SMF_HAS_OPENBLAS ON)
        message(STATUS "smf: OpenBLAS package found via config -- SMF_HAS_OPENBLAS=ON")
    endif()
endif()


if(SMF_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# ---- IPOPT adapter (optional) -------------------------------------------
if(SMF_BUILD_IPOPT_ADAPTER)
    target_sources(smf PRIVATE src/ipopt_adapter.cpp)
    target_compile_definitions(smf PUBLIC SMF_BUILD_IPOPT_ADAPTER=1)
endif()

# ---- C ABI wrapper (optional) -------------------------------------------
if(SMF_BUILD_C_API)
    target_sources(smf PRIVATE src/smf_c.cpp)
    target_compile_definitions(smf PUBLIC SMF_BUILD_C_API=1)
    if(SMF_BUILD_TESTS)
        add_executable(test_c_api tests/test_c_api.cpp)
        target_link_libraries(test_c_api PRIVATE smf GTest::gtest_main)
        add_test(NAME CApiTest COMMAND test_c_api)
    endif()
endif()
# ---- Benchmarks subdirectory --------------------------------------------
if(SMF_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# ---- Install targets ----------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS smf EXPORT smfTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT smfTargets
    FILE smfTargets.cmake
    NAMESPACE smf::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/smf)

configure_package_config_file(
    cmake/smfConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/smfConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/smf)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/smfConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/smfConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/smfConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/smf)

# ---- Summary ------------------------------------------------------------
message(STATUS "")
message(STATUS "smf ${PROJECT_VERSION} configuration summary")
message(STATUS "  C++ standard  : ${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenMP        : ${SMF_ENABLE_OPENMP}")
message(STATUS "  MKL           : ${SMF_USE_MKL}")
message(STATUS "  AMD found     : ${SMF_HAS_AMD}")
message(STATUS "  METIS found   : ${SMF_HAS_METIS}")
message(STATUS "  CHOLMOD found : ${SMF_HAS_CHOLMOD}")
message(STATUS "  Deterministic : ${SMF_DETERMINISTIC}")
message(STATUS "  Build tests   : ${SMF_BUILD_TESTS}")
message(STATUS "  Build benchmarks: ${SMF_BUILD_BENCHMARKS}")
message(STATUS "  C API wrapper : ${SMF_BUILD_C_API}")
message(STATUS "  CHOLMOD tests : ${SMF_BUILD_CHOLMOD_COMPARE}")
message(STATUS "  Sanitizers    : ${SMF_SANITIZE}")
message(STATUS "")
