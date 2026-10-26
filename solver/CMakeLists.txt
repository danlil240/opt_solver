cmake_minimum_required(VERSION 3.20)
project(smf VERSION 0.1.0 LANGUAGES CXX)

# ---- C++ standard -------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- CMake module path & optional local conda/mamba prefix ---------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(_smf_default_conda_lib "")
if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/Library")
    file(TO_CMAKE_PATH "$ENV{CONDA_PREFIX}/Library" _smf_default_conda_lib)
elseif(DEFINED ENV{USERPROFILE} AND EXISTS "$ENV{USERPROFILE}/mamba/envs/smfdeps/Library")
    file(TO_CMAKE_PATH "$ENV{USERPROFILE}/mamba/envs/smfdeps/Library" _smf_default_conda_lib)
endif()
if(_smf_default_conda_lib)
    list(APPEND CMAKE_PREFIX_PATH "${_smf_default_conda_lib}")
endif()

# ---- Options ------------------------------------------------------------
option(SMF_ENABLE_OPENMP       "Enable OpenMP parallelism"              ON)
set(_smf_default_use_mkl OFF)
if(_smf_default_conda_lib AND EXISTS "${_smf_default_conda_lib}/include/mkl.h")
    set(_smf_default_use_mkl ON)
endif()
option(SMF_USE_MKL             "Use Intel MKL instead of OpenBLAS"      ${_smf_default_use_mkl})
option(SMF_USE_METIS           "Enable METIS 5 fill-reducing ordering"  ON)
option(SMF_USE_SUITESPARSE_AMD "Enable SuiteSparse AMD ordering"        ON)
option(SMF_DETERMINISTIC       "Enable deterministic (reproducible) mode" OFF)
option(SMF_BUILD_TESTS         "Build unit tests"                       ON)
option(SMF_BUILD_BENCHMARKS    "Build benchmarks"                       ON)
option(SMF_SANITIZE            "Enable -fsanitize=address,undefined"    OFF)
option(SMF_BUILD_IPOPT_ADAPTER "Build smf IPOPT TSymLinearSolver adapter" OFF)
option(SMF_BUILD_C_API         "Build C ABI wrapper for smf"              OFF)
option(SMF_BUILD_CHOLMOD_COMPARE "Build CHOLMOD comparison tests (ON if CHOLMOD found)" OFF)
option(SMF_BUILD_MA97_PLUGIN   "Build libsmf_ma97.so HSL MA97 plugin (IPOPT dynamic loader)" ON)
option(SMF_REQUIRE_BENCHMARK_BASELINES "Fail configuration/execution when required benchmark baselines are missing" OFF)
set(SMF_REQUIRED_BASELINES "" CACHE STRING "Semicolon-separated list of required baseline solvers when SMF_REQUIRE_BENCHMARK_BASELINES=ON")

# ---- Compiler flags -----------------------------------------------------
if(MSVC)
    add_compile_options(/W4 /EHsc /fp:precise)
    # Vector ISA for the hand-written dense kernels (the GCC/Clang branch
    # uses -march=native). Set to "" for a baseline SSE2 build.
    set(SMF_MSVC_ARCH "AVX2" CACHE STRING "MSVC /arch: value for smf code (empty = compiler default)")
    if(SMF_MSVC_ARCH)
        add_compile_options(/arch:${SMF_MSVC_ARCH})
    endif()
    if(SMF_SANITIZE)
        add_compile_options(/fsanitize=address /Zi)
    else()
        add_compile_options($<$<CONFIG:Release>:/O2>)
    endif()
else()
    add_compile_options(-Wall -Wextra -Wpedantic -fno-fast-math -g)
    if(SMF_SANITIZE)
        add_compile_options(-O1 -fno-omit-frame-pointer -fsanitize=address,undefined)
        add_link_options(-fsanitize=address,undefined)
    else()
        add_compile_options(-O3 -march=native)
    endif()
endif()

if(SMF_DETERMINISTIC)
    add_compile_definitions(SMF_DETERMINISTIC=1)
endif()

# ---- BLAS & LAPACK + backend verification -------------------------------
set(SMF_HAS_OPENBLAS OFF)
set(SMF_BLAS_REFERENCE OFF)
set(SMF_BLAS_BACKEND_NAME "unknown")

# MKL threading layer: "sequential" keeps every MKL call (BLAS, PARDISO)
# single-threaded; "intel" links mkl_intel_thread + libiomp5 so MKL honours
# mkl_set_num_threads / MKL_NUM_THREADS (needed for multi-thread benchmarks).
set(SMF_MKL_THREADING "sequential" CACHE STRING "MKL threading layer: sequential or intel")
set_property(CACHE SMF_MKL_THREADING PROPERTY STRINGS sequential intel)

if(SMF_USE_MKL)
    if(SMF_MKL_THREADING STREQUAL "intel")
        set(_smf_mkl_thread_libs
            "${_smf_default_conda_lib}/lib/mkl_intel_thread_dll.lib"
            "${_smf_default_conda_lib}/lib/libiomp5md.lib")
    else()
        set(_smf_mkl_thread_libs "${_smf_default_conda_lib}/lib/mkl_sequential_dll.lib")
    endif()
    if(NOT BLAS_LIBRARIES AND _smf_default_conda_lib
       AND EXISTS "${_smf_default_conda_lib}/lib/mkl_intel_lp64_dll.lib"
       AND EXISTS "${_smf_default_conda_lib}/lib/mkl_sequential_dll.lib"
       AND EXISTS "${_smf_default_conda_lib}/lib/mkl_core_dll.lib")
        set(BLAS_LIBRARIES
            "${_smf_default_conda_lib}/lib/mkl_intel_lp64_dll.lib"
            ${_smf_mkl_thread_libs}
            "${_smf_default_conda_lib}/lib/mkl_core_dll.lib")
        set(LAPACK_LIBRARIES ${BLAS_LIBRARIES})
        set(BLAS_FOUND TRUE)
        set(LAPACK_FOUND TRUE)
    else()
        set(BLA_VENDOR Intel10_64lp)
        find_package(BLAS REQUIRED)
        find_package(LAPACK REQUIRED)
    endif()
    find_path(MKL_INCLUDE_DIR mkl.h
        HINTS ENV MKLROOT "${_smf_default_conda_lib}/include"
        PATH_SUFFIXES include
        PATHS /opt/intel/oneapi/mkl/latest/include /usr/include/mkl /usr/include)
    if(NOT MKL_INCLUDE_DIR)
        message(FATAL_ERROR "smf: SMF_USE_MKL=ON requested, but mkl.h was not found.")
    endif()
    set(SMF_BLAS_BACKEND_NAME "mkl")
    message(STATUS "smf: Intel MKL verified (include: ${MKL_INCLUDE_DIR}, libs: ${BLAS_LIBRARIES})")
else()
    find_package(BLAS QUIET)
    find_package(LAPACK QUIET)
    if(BLAS_FOUND AND LAPACK_FOUND)
        set(SMF_BLAS_BACKEND_NAME "system")
        find_package(OpenBLAS QUIET CONFIG)
        if(OpenBLAS_FOUND)
            set(SMF_HAS_OPENBLAS ON)
            set(SMF_BLAS_BACKEND_NAME "openblas")
            message(STATUS "smf: OpenBLAS package found via config -- SMF_HAS_OPENBLAS=ON")
        else()
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
                set(SMF_BLAS_BACKEND_NAME "openblas")
                message(STATUS "smf: OpenBLAS cblas.h found at ${OPENBLAS_CBLAS_INCLUDE_DIR} -- SMF_HAS_OPENBLAS=ON")
            endif()
        endif()
    else()
        set(SMF_BLAS_REFERENCE ON)
        set(SMF_BLAS_BACKEND_NAME "reference")
        set(BLAS_LIBRARIES "")
        set(LAPACK_LIBRARIES "")
        message(WARNING "smf: External BLAS/LAPACK not found -- using portable reference BLAS/LAPACK (SMF_BLAS_REFERENCE=1)")
    endif()
endif()

# ---- Optional: OpenMP ---------------------------------------------------
if(SMF_ENABLE_OPENMP)
    find_package(OpenMP QUIET COMPONENTS CXX)
    if(NOT OpenMP_CXX_FOUND)
        message(WARNING "smf: OpenMP not found -- building single-threaded.")
    endif()
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

# ---- Optional: MA27 via CoinHSL (for comparison benchmarks) ----------------
set(SMF_HAS_MA27 OFF)
# COINHSL_ROOT may point at an extracted CoinHSL archive (bin/ lib/ include/).
set(COINHSL_ROOT "" CACHE PATH "Root of an extracted CoinHSL archive")
set(_smf_coinhsl_lib_hints /usr/local/lib)
set(_smf_coinhsl_inc_hints /usr/local/include/coin-or/hsl)
set(_smf_coinhsl_bin_hints "")
if(COINHSL_ROOT)
    list(APPEND _smf_coinhsl_lib_hints "${COINHSL_ROOT}/lib" "${COINHSL_ROOT}/bin")
    list(APPEND _smf_coinhsl_inc_hints "${COINHSL_ROOT}/include")
    set(_smf_coinhsl_bin_hints "${COINHSL_ROOT}/bin")
endif()
# MSVC cannot consume the MinGW .dll.a import library; prefer the MSVC
# import lib generated from coinhsl_msvc.def (coinhsl.lib) when present.
find_library(COINHSL_LIBRARY NAMES coinhsl coinhsl_msvc libcoinhsl
    HINTS ${_smf_coinhsl_lib_hints}
    DOC "CoinHSL library containing MA27")
find_path(COINHSL_INCLUDE_DIR CoinHslConfig.h
    HINTS ${_smf_coinhsl_inc_hints}
    DOC "CoinHSL include directory")
if(COINHSL_LIBRARY)
    set(SMF_HAS_MA27 ON)
    # Locate the MinGW runtime DLLs that libcoinhsl.dll depends on.
    set(SMF_COINHSL_DLL_DIR "" CACHE PATH
        "Directory containing libcoinhsl.dll and its MinGW runtime DLLs")
    if(NOT SMF_COINHSL_DLL_DIR)
        foreach(_d IN LISTS _smf_coinhsl_bin_hints)
            if(EXISTS "${_d}/libcoinhsl.dll")
                set(SMF_COINHSL_DLL_DIR "${_d}")
            endif()
        endforeach()
    endif()
    message(STATUS "smf: CoinHSL/MA27 found at ${COINHSL_LIBRARY} -- SMF_HAS_MA27=ON")
else()
    message(STATUS "smf: CoinHSL not found -- MA27 benchmark disabled")
endif()

# ---- Optional: MUMPS (for comparison benchmarks) -------------------------
set(SMF_HAS_MUMPS OFF)
find_library(DMUMPS_LIBRARY NAMES dmumps_seq dmumps
    HINTS "${_smf_default_conda_lib}/lib" /usr/lib /usr/lib/x86_64-linux-gnu
    DOC "MUMPS double-precision library")
find_path(MUMPS_INCLUDE_DIR dmumps_c.h
    HINTS "${_smf_default_conda_lib}/include" /usr/include
    DOC "MUMPS include directory")
if(DMUMPS_LIBRARY AND MUMPS_INCLUDE_DIR)
    find_library(MUMPS_COMMON_LIBRARY NAMES mumps_common_seq mumps_common
        HINTS "${_smf_default_conda_lib}/lib" /usr/lib /usr/lib/x86_64-linux-gnu)
    find_library(MPISEQ_LIBRARY NAMES mpiseq_seq mpiseq
        HINTS "${_smf_default_conda_lib}/lib" /usr/lib /usr/lib/x86_64-linux-gnu)
    find_library(PORD_LIBRARY NAMES pord
        HINTS "${_smf_default_conda_lib}/lib" /usr/lib /usr/lib/x86_64-linux-gnu)
    set(SMF_HAS_MUMPS ON)
    message(STATUS "smf: MUMPS found at ${DMUMPS_LIBRARY} -- SMF_HAS_MUMPS=ON")
else()
    message(STATUS "smf: MUMPS not found -- MUMPS benchmark disabled")
endif()

# ---- Benchmark baseline adapters: consolidated dependency interface ------
# Adapter sources are compiled into both smf_bench_common (benchmarks/) and
# test_benchmark_adapters (tests/), so discovered baseline includes, libraries
# and activation defines are carried by a single interface target.
add_library(smf_bench_baselines INTERFACE)
if(SMF_HAS_CHOLMOD AND CHOLMOD_LIBRARY AND CHOLMOD_INCLUDE_DIR)
    target_link_libraries(smf_bench_baselines INTERFACE ${CHOLMOD_LIBRARY})
    target_include_directories(smf_bench_baselines INTERFACE ${CHOLMOD_INCLUDE_DIR})
    target_compile_definitions(smf_bench_baselines INTERFACE SMF_HAS_CHOLMOD_IMPL=1)
    # Optional SuiteSparse support libraries (required by static CHOLMOD
    # builds; harmless when CHOLMOD is a shared library).
    foreach(_ss_lib suitesparseconfig amd colamd camd ccolamd)
        find_library(SMF_SS_${_ss_lib}_LIBRARY NAMES ${_ss_lib})
        if(SMF_SS_${_ss_lib}_LIBRARY)
            target_link_libraries(smf_bench_baselines INTERFACE
                ${SMF_SS_${_ss_lib}_LIBRARY})
        endif()
    endforeach()
endif()
if(SMF_HAS_MUMPS AND DMUMPS_LIBRARY AND MUMPS_INCLUDE_DIR)
    set(_smf_mumps_bench_libs ${DMUMPS_LIBRARY})
    if(MUMPS_COMMON_LIBRARY)
        list(APPEND _smf_mumps_bench_libs ${MUMPS_COMMON_LIBRARY})
    endif()
    if(PORD_LIBRARY)
        list(APPEND _smf_mumps_bench_libs ${PORD_LIBRARY})
    endif()
    if(MPISEQ_LIBRARY)
        list(APPEND _smf_mumps_bench_libs ${MPISEQ_LIBRARY})
    endif()
    target_link_libraries(smf_bench_baselines INTERFACE ${_smf_mumps_bench_libs})
    target_include_directories(smf_bench_baselines INTERFACE ${MUMPS_INCLUDE_DIR})
    target_compile_definitions(smf_bench_baselines INTERFACE SMF_HAS_MUMPS_IMPL=1)
endif()
if(SMF_USE_MKL AND MKL_INCLUDE_DIR)
    # MKL PARDISO symbols are exported by mkl_rt, already linked via BLAS.
    target_include_directories(smf_bench_baselines INTERFACE ${MKL_INCLUDE_DIR})
    target_compile_definitions(smf_bench_baselines INTERFACE SMF_HAS_MKL_PARDISO=1)
endif()
if(SMF_HAS_MA27 AND COINHSL_LIBRARY)
    target_link_libraries(smf_bench_baselines INTERFACE ${COINHSL_LIBRARY})
    target_compile_definitions(smf_bench_baselines INTERFACE SMF_HAS_MA27_IMPL=1)
endif()

# ---- Runtime dependency DLL staging (Windows) -----------------------------
# Point SMF_DEPS_DLL_DIR at a directory containing dependency DLLs
# (e.g. a conda env's Library/bin) and they are copied next to executables
# registered via smf_stage_runtime_dlls(target).
set(_smf_default_dll_dir "")
if(_smf_default_conda_lib AND EXISTS "${_smf_default_conda_lib}/bin")
    set(_smf_default_dll_dir "${_smf_default_conda_lib}/bin")
endif()
set(SMF_DEPS_DLL_DIR "${_smf_default_dll_dir}" CACHE PATH
    "Directory containing third-party runtime DLLs to stage beside executables")
function(smf_stage_runtime_dlls target)
    if(NOT (WIN32 AND SMF_DEPS_DLL_DIR AND EXISTS "${SMF_DEPS_DLL_DIR}"))
        return()
    endif()
    file(GLOB _smf_dep_dlls "${SMF_DEPS_DLL_DIR}/mkl_*.dll")
    if(_smf_dep_dlls)
        list(FILTER _smf_dep_dlls EXCLUDE REGEX
            "blacs|scalapack|cdft|tbb_thread")
    endif()
    foreach(_extra
            libiomp5md.dll libomp.dll
            cholmod.dll amd.dll camd.dll colamd.dll ccolamd.dll
            suitesparseconfig.dll metis.dll
            dmumps.dll cmumps.dll smumps.dll zmumps.dll
            libblas.dll liblapack.dll libcblas.dll)
        if(EXISTS "${SMF_DEPS_DLL_DIR}/${_extra}")
            list(APPEND _smf_dep_dlls "${SMF_DEPS_DLL_DIR}/${_extra}")
        endif()
    endforeach()
    # CoinHSL/MA27 ships as a MinGW/gfortran DLL; it needs its own runtime
    # stack (gfortran, openblas, winpthread, ...) staged alongside.
    if(SMF_COINHSL_DLL_DIR AND EXISTS "${SMF_COINHSL_DLL_DIR}")
        foreach(_extra
                libcoinhsl.dll libhsl.dll libopenblas.dll libgfortran-5.dll
                libgfortran-4.dll libgfortran-3.dll libgfortran.dll
                libgomp-1.dll libgcc_s_seh-1.dll libquadmath-0.dll
                libwinpthread-1.dll libstdc++-6.dll libssp-0.dll
                libatomic-1.dll libgcc_s_dw2-1.dll libgcc_s_sjlj-1.dll)
            if(EXISTS "${SMF_COINHSL_DLL_DIR}/${_extra}")
                list(APPEND _smf_dep_dlls "${SMF_COINHSL_DLL_DIR}/${_extra}")
            endif()
        endforeach()
    endif()
    if(_smf_dep_dlls)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${_smf_dep_dlls} "$<TARGET_FILE_DIR:${target}>"
            VERBATIM)
    endif()
endfunction()

# ---- Enforce required benchmark baselines when requested -----------------
if(SMF_REQUIRE_BENCHMARK_BASELINES)
    set(_smf_req_list "${SMF_REQUIRED_BASELINES}")
    if(NOT _smf_req_list)
        set(_smf_req_list "MA27;CHOLMOD;MUMPS")
    endif()
    foreach(_base IN LISTS _smf_req_list)
        string(TOUPPER "${_base}" _base_upper)
        if(_base_upper STREQUAL "MA27" AND NOT SMF_HAS_MA27)
            message(FATAL_ERROR "smf: SMF_REQUIRE_BENCHMARK_BASELINES=ON requires MA27, which was not found.")
        elseif(_base_upper STREQUAL "CHOLMOD" AND NOT SMF_HAS_CHOLMOD)
            message(FATAL_ERROR "smf: SMF_REQUIRE_BENCHMARK_BASELINES=ON requires CHOLMOD, which was not found.")
        elseif(_base_upper STREQUAL "MUMPS" AND NOT SMF_HAS_MUMPS)
            message(FATAL_ERROR "smf: SMF_REQUIRE_BENCHMARK_BASELINES=ON requires MUMPS, which was not found.")
        endif()
    endforeach()
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
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        message(STATUS "smf: System GTest not found -- using in-tree GTest fallback")
        add_library(smf_gtest STATIC
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gtest_fallback/gtest_lib.cpp)
        target_include_directories(smf_gtest PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gtest_fallback/include)
        add_library(smf_gtest_main STATIC
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gtest_fallback/gtest_main.cpp)
        target_link_libraries(smf_gtest_main PUBLIC smf_gtest)
        add_library(GTest::GTest ALIAS smf_gtest)
        add_library(GTest::Main ALIAS smf_gtest_main)
        add_library(GTest::gtest ALIAS smf_gtest)
        add_library(GTest::gtest_main ALIAS smf_gtest_main)
    endif()
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
    src/ordering_matching.cpp
    src/ordering_metis.cpp
    src/sym_graph.cpp
    src/etree.cpp
    src/supernode_detection.cpp
    src/assembly_tree.cpp
    src/symbolic_analysis.cpp
    src/frontal_matrix.cpp
    src/numeric_front.cpp
    src/blocked_indef.cpp
    src/ldlt_kernel.cpp
    src/multifrontal_serial.cpp
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
    src/refinement.cpp
    src/dispatch_policy.cpp
    src/solver.cpp
    src/workspace.cpp
    src/parallel/task_tree.cpp
    src/parallel/blas_thread_guard.cpp
    src/parallel/determinism.cpp
    src/parallel/solve_pool.cpp
    src/coo_to_csc.cpp
    src/solve_sparse_fwd.cpp
    src/matrix_market.cpp
)

target_include_directories(smf PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(SMF_USE_MKL AND MKL_INCLUDE_DIR)
    target_include_directories(smf PUBLIC ${MKL_INCLUDE_DIR})
elseif(SMF_HAS_OPENBLAS AND OPENBLAS_CBLAS_INCLUDE_DIR)
    target_include_directories(smf PUBLIC ${OPENBLAS_CBLAS_INCLUDE_DIR})
endif()

# Required for linking smf into shared libraries (e.g. libsmf_ma97.so)
set_property(TARGET smf PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(smf PUBLIC
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

if(SMF_ENABLE_OPENMP AND TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(smf PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(smf PUBLIC SMF_PARALLEL=1)
    if(MSVC)
        target_compile_options(smf PUBLIC /openmp:llvm)
    endif()
endif()

if(SMF_HAS_AMD)
    target_link_libraries(smf PUBLIC SuiteSparseAMD::amd)
    get_filename_component(_amd_parent_inc "${SuiteSparseAMD_INCLUDE_DIR}" DIRECTORY)
    target_include_directories(smf PUBLIC
        "${SuiteSparseAMD_INCLUDE_DIR}"
        "${_amd_parent_inc}"
        /usr/include/suitesparse)
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
    $<$<BOOL:${SMF_BLAS_REFERENCE}>:SMF_BLAS_REFERENCE=1>
    $<$<BOOL:${SMF_REQUIRE_BENCHMARK_BASELINES}>:SMF_REQUIRE_BENCHMARK_BASELINES=1>
    SMF_BLAS_BACKEND_STR="${SMF_BLAS_BACKEND_NAME}"
)


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

# ---- HSL MA97 plugin (optional) -----------------------------------------
# Builds libsmf_ma97.so: an IPOPT-loadable shared library that implements
# the 7 HSL MA97 C interface functions backed by smf.  After install, use:
#   s_opts["linear_solver"] = "ma97";
#   s_opts["hsllib"]        = "/usr/local/lib/libsmf_ma97.so";
if(SMF_BUILD_MA97_PLUGIN)
    add_library(smf_ma97 SHARED src/smf_ma97_plugin.cpp)
    target_link_libraries(smf_ma97 PRIVATE smf)
    target_include_directories(smf_ma97 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set_target_properties(smf_ma97 PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        SOVERSION 1
        VERSION   1.0.0)
    smf_stage_runtime_dlls(smf_ma97)
    install(TARGETS smf_ma97
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    message(STATUS "smf: MA97 plugin enabled -- will build libsmf_ma97.so")
endif()


message(STATUS "")
message(STATUS "smf ${PROJECT_VERSION} configuration summary")
message(STATUS "  C++ standard  : ${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenMP        : ${SMF_ENABLE_OPENMP}")
message(STATUS "  MKL           : ${SMF_USE_MKL}")
message(STATUS "  AMD found     : ${SMF_HAS_AMD}")
message(STATUS "  METIS found   : ${SMF_HAS_METIS}")
message(STATUS "  CHOLMOD found : ${SMF_HAS_CHOLMOD}")
message(STATUS "  MA27 (CoinHSL): ${SMF_HAS_MA27}")
message(STATUS "  MUMPS         : ${SMF_HAS_MUMPS}")
message(STATUS "  Deterministic : ${SMF_DETERMINISTIC}")
message(STATUS "  Build tests   : ${SMF_BUILD_TESTS}")
message(STATUS "  Build benchmarks: ${SMF_BUILD_BENCHMARKS}")
message(STATUS "  C API wrapper : ${SMF_BUILD_C_API}")
message(STATUS "  MA97 plugin   : ${SMF_BUILD_MA97_PLUGIN}")
message(STATUS "  CHOLMOD tests : ${SMF_BUILD_CHOLMOD_COMPARE}")
message(STATUS "  Sanitizers    : ${SMF_SANITIZE}")
message(STATUS "")
