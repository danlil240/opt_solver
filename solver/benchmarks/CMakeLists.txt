cmake_minimum_required(VERSION 3.20)

# Helper macro: add a standalone benchmark executable
macro(smf_benchmark name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE smf)
    if(TARGET Eigen3::Eigen)
        target_link_libraries(${name} PRIVATE Eigen3::Eigen)
        target_compile_definitions(${name} PRIVATE SMF_HAS_EIGEN=1)
    endif()
endmacro()

smf_benchmark(bench_poisson)
smf_benchmark(bench_kkt_ocp)
smf_benchmark(bench_random_symmetric)
smf_benchmark(bench_suite_sparse_matrix_market)
smf_benchmark(bench_compare)
