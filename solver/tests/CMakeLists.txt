add_executable(test_hello test_hello.cpp)
target_link_libraries(test_hello PRIVATE smf GTest::GTest GTest::Main)
add_test(NAME Smoke.Hello COMMAND test_hello)

add_executable(test_types test_types.cpp)
target_link_libraries(test_types PRIVATE smf GTest::GTest GTest::Main)
add_test(NAME Types COMMAND test_types)

# --- pg:1 test targets ---

add_executable(test_matrix_check test_matrix_check.cpp)
target_link_libraries(test_matrix_check PRIVATE smf GTest::GTest GTest::Main)
add_test(NAME MatrixCheck COMMAND test_matrix_check)

add_executable(test_blas_wrap test_blas_wrap.cpp)
target_link_libraries(test_blas_wrap PRIVATE smf GTest::GTest GTest::Main)
add_test(NAME BlasWrap COMMAND test_blas_wrap)

add_executable(test_arena test_arena.cpp)
target_link_libraries(test_arena PRIVATE smf GTest::GTest GTest::Main)
add_test(NAME Arena COMMAND test_arena)

add_executable(test_cholesky_dense test_cholesky_dense.cpp)
target_link_libraries(test_cholesky_dense PRIVATE smf GTest::GTest GTest::Main)
add_test(NAME CholeskyDense COMMAND test_cholesky_dense)

add_executable(test_ldlt_dense test_ldlt_dense.cpp)
target_link_libraries(test_ldlt_dense PRIVATE smf GTest::GTest GTest::Main)
add_test(NAME LDLTDense COMMAND test_ldlt_dense)

add_executable(test_inertia_dense test_inertia_dense.cpp)
target_link_libraries(test_inertia_dense PRIVATE smf GTest::GTest GTest::Main)
add_test(NAME InertiaDense COMMAND test_inertia_dense)

# --- pg:2 test targets ---

add_executable(test_ordering_amd test_ordering_amd.cpp)
target_link_libraries(test_ordering_amd PRIVATE smf GTest::GTest GTest::Main)
add_test(NAME OrderingAMD COMMAND test_ordering_amd)

add_executable(test_ordering_metis test_ordering_metis.cpp)
target_link_libraries(test_ordering_metis PRIVATE smf GTest::GTest GTest::Main)
add_test(NAME OrderingMETIS COMMAND test_ordering_metis)

add_executable(test_sym_graph test_sym_graph.cpp)
target_link_libraries(test_sym_graph PRIVATE smf GTest::GTest GTest::Main)
add_test(NAME SymGraph COMMAND test_sym_graph)

# --- M2.S1 test target ---

add_executable(test_etree test_etree.cpp)
target_link_libraries(test_etree PRIVATE smf GTest::GTest GTest::Main)
add_test(NAME ETree COMMAND test_etree)

# --- M2.S2 test target ---

add_executable(test_supernode test_supernode.cpp)
target_link_libraries(test_supernode PRIVATE smf GTest::GTest GTest::Main)
add_test(NAME Supernode COMMAND test_supernode)

# --- M2.S3 test target ---

add_executable(test_assembly_tree test_assembly_tree.cpp)
target_link_libraries(test_assembly_tree PRIVATE smf GTest::GTest GTest::Main)
add_test(NAME AssemblyTree COMMAND test_assembly_tree)

# --- M2.S4 test target ---

add_executable(test_symbolic test_symbolic.cpp)
target_link_libraries(test_symbolic PRIVATE smf GTest::GTest GTest::Main)
add_test(NAME Symbolic COMMAND test_symbolic)

# --- pg:3a test targets ---

add_executable(test_frontal_matrix test_frontal_matrix.cpp)
target_link_libraries(test_frontal_matrix PRIVATE smf GTest::GTest GTest::Main)
add_test(NAME FrontalMatrix COMMAND test_frontal_matrix)

# --- M3.B1 test target ---

add_executable(test_factor_stack test_factor_stack.cpp)
target_link_libraries(test_factor_stack PRIVATE smf GTest::gtest_main)
add_test(NAME FactorStack COMMAND test_factor_stack)

add_executable(test_pivoting test_pivoting.cpp)
target_link_libraries(test_pivoting PRIVATE smf GTest::GTest GTest::Main)
add_test(NAME Pivoting COMMAND test_pivoting)

add_executable(test_factor_posdef test_factor_posdef.cpp)
target_link_libraries(test_factor_posdef PRIVATE smf GTest::gtest_main)
add_test(NAME FactorPosdef COMMAND test_factor_posdef)

add_executable(test_factor_indef test_factor_indef.cpp)
target_link_libraries(test_factor_indef PRIVATE smf GTest::gtest_main)
add_test(NAME FactorIndef COMMAND test_factor_indef)

add_executable(test_diagnostics test_diagnostics.cpp)
target_link_libraries(test_diagnostics PRIVATE smf GTest::gtest_main)
add_test(NAME Diagnostics COMMAND test_diagnostics)

add_executable(test_solve_forward test_solve_forward.cpp)
target_link_libraries(test_solve_forward PRIVATE smf GTest::gtest_main)
add_test(NAME SolveForward COMMAND test_solve_forward)

add_executable(test_solve_backward test_solve_backward.cpp)
target_link_libraries(test_solve_backward PRIVATE smf GTest::gtest_main)
add_test(NAME SolveBackward COMMAND test_solve_backward)

add_executable(test_solve_diag test_solve_diag.cpp)
target_link_libraries(test_solve_diag PRIVATE smf GTest::gtest_main)
add_test(NAME SolveDiag COMMAND test_solve_diag)

add_executable(test_solve_end_to_end test_solve_end_to_end.cpp)
target_link_libraries(test_solve_end_to_end PRIVATE smf GTest::gtest_main)
add_test(NAME SolveEndToEnd COMMAND test_solve_end_to_end)

# --- M4.S2 test target ---

add_executable(test_factor_solve test_factor_solve.cpp)
target_link_libraries(test_factor_solve PRIVATE smf GTest::gtest_main)
add_test(NAME FactorSolve COMMAND test_factor_solve)

# --- M5.S1 test target ---

add_executable(test_repeated_factor test_repeated_factor.cpp)
target_link_libraries(test_repeated_factor PRIVATE smf GTest::gtest_main)
add_test(NAME RepeatedFactor COMMAND test_repeated_factor)

# --- M5.S2 test target ---

add_executable(test_singular test_singular.cpp)
target_link_libraries(test_singular PRIVATE smf GTest::gtest_main)
add_test(NAME Singular COMMAND test_singular)

# --- M5.S3 test target ---

add_executable(test_dod_regression test_dod_regression.cpp)
target_link_libraries(test_dod_regression PRIVATE smf GTest::gtest_main)
add_test(NAME DodRegression COMMAND test_dod_regression)
set_tests_properties(DodRegression PROPERTIES LABELS "dod_regression")

# --- M7.A1 test target ---

add_executable(test_scaling_equilib test_scaling_equilib.cpp)
target_link_libraries(test_scaling_equilib PRIVATE smf GTest::gtest_main)
add_test(NAME ScalingEquilib COMMAND test_scaling_equilib)


# --- M6.A1 test target ---

add_executable(test_parallel_factor test_parallel_factor.cpp)
target_link_libraries(test_parallel_factor PRIVATE smf GTest::gtest_main)
add_test(NAME ParallelFactor COMMAND test_parallel_factor)

# --- M6.B1 test target ---

add_executable(test_blas_thread_guard test_blas_thread_guard.cpp)
target_link_libraries(test_blas_thread_guard PRIVATE smf GTest::gtest_main)
add_test(NAME BlasThreadGuard COMMAND test_blas_thread_guard)

# --- M6.G1 test target ---

add_executable(test_parallel_determinism test_parallel_determinism.cpp)
target_link_libraries(test_parallel_determinism PRIVATE smf GTest::gtest_main)
add_test(NAME ParallelDeterminism COMMAND test_parallel_determinism)

# --- M8.S1 IPOPT adapter test (gated) ---

if(SMF_BUILD_IPOPT_ADAPTER)
    add_executable(test_ipopt_adapter test_ipopt_adapter.cpp)
    target_link_libraries(test_ipopt_adapter PRIVATE smf GTest::GTest GTest::Main)
    add_test(NAME IpoptAdapter COMMAND test_ipopt_adapter)
endif()

# --- M9.S1 COO input test ---

add_executable(test_coo_input test_coo_input.cpp)
target_link_libraries(test_coo_input PRIVATE smf GTest::gtest_main)
add_test(NAME CooInput COMMAND test_coo_input)

# --- M9.S3 Sparse forward solve test ---

add_executable(test_sparse_fwd_solve test_sparse_fwd_solve.cpp)
target_link_libraries(test_sparse_fwd_solve PRIVATE smf GTest::gtest_main)
add_test(NAME SparseFwdSolve COMMAND test_sparse_fwd_solve)
