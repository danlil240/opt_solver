# End-to-end smoke test: stock JuMP + Ipopt.jl using published smf_jll.
# Run with: SMF_PLUGIN_STATS=1 julia packaging/ipopt_jll/test_ipopt_smf.jl

using Test
using JuMP
using Ipopt
import smf_jll

@testset "smf_jll C ABI" begin
    abi = ccall((:smf_c_abi_version, smf_jll.libsmf_c), Cint, ())
    ver = unsafe_string(ccall((:smf_version, smf_jll.libsmf_c), Cstring, ()))
    blas = unsafe_string(ccall((:smf_blas_backend, smf_jll.libsmf_c), Cstring, ()))
    println("[smf_jll] abi=$abi version=$ver blas=$blas")
    @test abi == 1
    @test !isempty(ver)
    @test !isempty(blas)
    @test isfile(smf_jll.libsmf_ma97_path)
end

@testset "Ipopt.jl + smf_jll HS071" begin
    model = Model(Ipopt.Optimizer)
    set_attribute(model, "print_level", 0)
    set_attribute(model, "tol", 1e-9)
    set_attribute(model, "linear_solver", "ma97")
    set_attribute(model, "hsllib", smf_jll.libsmf_ma97_path)

    @variable(model, 1.0 <= x[1:4] <= 5.0)
    set_start_value.(x, [1.0, 5.0, 5.0, 1.0])
    @objective(model, Min, x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3])
    @constraint(model, x[1] * x[2] * x[3] * x[4] >= 25.0)
    @constraint(model, sum(x[i]^2 for i in 1:4) == 40.0)

    optimize!(model)
    @test termination_status(model) in (MOI.LOCALLY_SOLVED, MOI.OPTIMAL)
    @test isapprox(objective_value(model), 17.014017145179; atol=1e-6)
    println("[HS071] obj=$(objective_value(model))")
end

@testset "Ipopt.jl + smf_jll non-convex inertia correction" begin
    n = 40
    model = Model(Ipopt.Optimizer)
    set_attribute(model, "print_level", 0)
    set_attribute(model, "tol", 1e-8)
    set_attribute(model, "linear_solver", "ma97")
    set_attribute(model, "hsllib", smf_jll.libsmf_ma97_path)

    @variable(model, -3.0 <= x[1:n] <= 3.0)
    for i in 1:n
        set_start_value(x[i], isodd(i) ? 0.05 : -0.05)
    end
    @objective(model, Min, sum(-5.0 * x[i]^2 + x[i]^4 for i in 1:n))
    @constraint(model, x[1] + x[2] == 2.0)

    optimize!(model)
    @test termination_status(model) in (MOI.LOCALLY_SOLVED, MOI.OPTIMAL)
    println("[InertiaCorrection40] obj=$(objective_value(model))")
end

println("ALL JULIA IPOPT + SMF_JLL CHECKS PASSED")
