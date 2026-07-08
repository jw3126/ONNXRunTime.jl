# This file is neither included from `runtests.jl` nor run in CI.
#
# Run it with `julia test/test_cuda_extension.jl`. This requires that
# Julia is installed with juliaup and will involve downloading of a
# lot of big artifacts and a couple of julia versions. The output will
# contain lots of error messages from caught errors; what matters is
# that all testsets pass.

using Test

juliaup_found = false
try run(pipeline(`juliaup --version`, stdout = devnull, stderr = devnull))
    global juliaup_found = true
catch e
end

if !juliaup_found
    error("`juliaup` needs to be installed for the CUDA extension tests")
end

const tested_julia_versions = ["1.9", "1.10", "1.11", "1.12"]

for version in tested_julia_versions
    run(`juliaup add $version`)
end

package_path = dirname(@__DIR__)
onnx_path = joinpath(@__DIR__, "data", "copy2d.onnx")

function with_environment(f::Function; cuda_runtime_version)
    mktempdir() do env
        write(joinpath(env, "LocalPreferences.toml"),
              """
              [CUDA_Runtime_jll]
              version = "$(cuda_runtime_version)"
              """)
        write(joinpath(env, "Project.toml"),
              """
              [extras]
              CUDA_Runtime_jll = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
              """)
        f(env)
    end
end

@testset "Julia $(julia_version) CUDA.jl $(cuda_version) CUDA runtime 12" for julia_version in tested_julia_versions, cuda_version in 4:6
    # CUDA 6 is only compatible with Julia 1.10 and later.
    julia_version == "1.9" && cuda_version == 6 && continue
    # CUDA 4 is not installable on Julia 1.10 and later.
    VersionNumber(julia_version) > v"1.9" && cuda_version < 5 && continue

    with_environment(cuda_runtime_version = "12") do env
        install_script = """
                         using Pkg
                         Pkg.develop(path = "$(package_path)")
                         Pkg.add(name = "CUDA", version = "$(cuda_version)")
                         Pkg.add(name = "cuDNN")
                         """
        @test success(run(`julia +$(julia_version) --project=$(env) -e "$(install_script)"`))
        # Correct dependencies for :cuda.
        test_script = """
                      using ONNXRunTime, CUDA, cuDNN
                      load_inference("$(onnx_path)", execution_provider = :cuda)
                      """
        @test success(run(`julia +$(julia_version) --project=$(env) -e "$(test_script)"`))
        # Neither CUDA nor cuDNN loaded.
        test_script = """
                      using ONNXRunTime
                      load_inference("$(onnx_path)", execution_provider = :cuda)
                      """
        @test_throws ProcessFailedException run(`julia +$(julia_version) --project=$(env) -e "$(test_script)"`)
        # Neither CUDA nor cuDNN loaded but running on CPU, so it's fine.
        test_script = """
                      using ONNXRunTime
                      load_inference("$(onnx_path)", execution_provider = :cpu)
                      """
        @test success(run(`julia +$(julia_version) --project=$(env) -e "$(test_script)"`))
        # CUDA not loaded. Well, cuDNN pulls in CUDA so this passes anyway.
        # Update: It does on Julia 1.9 but not later.
        test_script = """
                      using ONNXRunTime
                      using cuDNN
                      load_inference("$(onnx_path)", execution_provider = :cuda)
                      """
        if VersionNumber(julia_version) <= v"1.9"
            @test success(run(`julia +$(julia_version) --project=$(env) -e "$(test_script)"`))
        else
            @test_throws ProcessFailedException run(`julia +$(julia_version) --project=$(env) -e "$(test_script)"`)
        end
        # CUDA not loaded but running on CPU, so it's fine.
        test_script = """
                      using ONNXRunTime
                      using cuDNN
                      load_inference("$(onnx_path)", execution_provider = :cpu)
                      """
        @test success(run(`julia +$(julia_version) --project=$(env) -e "$(test_script)"`))
        # cuDNN not loaded.
        test_script = """
                      using ONNXRunTime
                      using CUDA
                      load_inference("$(onnx_path)", execution_provider = :cuda)
                      """
        @test_throws ProcessFailedException run(`julia +$(julia_version) --project=$(env) -e "$(test_script)"`)
        # cuDNN not loaded but running on CPU, so it's fine.
        test_script = """
                      using ONNXRunTime
                      using CUDA
                      load_inference("$(onnx_path)", execution_provider = :cpu)
                      """
        @test success(run(`julia +$(julia_version) --project=$(env) -e "$(test_script)"`))
    end
end
