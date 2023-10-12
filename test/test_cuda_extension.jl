# This file is neither included from `runtests.jl` nor run in CI.
#
# Run it with `julia tests/test_cuda_extension.jl`. This requires that
# Julia is installed with juliaup and will involve downloading of a
# lot of big artifacts. The output will contain lots of error messages
# from caught errors; what matters is that all testsets pass.

using Test

juliaup_found = false
try run(pipeline(`juliaup --version`, stdout = devnull, stderr = devnull))
    global juliaup_found = true
catch e
end

if !juliaup_found
    error("`juliaup` needs to be installed for the CUDA extension tests")
end

wait(run(`juliaup add 1.9`, wait = false))

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

@testset "Julia 1.9 CUDA 3" begin
    with_environment(cuda_runtime_version = "11.8") do env
        install_script = """
                         using Pkg
                         Pkg.develop(path = "$(package_path)")
                         Pkg.add(name = "CUDA", version = "3")
                         """
        # CUDA 3 is not possible to install together with ONNXRunTime
        # on Julia 1.9 due to Compat requirements.
        @test_throws ProcessFailedException run(`julia +1.9 --project=$(env) -e "$(install_script)"`)
    end
end

@testset "Julia 1.9 CUDA.jl $(cuda_version) CUDA runtime 11.8" for cuda_version in (4, 5)
    with_environment(cuda_runtime_version = "11.8") do env
        install_script = """
                         using Pkg
                         Pkg.develop(path = "$(package_path)")
                         Pkg.add(name = "CUDA", version = "$(cuda_version)")
                         Pkg.add(name = "cuDNN")
                         """
        @test success(run(`julia +1.9 --project=$(env) -e "$(install_script)"`))
        # Correct dependencies for :cuda.
        test_script = """
                      using ONNXRunTime, CUDA, cuDNN
                      load_inference("$(onnx_path)", execution_provider = :cuda)
                      """
        @test success(run(`julia +1.9 --project=$(env) -e "$(test_script)"`))
        # Neither CUDA nor cuDNN loaded.
        test_script = """
                      using ONNXRunTime
                      load_inference("$(onnx_path)", execution_provider = :cuda)
                      """
        @test_throws ProcessFailedException run(`julia +1.9 --project=$(env) -e "$(test_script)"`)
        # Neither CUDA nor cuDNN loaded but running on CPU, so it's fine.
        test_script = """
                      using ONNXRunTime
                      load_inference("$(onnx_path)", execution_provider = :cpu)
                      """
        # CUDA not loaded. Well, cuDNN pulls in CUDA so this passes anyway.
        test_script = """
                      using ONNXRunTime
                      using cuDNN
                      load_inference("$(onnx_path)", execution_provider = :cuda)
                      """
        @test success(run(`julia +1.9 --project=$(env) -e "$(test_script)"`))
        # CUDA not loaded but running on CPU, so it's fine.
        test_script = """
                      using ONNXRunTime
                      using cuDNN
                      load_inference("$(onnx_path)", execution_provider = :cpu)
                      """
        @test success(run(`julia +1.9 --project=$(env) -e "$(test_script)"`))
        # cuDNN not loaded.
        test_script = """
                      using ONNXRunTime
                      using CUDA
                      load_inference("$(onnx_path)", execution_provider = :cuda)
                      """
        @test_throws ProcessFailedException run(`julia +1.9 --project=$(env) -e "$(test_script)"`)
        # cuDNN not loaded but running on CPU, so it's fine.
        test_script = """
                      using ONNXRunTime
                      using CUDA
                      load_inference("$(onnx_path)", execution_provider = :cpu)
                      """
        @test success(run(`julia +1.9 --project=$(env) -e "$(test_script)"`))
    end
end

@testset "Julia 1.9 CUDA.jl $(cuda_version) CUDA runtime 11.6" for cuda_version in (4, 5)
    with_environment(cuda_runtime_version = "11.6") do env
        install_script = """
                         using Pkg
                         Pkg.develop(path = "$(package_path)")
                         Pkg.add(name = "CUDA", version = "$(cuda_version)")
                         Pkg.add(name = "cuDNN")
                         """
        @test success(run(`julia +1.9 --project=$(env) -e "$(install_script)"`))
        # Correct dependencies for :cuda. CUDA runtime version is
        # lower than officially supported but close enough to at least
        # load so there will be a warning but no error.
        test_script = """
                      using ONNXRunTime, CUDA, cuDNN
                      load_inference("$(onnx_path)", execution_provider = :cuda)
                      """
        @test success(run(`julia +1.9 --project=$(env) -e "$(test_script)"`))
    end
end

@testset "Julia 1.9 CUDA.jl $(cuda_version) CUDA runtime 12.1" for cuda_version in (4, 5)
    with_environment(cuda_runtime_version = "12.1") do env
        install_script = """
                         using Pkg
                         Pkg.develop(path = "$(package_path)")
                         Pkg.add(name = "CUDA", version = "$(cuda_version)")
                         Pkg.add(name = "cuDNN")
                         """
        @test success(run(`julia +1.9 --project=$(env) -e "$(install_script)"`))
        # Correct dependencies for :cuda but fails due to bad version
        # of CUDA runtime.
        test_script = """
                      using ONNXRunTime, CUDA, cuDNN
                      load_inference("$(onnx_path)", execution_provider = :cuda)
                      """
        @test_throws ProcessFailedException run(`julia +1.9 --project=$(env) -e "$(test_script)"`)
    end
end
