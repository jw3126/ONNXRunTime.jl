module TestVersions

using Test
using ONNXRunTime.CAPI: OrtGetApiBase, GetVersionString
using ONNXRunTime: onnxruntime_version, cuda_runtime_supported_version

# Verify that the ONNXRunTime artifacts are synchronized with the
# information in `src/versions.jl`.
@testset "ONNXRunTime library version" begin
    @test GetVersionString(OrtGetApiBase()) == string(onnxruntime_version)
end

# Verify that the README information about the required CUDA runtime
# version matches `src/versions.jl`. This is difficult to fully
# automate but check that at least all mentions of
# `set_runtime_version!` use the right version.
@testset "Minimum CUDA runtime version in README." begin
    s = read(joinpath(dirname(@__DIR__), "README.md"), String)
    matches = collect(eachmatch(r"set_runtime_version!\(v\"(.+)\"\)", s))
    # If this test fails, the README has changed so much that this
    # testset is outdated and should be updated or removed.
    @test !isempty(matches)

    # If any of these tests fail, update the README information about
    # supported CUDA runtime versions.
    for m in matches
        v = only(m.captures)
        @test VersionNumber(v) == cuda_runtime_supported_version
    end
end

end
