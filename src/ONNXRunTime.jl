module ONNXRunTime

using Artifacts
using LazyArtifacts
using ONNXRuntime_jll

using Requires: @require

function _perm(arr::AbstractArray{T,N}) where {T,N}
    ntuple(i->N+1-i, N)
end
function reversedims(arr)
    permutedims(arr, _perm(arr))
end
function reversedims_lazy(arr)
    PermutedDimsArray(arr, _perm(arr))
end

const EXECUTION_PROVIDERS = [:cpu, :cuda]

const artifact_dir_map = Dict{Symbol, String}()

include("../.pkg/platform_augmentation.jl")

function __init__()
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" include("cuda.jl")

    # Workaround/replacement for Artifacts.@artifact_str using the local Artifacts.toml
    function artifact_dir(m::Module, artifact_name::String, p::Platform)
        artifacts_toml = find_artifacts_toml(pathof(m))
        h = artifact_hash(artifact_name, artifacts_toml; platform = p)
        path = artifact_path(h)
        return path
    end

    artifact_dir_map[:cpu] = artifact_dir(ONNXRuntime_jll, "ONNXRuntime", HostPlatform())
    artifact_dir_map[:cuda] = artifact_dir(ONNXRuntime_jll, "ONNXRuntime", augment_platform!(HostPlatform(), "cuda"))
end

include("capi.jl")
include("highlevel.jl")

end #module
