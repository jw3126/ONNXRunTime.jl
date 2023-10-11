module ONNXRunTime
if !isdefined(Base, :get_extension)
    using Requires: @require
end

function _perm(arr::AbstractArray{T,N}) where {T,N}
    ntuple(i->N+1-i, N)
end
function reversedims(arr)
    permutedims(arr, _perm(arr))
end
function reversedims_lazy(arr)
    PermutedDimsArray(arr, _perm(arr))
end

include("capi.jl")
include("highlevel.jl")

@static if !isdefined(Base, :get_extension)
    function __init__()
        @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" begin
            CUDA.functional() && include("cuda.jl")
        end
    end
end

end #module
