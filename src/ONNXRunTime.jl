module ONNXRunTime

function _perm(arr::AbstractArray{T,N}) where {T,N}
    return ntuple(i->N+1-i, N)
end
function reversedims(arr)
    return permutedims(arr, _perm(arr))
end
function reversedims(arr::AbstractArray{T,0}) where {T}
    return arr
end
function reversedims_lazy(arr)
    return PermutedDimsArray(arr, _perm(arr))
end

include("versions.jl")
include("capi.jl")
include("highlevel.jl")

end
