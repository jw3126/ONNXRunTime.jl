module ONNXRunTime

function _perm(arr::AbstractArray{T,N}) where {T,N}
    ntuple(i->N+1-i, N)
end
function reversedims(arr)
    permutedims(arr, _perm(arr))
end
function reversedims_lazy(arr)
    PermutedDimsArray(arr, _perm(arr))
end

include("versions.jl")
include("capi.jl")
include("highlevel.jl")

end #module
