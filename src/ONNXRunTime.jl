module ONNXRunTime
using Requires:@require
import TimerOutputs

const TIMER = TimerOutputs.TimerOutput()

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

function __init__()
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" include("cuda.jl")
end

end #module
