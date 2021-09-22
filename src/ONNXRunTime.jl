module ONNXRunTime
using Requires:@require

include("carray.jl")
include("capi.jl")
include("highlevel.jl")

function __init__()
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" include("cuda.jl")
end

end #module
