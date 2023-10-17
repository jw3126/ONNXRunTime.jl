include("test_versions.jl")
include("test_highlevel.jl")
include("test_capi.jl")

import CUDA
if CUDA.functional()  
    include("test_cuda.jl")
else
    msg = """
    Skipping CUDA.jl not functional. Skipping CUDA tests.
    """
    @warn msg
end
