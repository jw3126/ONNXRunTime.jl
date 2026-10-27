include("test_versions.jl")
include("test_highlevel.jl")
include("test_capi.jl")

using CUDA: CUDA

if CUDA.functional()
    include("test_cuda.jl")
else
    @warn "CUDA.jl is not functional. Skipping CUDA tests."
end
