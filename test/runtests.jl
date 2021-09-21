include("test_capi.jl")
include("test_highlevel.jl")
try
    import CUDA
    global CUDA_INSTALLED = true
catch
    global CUDA_INSTALLED = false
end
CUDA_FUNCTIONAL = CUDA.functional()
if CUDA_INSTALLED && CUDA_FUNCTIONAL
    @info """
    Found a working CUDA.jl package, running GPU tests
    """
    include("test_cuda.jl")
else
    msg = """
    Skipping CUDA tests. Got
    CUDA_INSTALLED = $CUDA_INSTALLED
    CUDA_FUNCTIONAL = $CUDA_FUNCTIONAL
    """
    @warn msg
end
