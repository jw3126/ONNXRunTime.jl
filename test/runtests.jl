include("test_carray.jl")
include("test_highlevel.jl")
include("test_capi.jl")

CUDA_INSTALLED = false
CUDA_FUNCTIONAL = false
try
    import CUDA
    global CUDA_INSTALLED = true
    global CUDA_FUNCTIONAL = CUDA.functional()
catch
end
if CUDA_INSTALLED && CUDA_FUNCTIONAL
    @info """
    Found a working CUDA.jl package, running CUDA tests
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
