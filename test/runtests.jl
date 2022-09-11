include("test_highlevel.jl")
include("test_capi.jl")

CUDA_INSTALLED = false
CUDA_FUNCTIONAL = false
try
    import CUDA
    global CUDA_INSTALLED = true
    CUDA.versioninfo()
    global CUDA_FUNCTIONAL = true
catch err
    showerror(stderr, err)
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
