import .CUDA
CUDA.libcufft()
CUDA.libcudnn()
CUDA.libcurand()
path_cublas = CUDA.libcublas()
using Libdl
if isdefined(CUDA, :libcudart)
    CUDA.libcudart()
else
    @warn """
    HACK
    CUDA.jl version not expose `libcudart`. See
    https://discourse.julialang.org/t/shared-c-dependencies-and-artifacts/68525/4?u=jw3126
    But as an implementation detail libcudart does ship with the cublas artifact
    """
    dir = splitdir(path_cublas)[1]
    path_cudart = joinpath(dir, "libcudart.so.11.0")
    dlopen(path_cudart)
end
