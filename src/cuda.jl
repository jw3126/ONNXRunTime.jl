import .CUDA
CUDA.libcufft()
CUDA.libcudnn()
CUDA.libcurand()
path_cublas = CUDA.libcublas()
using Libdl
# HACK
# CUDA does not yet? expose `libcudart`:
# https://discourse.julialang.org/t/shared-c-dependencies-and-artifacts/68525/4?u=jw3126
# But as an implementation detail libcudart does ship with the artifact
dir = splitdir(path_cublas)[1]
path_cudart = joinpath(dir, "libcudart.so.11.0")
dlopen(path_cudart)
