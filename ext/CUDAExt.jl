module CUDAExt

# These functions are only defined for diagnostic purposes. Otherwise
# the CUDA extension only relies on the CUDA and cuDNN dependencies to
# have loaded the libraries needed by ONNXRunTime's CUDA execution
# provider.
import CUDA
cuda_functional() = CUDA.functional()
cuda_runtime_version() = CUDA.runtime_version()

end
