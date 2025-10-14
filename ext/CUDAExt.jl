module CUDAExt
import ONNXRunTime
import CUDA

# These calls are only being made for diagnostic purposes. Otherwise
# the CUDA extension only relies on the CUDA and cuDNN dependencies to
# have loaded the libraries needed by ONNXRunTime's CUDA execution
# provider.
function __init__()
    ONNXRunTime.cuda_is_loaded[] = true
    ONNXRunTime.cuda_is_functional[] = CUDA.functional()
    ONNXRunTime.cuda_runtime_version[] = CUDA.runtime_version()
end

end
