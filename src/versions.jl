# Version number of the ONNXRunTime library and supported versions of
# the CUDA runtime for GPU processing with the CUDA execution
# provider.
#
# * `onnxruntime_version`: This number must match the version number
#   reported by the ONNXRunTime library, which is verified in the
#   tests. The only real purpose of this variable is to help keep the
#   next one up to date when the library is updated.
#
# * `cuda_runtime_supported_version`: This is the lowest supported
#   version of the ONNX runtime library, which should match the
#   information from
#   https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements
#
# * `cuda_runtime_upper_bound`: The lowest CUDA runtime version which
#   is *not* accepted. Presumably CUDA runtime follows semantic
#   versioning so this can automatically be set to the next major
#   version.
const onnxruntime_version = v"1.20.1"
const cuda_runtime_supported_version = v"12.0"
const cuda_runtime_upper_bound = VersionNumber(cuda_runtime_supported_version.major + 1)
