# ONNXRunTime

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jw3126.github.io/ONNXRunTime.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jw3126.github.io/ONNXRunTime.jl/dev)
[![Build Status](https://github.com/jw3126/ONNXRunTime.jl/workflows/CI/badge.svg)](https://github.com/jw3126/ONNXRunTime.jl/actions)
[![Coverage](https://codecov.io/gh/jw3126/ONNXRunTime.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jw3126/ONNXRunTime.jl)

[ONNXRunTime](https://github.com/jw3126/ONNXRunTime.jl) provides inofficial [julia](https://github.com/JuliaLang/julia) bindings for [onnxruntime](https://github.com/microsoft/onnxruntime).
It exposes both a low level interface, that mirrors the official [C-API](https://github.com/microsoft/onnxruntime/blob/v1.8.1/include/onnxruntime/core/session/onnxruntime_c_api.h#L347), as well as an high level interface.

Contributions are welcome.

# Usage
The high level API works as follows:
```julia

julia> import ONNXRunTime as ORT

julia> path = ORT.testdatapath("increment2x3.onnx"); # path to a toy model

julia> model = ORT.load_inference(path);

julia> input = Dict("input" => randn(Float32,2,3))
Dict{String, Matrix{Float32}} with 1 entry:
  "input" => [1.68127 1.18192 -0.474021; -1.13518 1.02199 2.75168]

julia> model(input)
Dict{String, Matrix{Float32}} with 1 entry:
  "output" => [2.68127 2.18192 0.525979; -0.135185 2.02199 3.75168]
```

For GPU usage the CUDA and cuDNN packages are required and the CUDA
runtime needs to be set to 11.8 or a later 11.x version. To set this
up, do

```julia
pkg> add CUDA cuDNN

julia> import CUDA

julia> CUDA.set_runtime_version!(v"11.8")
```

Then GPU inference is simply

```julia
julia> import CUDA, cuDNN

julia> ORT.load_inference(path, execution_provider=:cuda)
```

CUDA provider options can be specified
```
julia> ORT.load_inference(path, execution_provider=:cuda,
                          provider_options=(;cudnn_conv_algo_search=:HEURISTIC))
```

Memory allocated by a model is eventually automatically released after
it goes out of scope, when the model object is deleted by the garbage
collector. It can also be immediately released with `release(model)`.

The low level API mirrors the offical [C-API](https://github.com/microsoft/onnxruntime/blob/v1.8.1/include/onnxruntime/core/session/onnxruntime_c_api.h#L347). The above example looks like this:
```julia
using ONNXRunTime.CAPI
using ONNXRunTime: testdatapath

api = GetApi();
env = CreateEnv(api, name="myenv");
so = CreateSessionOptions(api);
path = testdatapath("increment2x3.onnx");
session = CreateSession(api, env, path, so);
mem = CreateCpuMemoryInfo(api);
input_array = randn(Float32, 2,3)
input_tensor = CreateTensorWithDataAsOrtValue(api, mem, vec(input_array), size(input_array));
run_options = CreateRunOptions(api);
input_names = ["input"];
output_names = ["output"];
inputs = [input_tensor];
outputs = Run(api, session, run_options, input_names, inputs, output_names);
output_tensor = only(outputs);
output_array = GetTensorMutableData(api, output_tensor);
```

# Alternatives
* Use the onnxruntime python bindings via [PyCall.jl](https://github.com/JuliaPy/PyCall.jl).
* [ONNX.jl](https://github.com/FluxML/ONNX.jl)
* [ONNXNaiveNASflux.jl](https://github.com/DrChainsaw/ONNXNaiveNASflux.jl)

# Complements

* [ONNXLowLevel.jl](https://github.com/GunnarFarneback/ONNXLowLevel.jl) cannot
  run inference but can be used to investigate, create, or manipulate ONNX
  files.

# Breaking Changes in version 0.4.

* Support for CUDA.jl is changed from version 3 to versions 4 and 5.

* Support for Julia versions less than 1.9 is dropped. The reason for
  this is to switch the conditional support of GPUs from being based
  on the Requires package to being a package extension. As a
  consequence the ONNXRunTime GPU support can now be precompiled and
  the CUDA.jl versions can be properly controlled via Compat.

# Setting the CUDA Runtime Version in Tests

For GPU tests using ONNXRunTime, naturally the tests must depend on
and import CUDA and cuDNN. Additionally a supported CUDA runtime
version needs to be used, which can be somewhat tricky to set up for
the tests.

First some background. What `CUDA.set_runtime_version!(v"11.8")`
effectively does is to

1. Add a `LocalPreferences.toml` file containing

```
[CUDA_Runtime_jll]
version = "11.8"
```

2. In `Project.toml`, add
```
[extras]
CUDA_Runtime_jll = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
```

If your test environment is defined by a `test` target in the top
`Project.toml` you need to

1. Add a `LocalPreferences.toml` in your top directory with the same
contents as above.

2. Add `CUDA_Runtime_jll` to the `extras` section of `Project.toml`.

3. Add `CUDA_Runtime_jll` to the `test` target of `Project.toml`.

If your test environment is defined by a `Project.toml` in the `test`
directory, you instead need to

1. Add a `test/LocalPreferences.toml` file with the same contents as
above.

2. Add `CUDA_Runtime_jll` to the `extras` section of `test/Project.toml`.
