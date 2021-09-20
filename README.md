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

julia> using ONNXRunTime

julia> path = ONNXRunTime.testdatapath("increment2x3.onnx"); # path to a toy model

julia> model = ONNXRunTime.load_inference(path);

julia> input = Dict("input" => randn(Float32,2,3))
Dict{String, Matrix{Float32}} with 1 entry:
  "input" => [1.68127 1.18192 -0.474021; -1.13518 1.02199 2.75168]

julia> model(input)
Dict{String, Matrix{Float32}} with 1 entry:
  "output" => [2.68127 2.18192 0.525979; -0.135185 2.02199 3.75168]
```

The low level API mirrors the offical [C-API](https://github.com/microsoft/onnxruntime/blob/v1.8.1/include/onnxruntime/core/session/onnxruntime_c_api.h#L347). The above example looks like this:
```julia
using ONNXRunTime

api = GetApi()
env = CreateEnv(api, name="myenv")
so = CreateSessionOptions(api)
path = ONNXRunTime.testdatapath("increment2x3.onnx")
session = CreateSession(api, env, path)
mem = CreateCpuMemoryInfo(api)
input_array = randn(Float32, 2,3)
input_tensor = CreateTensorWithDataAsOrtValue(api, mem, input_array)
run_options = CreateRunOptions(api)
input_names = ["input"]
output_names = ["output"]
inputs = [input_tensor]
outputs = Run(api, session, run_options, input_names, inputs, output_names)
output_tensor = only(outputs)
output_array = GetTensorMutableData(api, output_tensor)
```

# Alternatives
* Use the onnxruntime python bindings via [PyCall.jl](https://github.com/JuliaPy/PyCall.jl). I experienced problems garbage collection with this approach however.
