module ONNXRunTime
using LazyArtifacts
using DataStructures: OrderedDict

include("capi.jl")

function create_memory_info(api, execution_provider)
    @argcheck execution_provider in EXECUTION_PROVIDERS
    if execution_provider === :cpu
        CreateCpuMemoryInfo(api)
    else
        error("TODO")
    end
end
struct InferenceSession1
    api::OrtApi
    execution_provider::Symbol
    #env::OrtEnv
    session::OrtSession
    meminfo::OrtMemoryInfo
    allocater::OrtAllocator
    _input_names::Vector{String}
    _output_names::Vector{String}
end
input_names(o::InferenceSession1) = o._input_names
output_names(o::InferenceSession1) = o._output_names

function input_names(api::OrtApi, session::OrtSession, allocater::OrtAllocator)::Vector{String}
    n = SessionGetInputCount(api, session)
    map(0:n-1) do i
        SessionGetInputName(api, session, Csize_t(i), allocater)
    end
end
function output_names(api::OrtApi, session::OrtSession, allocater::OrtAllocator)::Vector{String}
    n = SessionGetOutputCount(api, session)
    map(0:n-1) do i
        SessionGetOutputName(api, session, Csize_t(i), allocater)
    end
end

function load_inference(path::AbstractString; execution_provider=:cpu,
        envname="defaultenv",
                       )::InferenceSession1
    api = GetApi(;execution_provider)
    env = CreateEnv(api, name=envname)
    so = CreateSessionOptions(api)
    session = CreateSession(api, env, path)
    meminfo = create_memory_info(api, execution_provider)
    allocater = CreateAllocator(api, session, meminfo)
    _input_names =input_names(api, session, allocater)
    _output_names=output_names(api, session, allocater)
    # TODO Is aliasing supported by ONNX? It will cause bugs, so lets forbid it.
    @check allunique(_input_names)
    @check allunique(_output_names)
    return InferenceSession1(api, execution_provider, session, meminfo, allocater,
        _input_names,
        _output_names,
    )
end

function make_input_tensor(o::InferenceSession1, inputs, key)
    arr = inputs[keytype(inputs)(key)]
    return CreateTensorWithDataAsOrtValue(o.api, o.meminfo, arr)
end

function prepare_inputs(o::InferenceSession1, inputs)
    names = input_names(o)
    tens = OrtValue[make_input_tensor(o, inputs, key) for key in names]
    names, tens
end

keytype(o) = Base.keytype(o)
keytype(o::NamedTuple) = Symbol

function make_output(o, inputs::NamedTuple, output_names, output_tensors)
    @argcheck length(output_names) == length(output_tensors)
    pairs = (Symbol(key) => GetTensorMutableData(o.api, val) for (key, val) in zip(output_names, output_tensors))
    (;pairs...)
end
function make_output(o, inputs::AbstractDict, output_names, output_tensors)
    @argcheck length(output_names) == length(output_tensors)
    ret = OrderedDict{keytype(inputs), AbstractArray}()
    for (key, val) in zip(output_names, output_tensors)
        ret[key] = GetTensorMutableData(o.api, val)
    end
    ret
end
function (o::InferenceSession1)(
        inputs,
        output_names=output_names(o)
    )
    @argcheck o.execution_provider in EXECUTION_PROVIDERS
    @argcheck eltype(output_names) <: Union{AbstractString, Symbol}
    @argcheck keytype(inputs) <: Union{AbstractString, Symbol}
    expected_input_names = ONNXRunTime.input_names(o)
    for key in keys(inputs)
        if !(String(key) in expected_input_names)
            msg = """
            Invalid input name.
            Expected: $(expected_input_names)
            Got: $(key)
            """
            throw(ArgumentError(msg))
        end
    end
    expected_output_names = ONNXRunTime.output_names(o)
    for name in output_names
        if !(String(name) in expected_output_names)
            msg = """
            Invalid output name.
            Expected: $(expected_output_names)
            Got: $(name)
            """
            throw(ArgumentError(msg))
        end
    end
    inp_names, input_tensors = prepare_inputs(o, inputs)
    run_options    = nothing
    output_tensors = Run(o.api, o.session, run_options, inp_names, input_tensors, output_names)
    make_output(o, inputs, output_names, output_tensors)
end

end #module
