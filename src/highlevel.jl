using ArgCheck
using LazyArtifacts
using DataStructures: OrderedDict
using DocStringExtensions
################################################################################
##### testdatapath
################################################################################
function testdatapath(args...)
    joinpath(@__DIR__, "..", "test", "data", args...)
end


using .CAPI
using .CAPI: juliatype, EXECUTION_PROVIDERS
export InferenceSession, load_inference

"""
    $TYPEDEF

Represents an infernence session. Should only be created by calling [`load_inference`](@ref).
"""
struct InferenceSession
    api::OrtApi
    execution_provider::Symbol
    #env::OrtEnv
    session::OrtSession
    meminfo::OrtMemoryInfo
    allocator::OrtAllocator
    _input_names::Vector{String}
    _output_names::Vector{String}
end
input_names(o::InferenceSession) = o._input_names
output_names(o::InferenceSession) = o._output_names

function input_names(api::OrtApi, session::OrtSession, allocator::OrtAllocator)::Vector{String}
    n = SessionGetInputCount(api, session)
    map(0:n-1) do i
        SessionGetInputName(api, session, Csize_t(i), allocator)
    end
end
function output_names(api::OrtApi, session::OrtSession, allocator::OrtAllocator)::Vector{String}
    n = SessionGetOutputCount(api, session)
    map(0:n-1) do i
        SessionGetOutputName(api, session, Csize_t(i), allocator)
    end
end

"""
    $TYPEDSIGNATURES
"""
function load_inference(path::AbstractString; execution_provider::Symbol=:cpu,
        envname::AbstractString="defaultenv",
                       )::InferenceSession
    api = GetApi(;execution_provider)
    env = CreateEnv(api, name=envname)
    if execution_provider === :cpu
        session_options = CreateSessionOptions(api)
    elseif execution_provider === :cuda
        if !(isdefined(@__MODULE__, :CUDA))
            @warn """
            The $(repr(execution_provider)) requires the CUDA.jl package to be available. Try adding `import CUDA` to your code.
            """
        end
        session_options = CreateSessionOptions(api)
        cuda_options = OrtCUDAProviderOptions()
        SessionOptionsAppendExecutionProvider_CUDA(api, session_options, cuda_options)
    else
        error("Unsupported execution_provider $execution_provider")
    end
    session = CreateSession(api, env, path, session_options)
    meminfo = CreateCpuMemoryInfo(api)
    allocator = CreateAllocator(api, session, meminfo)
    _input_names =input_names(api, session, allocator)
    _output_names=output_names(api, session, allocator)
    # TODO Is aliasing supported by ONNX? It will cause bugs, so lets forbid it.
    @check allunique(_input_names)
    @check allunique(_output_names)
    return InferenceSession(api, execution_provider, session, meminfo, allocator,
        _input_names,
        _output_names,
    )
end

function make_input_tensor(o::InferenceSession, inputs, key)
    arr = inputs[keytype(inputs)(key)]
    return CreateTensorWithDataAsOrtValue(o.api, o.meminfo, arr)
end

function prepare_inputs(o::InferenceSession, inputs)
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

"""
    (o::InferenceSession)(inputs [,output_names])

Run an [`InferenceSession`](@ref) on a collection of inputs. Here `inputs` can either
be a `NamedTuple` or an `AbstractDict`. Optionally `output_names` can be passed.
In this case only the outputs whose name is contained in `output_names` are computed.
"""
function (o::InferenceSession)(
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
