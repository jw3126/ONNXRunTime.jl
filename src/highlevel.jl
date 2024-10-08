using ArgCheck
using LazyArtifacts
using DataStructures: OrderedDict
using DocStringExtensions
import CEnum
################################################################################
##### testdatapath
################################################################################
function testdatapath(args...)
    joinpath(@__DIR__, "..", "test", "data", args...)
end


using .CAPI
using .CAPI: juliatype, EXECUTION_PROVIDERS
export InferenceSession, load_inference, release

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
    input_names::Vector{String}
    output_names::Vector{String}
end
function Base.show(io::IO, o::InferenceSession)
    print(io,
"""InferenceSession
    input_names:        $(input_names(o))
    output_names:       $(output_names(o))
    execution_provider: $(repr(o.execution_provider))
""")
end

input_names(o::InferenceSession) = o.input_names
output_names(o::InferenceSession) = o.output_names

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

function parse_logging_level(level::Symbol)
    level == :warning && return CAPI.ORT_LOGGING_LEVEL_WARNING
    level == :verbose && return CAPI.ORT_LOGGING_LEVEL_VERBOSE
    level == :info && return CAPI.ORT_LOGGING_LEVEL_INFO
    level == :error && return CAPI.ORT_LOGGING_LEVEL_ERROR
    level == :fatal && return CAPI.ORT_LOGGING_LEVEL_FATAL
    error("""
    $(level) is not a valid value for logging_level. The options are :verbose, :info, :warning (default), :error, and :fatal.
    """)
end

"""
    $TYPEDSIGNATURES

Load an ONNX file at `path` into an inference session.

*Keyword arguments:*
* `execution_provider`: Either `:cpu` or `:cuda`. The latter requires a
  CUDA capable GPU and the `CUDA` and `cuDNN` packages must first be imported.
* `envname`: Name used for logging purposes.
* `logging_level`: Level of diagnostic output. Options are `:verbose`,
  `:info`, `:warning` (default), `:error`, and `:fatal`.
* `provider_options`: Named tuple with options passed to the execution
  provider.

Note: Due to limitations of the C API `CreateEnv` function, `envname`
and `logging_level` can only be set once per process. Attempts to
change these are ignored.
"""
function load_inference(path::AbstractString; execution_provider::Symbol=:cpu,
        envname::AbstractString="defaultenv",
        logging_level::Symbol=:warning,
        provider_options::NamedTuple=(;)
    )::InferenceSession
    api = GetApi(;execution_provider)
    env = CreateEnv(api, name=envname, logging_level=parse_logging_level(logging_level))
    if execution_provider === :cpu
        if !isempty(provider_options)
            error("""
            No provider options are supported for the CPU execution provider.
            """)
        end
        session_options = CreateSessionOptions(api)
    elseif execution_provider === :cuda
        CUDAExt = Base.get_extension(@__MODULE__, :CUDAExt)
        if isnothing(CUDAExt)
            error("""
            The $(repr(execution_provider)) execution provider requires the CUDA.jl and cuDNN.jl packages to be available. Try adding `import CUDA, cuDNN` to your code.
            """)
        elseif !getfield(CUDAExt, :cuda_functional)()
            error("""
            The $(repr(execution_provider)) execution provider requires CUDA to be functional. See `CUDA.functional`.
            """)
        else
            cuda_runtime_version = getfield(CUDAExt, :cuda_runtime_version)()
            if !(cuda_runtime_supported_version <= cuda_runtime_version < cuda_runtime_upper_bound)
                error("""
                Found CUDA runtime version $(cuda_runtime_version). The $(repr(execution_provider)) execution provider requires a CUDA runtime version of at least $(cuda_runtime_supported_version) but less than $(cuda_runtime_upper_bound). See `CUDA.set_runtime_version!` and the package README.
                """)
            end
        end
        session_options = CreateSessionOptions(api)
        cuda_options_dict = Dict{Symbol, Any}(pairs(provider_options))
        if haskey(cuda_options_dict, :cudnn_conv_algo_search)
            # Look up enum values.
            value = cuda_options_dict[:cudnn_conv_algo_search]
            if value ∉ first.(CEnum.name_value_pairs(CAPI.OrtCudnnConvAlgoSearch))
                error("""
                $(value) is not a valid value for :cudnn_conv_algo_search.
                """)
            end
            cuda_options_dict[:cudnn_conv_algo_search] = getfield(CAPI, value)
        end
        cuda_options = OrtCUDAProviderOptions(; cuda_options_dict...)
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
    cstorage = vec(reversedims(arr)::Array)
    CreateTensorWithDataAsOrtValue(o.api, o.meminfo, cstorage, size(arr))
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
    # TODO
    # we can probably optimize this
    # use unsafe_GetTensorMutableData and afterwards set `val.isalive = false` and
    # `empty!(gchandles)`
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
In this case only the outputs whose names are contained in `output_names` are computed.
"""
function (o::InferenceSession)(
                               inputs,
                               output_names=nothing
                              )
    isalive(o) || error("Session has been released and can no longer be called.")
    if output_names === nothing
        output_names = @__MODULE__().output_names(o)
    end
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

"""
    release(o::InferenceSession)::Nothing

Release memory allocated to an [`InferenceSession`](@ref). This also
happens automatically when the object has gone out of scope and the
garbage collector deletes it.

However, there is no guarantee when that happens, so it can be useful
to manually release the memory. This is especially true when the model
has allocated GPU memory, which does not put pressure on the garbage
collector to run promptly.

Using the inference session after releasing is an error.
"""
function release(o::InferenceSession)
    CAPI.release(o.api, o.session)
    CAPI.release(o.api, o.meminfo)
    CAPI.release(o.api, o.allocator)
end

isalive(o::InferenceSession) = all(CAPI.isalive, (o.session, o.meminfo, o.allocator))
