module ONNXRunTime

include("capi.jl")

function create_memory_info(api, device)
    @argcheck device in DEVICES
    if device === :cpu
        CreateCpuMemoryInfo(api)
    else
        error("TODO")
    end
end
struct InferenceSession1
    api::OrtApi
    device::Symbol
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

function load_inference(path::AbstractString; device=:cpu,
        envname="defaultenv",
                       )::InferenceSession1
    api = GetApi(;device)
    env = CreateEnv(api, name=envname)
    so = CreateSessionOptions(api)
    session = CreateSession(api, env, path)
    meminfo = create_memory_info(api, device)
    allocater = CreateAllocator(api, session, meminfo)
    _input_names =input_names(api, session, allocater)
    _output_names=output_names(api, session, allocater)
    # TODO Is aliasing supported by ONNX? It will cause bugs, so lets forbid it.
    @check allunique(_input_names)
    @check allunique(_output_names)
    return InferenceSession1(api, device, session, meminfo, allocater,
        _input_names,
        _output_names,
    )
end

function (o::InferenceSession1)(inputs::AbstractDict{<:AbstractString, <: AbstractArray},
        output_names::AbstractVector{<:AbstractString} = output_names(o),
    )
    @argcheck o.device in DEVICES
    input_tensors = OrtValue[]
    input_names = String[]
    for key in sort!(collect(keys(inputs)))
        input_array  = inputs[key]
        input_tensor = CreateTensorWithDataAsOrtValue(o.api, o.meminfo, input_array)
        push!(input_tensors, input_tensor)
        push!(input_names, string(key))
    end
    run_options = CreateRunOptions(o.api)
    output_tensors = Run(o.api, o.session, run_options, input_names, input_tensors, output_names)
    Dict((name => GetTensorMutableData(o.api, ten)) for (name, ten) in zip(output_names, output_tensors))
end

end #module
