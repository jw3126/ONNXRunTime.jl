"""
module CAPI

This module closely follows the offical onnxruntime [C-API](https://github.com/microsoft/onnxruntime/blob/v1.8.1/include/onnxruntime/core/session/onnxruntime_c_api.h#L347).
See [here](https://github.com/microsoft/onnxruntime-inference-examples/blob/d031f879c9a8d33c8b7dc52c5bc65fe8b9e3960d/c_cxx/fns_candy_style_transfer/fns_candy_style_transfer.c) for a C code example.
"""
module CAPI

using DocStringExtensions
using Libdl
using CEnum: @cenum
using ArgCheck
using Pkg.Artifacts: @artifact_str

const LIB_CPU  = Ref(C_NULL)
const LIB_CUDA = Ref(C_NULL)

const EXECUTION_PROVIDERS = [:cpu, :cuda]

function set_lib!(path::AbstractString, execution_provider::Symbol)
    @argcheck ispath(path)
    LIB = libref(execution_provider)
    if LIB[] != C_NULL
        dlclose(LIB[])
    end
    LIB[] = dlopen(path)
end

function make_lib!(execution_provider)
    @argcheck execution_provider in EXECUTION_PROVIDERS
    root = if execution_provider === :cpu
        artifact"onnxruntime_cpu"
    elseif execution_provider === :cuda
        artifact"onnxruntime_gpu"
    else
        error("Unknown execution_provider $(repr(execution_provider))")
    end
    @check isdir(root)
    dir = joinpath(root, only(readdir(root)))
    @check isdir(dir)
    ext = if Sys.iswindows()
        ".dll"
    elseif Sys.isapple()
        ".dylib"
    else
        ".so"
    end
    path = joinpath(dir, "lib", "libonnxruntime" * ext)
    set_lib!(path, execution_provider)
end

function libref(execution_provider::Symbol)::Ref
    @argcheck execution_provider in EXECUTION_PROVIDERS
    if execution_provider === :cpu
        LIB_CPU
    elseif execution_provider === :cuda
        LIB_CUDA
    else
        error("Unreachable $(repr(execution_provider))")
    end
end

function libptr(execution_provider::Symbol)::Ptr
    ref = libref(execution_provider)
    if ref[] == C_NULL
        make_lib!(execution_provider)
    end
    return ref[]
end

function unsafe_load(ptr::Ptr)
    if ptr == C_NULL
        error("unsafe_load from NULL: $ptr")
    else
        Base.unsafe_load(ptr)
    end
end

struct OrtException <: Exception
    msg::String
end
function Base.showerror(io::IO, err::OrtException)
    println(io, err.msg)
end



################################################################################
##### OrtApi
################################################################################
"""
    $(TYPEDEF)
"""
struct OrtApiBase
    GetApi::Ptr{Cvoid}
    GetVersionString::Ptr{Cvoid}
    # a global constant, never released
end

"""
    $(TYPEDEF)
"""
struct OrtApi
    CreateStatus::Ptr{Cvoid}
    GetErrorCode::Ptr{Cvoid}
    GetErrorMessage::Ptr{Cvoid}
    CreateEnv::Ptr{Cvoid}
    CreateEnvWithCustomLogger::Ptr{Cvoid}
    EnableTelemetryEvents::Ptr{Cvoid}
    DisableTelemetryEvents::Ptr{Cvoid}
    CreateSession::Ptr{Cvoid}
    CreateSessionFromArray::Ptr{Cvoid}
    Run::Ptr{Cvoid}
    CreateSessionOptions::Ptr{Cvoid}
    SetOptimizedModelFilePath::Ptr{Cvoid}
    CloneSessionOptions::Ptr{Cvoid}
    SetSessionExecutionMode::Ptr{Cvoid}
    EnableProfiling::Ptr{Cvoid}
    DisableProfiling::Ptr{Cvoid}
    EnableMemPattern::Ptr{Cvoid}
    DisableMemPattern::Ptr{Cvoid}
    EnableCpuMemArena::Ptr{Cvoid}
    DisableCpuMemArena::Ptr{Cvoid}
    SetSessionLogId::Ptr{Cvoid}
    SetSessionLogVerbosityLevel::Ptr{Cvoid}
    SetSessionLogSeverityLevel::Ptr{Cvoid}
    SetSessionGraphOptimizationLevel::Ptr{Cvoid}
    SetIntraOpNumThreads::Ptr{Cvoid}
    SetInterOpNumThreads::Ptr{Cvoid}
    CreateCustomOpDomain::Ptr{Cvoid}
    CustomOpDomain_Add::Ptr{Cvoid}
    AddCustomOpDomain::Ptr{Cvoid}
    RegisterCustomOpsLibrary::Ptr{Cvoid}
    SessionGetInputCount::Ptr{Cvoid}
    SessionGetOutputCount::Ptr{Cvoid}
    SessionGetOverridableInitializerCount::Ptr{Cvoid}
    SessionGetInputTypeInfo::Ptr{Cvoid}
    SessionGetOutputTypeInfo::Ptr{Cvoid}
    SessionGetOverridableInitializerTypeInfo::Ptr{Cvoid}
    SessionGetInputName::Ptr{Cvoid}
    SessionGetOutputName::Ptr{Cvoid}
    SessionGetOverridableInitializerName::Ptr{Cvoid}
    CreateRunOptions::Ptr{Cvoid}
    RunOptionsSetRunLogVerbosityLevel::Ptr{Cvoid}
    RunOptionsSetRunLogSeverityLevel::Ptr{Cvoid}
    RunOptionsSetRunTag::Ptr{Cvoid}
    RunOptionsGetRunLogVerbosityLevel::Ptr{Cvoid}
    RunOptionsGetRunLogSeverityLevel::Ptr{Cvoid}
    RunOptionsGetRunTag::Ptr{Cvoid}
    RunOptionsSetTerminate::Ptr{Cvoid}
    RunOptionsUnsetTerminate::Ptr{Cvoid}
    CreateTensorAsOrtValue::Ptr{Cvoid}
    CreateTensorWithDataAsOrtValue::Ptr{Cvoid}
    IsTensor::Ptr{Cvoid}
    GetTensorMutableData::Ptr{Cvoid}
    FillStringTensor::Ptr{Cvoid}
    GetStringTensorDataLength::Ptr{Cvoid}
    GetStringTensorContent::Ptr{Cvoid}
    CastTypeInfoToTensorInfo::Ptr{Cvoid}
    GetOnnxTypeFromTypeInfo::Ptr{Cvoid}
    CreateTensorTypeAndShapeInfo::Ptr{Cvoid}
    SetTensorElementType::Ptr{Cvoid}
    SetDimensions::Ptr{Cvoid}
    GetTensorElementType::Ptr{Cvoid}
    GetDimensionsCount::Ptr{Cvoid}
    GetDimensions::Ptr{Cvoid}
    GetSymbolicDimensions::Ptr{Cvoid}
    GetTensorShapeElementCount::Ptr{Cvoid}
    GetTensorTypeAndShape::Ptr{Cvoid}
    GetTypeInfo::Ptr{Cvoid}
    GetValueType::Ptr{Cvoid}
    CreateMemoryInfo::Ptr{Cvoid}
    CreateCpuMemoryInfo::Ptr{Cvoid}
    CompareMemoryInfo::Ptr{Cvoid}
    MemoryInfoGetName::Ptr{Cvoid}
    MemoryInfoGetId::Ptr{Cvoid}
    MemoryInfoGetMemType::Ptr{Cvoid}
    MemoryInfoGetType::Ptr{Cvoid}
    AllocatorAlloc::Ptr{Cvoid}
    AllocatorFree::Ptr{Cvoid}
    AllocatorGetInfo::Ptr{Cvoid}
    GetAllocatorWithDefaultOptions::Ptr{Cvoid}
    AddFreeDimensionOverride::Ptr{Cvoid}
    GetValue::Ptr{Cvoid}
    GetValueCount::Ptr{Cvoid}
    CreateValue::Ptr{Cvoid}
    CreateOpaqueValue::Ptr{Cvoid}
    GetOpaqueValue::Ptr{Cvoid}
    KernelInfoGetAttribute_float::Ptr{Cvoid}
    KernelInfoGetAttribute_int64::Ptr{Cvoid}
    KernelInfoGetAttribute_string::Ptr{Cvoid}
    KernelContext_GetInputCount::Ptr{Cvoid}
    KernelContext_GetOutputCount::Ptr{Cvoid}
    KernelContext_GetInput::Ptr{Cvoid}
    KernelContext_GetOutput::Ptr{Cvoid}
    ReleaseEnv::Ptr{Cvoid}
    ReleaseStatus::Ptr{Cvoid}
    ReleaseMemoryInfo::Ptr{Cvoid}
    ReleaseSession::Ptr{Cvoid}
    ReleaseValue::Ptr{Cvoid}
    ReleaseRunOptions::Ptr{Cvoid}
    ReleaseTypeInfo::Ptr{Cvoid}
    ReleaseTensorTypeAndShapeInfo::Ptr{Cvoid}
    ReleaseSessionOptions::Ptr{Cvoid}
    ReleaseCustomOpDomain::Ptr{Cvoid}
    GetDenotationFromTypeInfo::Ptr{Cvoid}
    CastTypeInfoToMapTypeInfo::Ptr{Cvoid}
    CastTypeInfoToSequenceTypeInfo::Ptr{Cvoid}
    GetMapKeyType::Ptr{Cvoid}
    GetMapValueType::Ptr{Cvoid}
    GetSequenceElementType::Ptr{Cvoid}
    ReleaseMapTypeInfo::Ptr{Cvoid}
    ReleaseSequenceTypeInfo::Ptr{Cvoid}
    SessionEndProfiling::Ptr{Cvoid}
    SessionGetModelMetadata::Ptr{Cvoid}
    ModelMetadataGetProducerName::Ptr{Cvoid}
    ModelMetadataGetGraphName::Ptr{Cvoid}
    ModelMetadataGetDomain::Ptr{Cvoid}
    ModelMetadataGetDescription::Ptr{Cvoid}
    ModelMetadataLookupCustomMetadataMap::Ptr{Cvoid}
    ModelMetadataGetVersion::Ptr{Cvoid}
    ReleaseModelMetadata::Ptr{Cvoid}
    CreateEnvWithGlobalThreadPools::Ptr{Cvoid}
    DisablePerSessionThreads::Ptr{Cvoid}
    CreateThreadingOptions::Ptr{Cvoid}
    ReleaseThreadingOptions::Ptr{Cvoid}
    ModelMetadataGetCustomMetadataMapKeys::Ptr{Cvoid}
    AddFreeDimensionOverrideByName::Ptr{Cvoid}
    GetAvailableProviders::Ptr{Cvoid}
    ReleaseAvailableProviders::Ptr{Cvoid}
    GetStringTensorElementLength::Ptr{Cvoid}
    GetStringTensorElement::Ptr{Cvoid}
    FillStringTensorElement::Ptr{Cvoid}
    AddSessionConfigEntry::Ptr{Cvoid}
    CreateAllocator::Ptr{Cvoid}
    ReleaseAllocator::Ptr{Cvoid}
    RunWithBinding::Ptr{Cvoid}
    CreateIoBinding::Ptr{Cvoid}
    ReleaseIoBinding::Ptr{Cvoid}
    BindInput::Ptr{Cvoid}
    BindOutput::Ptr{Cvoid}
    BindOutputToDevice::Ptr{Cvoid}
    GetBoundOutputNames::Ptr{Cvoid}
    GetBoundOutputValues::Ptr{Cvoid}
    ClearBoundInputs::Ptr{Cvoid}
    ClearBoundOutputs::Ptr{Cvoid}
    TensorAt::Ptr{Cvoid}
    CreateAndRegisterAllocator::Ptr{Cvoid}
    SetLanguageProjection::Ptr{Cvoid}
    SessionGetProfilingStartTimeNs::Ptr{Cvoid}
    SetGlobalIntraOpNumThreads::Ptr{Cvoid}
    SetGlobalInterOpNumThreads::Ptr{Cvoid}
    SetGlobalSpinControl::Ptr{Cvoid}
    AddInitializer::Ptr{Cvoid}
    CreateEnvWithCustomLoggerAndGlobalThreadPools::Ptr{Cvoid}
    SessionOptionsAppendExecutionProvider_CUDA::Ptr{Cvoid}
    SessionOptionsAppendExecutionProvider_ROCM::Ptr{Cvoid}
    SessionOptionsAppendExecutionProvider_OpenVINO::Ptr{Cvoid}
    SetGlobalDenormalAsZero::Ptr{Cvoid}
    CreateArenaCfg::Ptr{Cvoid}
    ReleaseArenaCfg::Ptr{Cvoid}
    ModelMetadataGetGraphDescription::Ptr{Cvoid}
    SessionOptionsAppendExecutionProvider_TensorRT::Ptr{Cvoid}
    SetCurrentGpuDeviceId::Ptr{Cvoid}
    GetCurrentGpuDeviceId::Ptr{Cvoid}
    KernelInfoGetAttributeArray_float::Ptr{Cvoid}
    KernelInfoGetAttributeArray_int64::Ptr{Cvoid}
    CreateArenaCfgV2::Ptr{Cvoid}
    AddRunConfigEntry::Ptr{Cvoid}
    CreatePrepackedWeightsContainer::Ptr{Cvoid}
    ReleasePrepackedWeightsContainer::Ptr{Cvoid}
    CreateSessionWithPrepackedWeightsContainer::Ptr{Cvoid}
    CreateSessionFromArrayWithPrepackedWeightsContainer::Ptr{Cvoid}
end

const OrtStatusPtr = Ptr{Cvoid}

"""
    $TYPEDSIGNATURES
"""
function OrtGetApiBase(; execution_provider::Symbol = :cpu)::OrtApiBase
    @argcheck execution_provider in EXECUTION_PROVIDERS
    f = dlsym(libptr(execution_provider), :OrtGetApiBase)
    api_base = unsafe_load(@ccall $f()::Ptr{OrtApiBase})
end
"""
    $TYPEDSIGNATURES
"""
function GetVersionString(api_base::OrtApiBase)::String
    return unsafe_string(@ccall $(api_base.GetVersionString)()::Cstring)
end

const ORT_API_VERSION = 8
"""
    $TYPEDSIGNATURES
"""
function GetApi(api_base::OrtApiBase, ort_api_version::Integer = ORT_API_VERSION)::OrtApi
    ptr = @ccall $(api_base.GetApi)(ort_api_version::UInt32)::Ptr{OrtApi}
    unsafe_load(ptr)
end
GetApi(; execution_provider = :cpu) = GetApi(OrtGetApiBase(; execution_provider))

################################################################################
##### OrtEnv
################################################################################

for item in [
    (name=:Env                       , release=true),
    (name=:Status                    , release=true),
    (name=:MemoryInfo                , release=true),
    (name=:IoBinding                 , release=true),
    (name=:Session                   , release=true),
    (name=:Value                     , release=true),
    (name=:RunOptions                , release=true),
    (name=:TypeInfo                  , release=true),
    (name=:TensorTypeAndShapeInfo    , release=true),
    (name=:SessionOptions            , release=true),
    (name=:CustomOpDomain            , release=true),
    (name=:MapTypeInfo               , release=true),
    (name=:SequenceTypeInfo          , release=true),
    (name=:ModelMetadata             , release=true),
    #(name=:ThreadPoolParams          , release=true),
    (name=:ThreadingOptions          , release=true),
    (name=:PrepackedWeightsContainer , release=true),
    (name=:Allocator                 , release=true),
    (name=:ArenaCfg                  , release=true),
    #(name=:CUDAProviderOptions       , release=false),
]
    Obj = item.name
    OrtObj = Symbol(:Ort, Obj)
    ReleaseObj = Symbol(:Release, Obj)
    @eval begin
        """
            $(string($OrtObj))

        Wraps a pointer to the C object of type $(string($OrtObj)).
        """
        mutable struct $OrtObj
            ptr::Ptr{Cvoid}
            gchandles::Vector{Any}
            isalive::Bool
        end
    end
    if item.release
        if !(ReleaseObj in fieldnames(OrtApi))
            error("$ReleaseObj not in fieldnames(OrtApi)")
        end
        @eval function $ReleaseObj(api::OrtApi, obj::$OrtObj)::Nothing
            if obj.isalive
                f = api.$ReleaseObj
                ccall(f, Cvoid, (Ptr{Cvoid},), obj.ptr)
            end
            empty!(obj.gchandles)
            nothing
        end
        @eval function release(api::OrtApi, obj::$OrtObj)::Nothing
            $ReleaseObj(api, obj)
        end
    end
    @eval export $OrtObj
end

"""
    release(api::OrtApi, obj)::Nothing

Release memory owned by `obj`. The garbage collector should call this
function automatically. If it does not that's a bug that should be reported.

There might however be situations with high memory pressure. In these situations it might
help to call this function manually to release memory earlier. Using an object after releasing it is undefined behaviour.
"""
function release end

"""
    $TYPEDSIGNATURES

Create a julia object from the output of an api call. Check and release status_ptr.
"""
function into_julia(
    ::Type{T},
    api::OrtApi,
    objptr::Ref{Ptr{Cvoid}},
    status_ptr::Ptr{Cvoid},
    gchandles,
)::T where {T}
    check_and_release(api, status_ptr)
    ptr = objptr[]
    if ptr == C_NULL
        error("Unexpected Null ptr")
    end
    alive = true
    ret = T(ptr, gchandles, alive)
    finalizer(ret) do obj
        release(api, obj)
    end
    return ret
end

"""
    $TYPEDEF
"""
@cenum OrtLoggingLevel::UInt32 begin
    ORT_LOGGING_LEVEL_VERBOSE = 0
    ORT_LOGGING_LEVEL_INFO = 1
    ORT_LOGGING_LEVEL_WARNING = 2
    ORT_LOGGING_LEVEL_ERROR = 3
    ORT_LOGGING_LEVEL_FATAL = 4
end


"""
    $TYPEDSIGNATURES
"""
function GetErrorMessage(api::OrtApi, status::OrtStatusPtr)::String
    @argcheck status isa Ptr
    @argcheck status != C_NULL
    s = @ccall $(api.GetErrorMessage)(status::Ptr{Cvoid})::Cstring
    unsafe_string(s)
end
function check_and_release(api, status::OrtStatusPtr)::Nothing
    if status != C_NULL
        msg = GetErrorMessage(api, status)
        _release_status_ptr(api, status)
        throw(OrtException(msg))
    end
    return nothing
end
function _release_status_ptr(api::OrtApi, ptr::OrtStatusPtr)
    @ccall $(api.ReleaseStatus)(ptr::Ptr{Cvoid})::Cvoid
end

"""
    $TYPEDSIGNATURES
"""
function CreateEnv(
    api::OrtApi;
    logging_level::OrtLoggingLevel = ORT_LOGGING_LEVEL_WARNING,
    name::AbstractString,
)::OrtEnv
    p_ptr = Ref(C_NULL)
    gchandles = Any[api, name]
    status = @ccall $(api.CreateEnv)(
        logging_level::OrtLoggingLevel,
        name::Cstring,
        p_ptr::Ptr{Ptr{Cvoid}},
    )::OrtStatusPtr
    into_julia(OrtEnv, api, p_ptr, status, gchandles)
end

################################################################################
##### SessionOptions
################################################################################

"""
    $TYPEDSIGNATURES
"""
function CreateSessionOptions(api::OrtApi)::OrtSessionOptions
    p_ptr = Ref(C_NULL)
    status = @ccall $(api.CreateSessionOptions)(p_ptr::Ptr{Ptr{Cvoid}})::OrtStatusPtr
    gchandles = Any[api]
    into_julia(OrtSessionOptions, api, p_ptr, status, gchandles)
end

################################################################################
##### Session
################################################################################
"""
    $TYPEDSIGNATURES
"""
function CreateSession(
    api::OrtApi,
    env::OrtEnv,
    path::AbstractString,
    options::OrtSessionOptions,
)::OrtSession
    @argcheck ispath(path)
    p_ptr = Ref(C_NULL)
    gchandles = Any[api, env, path, options]
    status = @ccall $(api.CreateSession)(
        env.ptr::Ptr{Cvoid},
        path::Cstring,
        options.ptr::Ptr{Cvoid},
        p_ptr::Ptr{Ptr{Cvoid}},
    )::OrtStatusPtr
    into_julia(OrtSession, api, p_ptr, status, gchandles)
end

"""
    $TYPEDSIGNATURES
"""
function SessionGetInputCount(api::OrtApi, session::OrtSession)::Csize_t
    out = Ref{Csize_t}()
    GC.@preserve session begin
        status = @ccall $(api.SessionGetInputCount)(
            session.ptr::Ptr{Cvoid},
            out::Ptr{Csize_t},
        )::OrtStatusPtr
    end
    check_and_release(api, status)
    return out[]
end

"""
    $TYPEDSIGNATURES
"""
function CreateAllocator(
    api::OrtApi,
    session::OrtSession,
    meminfo::OrtMemoryInfo,
)::OrtAllocator
    p_ptr = Ref(C_NULL)
    gchandles = Any[api, session, meminfo]
    status = @ccall $(api.CreateAllocator)(
        session.ptr::Ptr{Cvoid},
        meminfo.ptr::Ptr{Cvoid},
        p_ptr::Ptr{Ptr{Cvoid}},
    )::OrtStatusPtr
    into_julia(OrtAllocator, api, p_ptr, status, gchandles)
end

"""
    $TYPEDSIGNATURES
"""
function AllocatorFree(api::OrtApi, allocator::OrtAllocator, ptr::Ptr)
    GC.@preserve api allocator begin
        status = @ccall $(api.AllocatorFree)(
            allocator.ptr::Ptr{Cvoid},
            ptr::Ptr{Cvoid},
        )::OrtStatusPtr
        check_and_release(api, status)
    end
end

"""
    $TYPEDSIGNATURES
"""
function SessionGetInputName(
    api::OrtApi,
    session::OrtSession,
    index::Integer,
    allocator::OrtAllocator,
)::String
    p_ptr = Ref{Cstring}(C_NULL)
    GC.@preserve api session allocator begin
        status = @ccall $(api.SessionGetInputName)(
            session.ptr::Ptr{Cvoid},
            index::Csize_t,
            allocator.ptr::Ptr{Cvoid},
            p_ptr::Ptr{Cstring},
        )::OrtStatusPtr
        check_and_release(api, status)
        ret = unsafe_string(p_ptr[])
        AllocatorFree(api, allocator, pointer(p_ptr[]))
        return ret
    end
end

"""
    $TYPEDSIGNATURES
"""
function SessionGetOutputName(
    api::OrtApi,
    session::OrtSession,
    index::Integer,
    allocator::OrtAllocator,
)::String
    p_ptr = Ref{Cstring}(C_NULL)
    GC.@preserve api session allocator begin
        status = @ccall $(api.SessionGetOutputName)(
            session.ptr::Ptr{Cvoid},
            index::Csize_t,
            allocator.ptr::Ptr{Cvoid},
            p_ptr::Ptr{Cstring},
        )::OrtStatusPtr
        check_and_release(api, status)
        ret = unsafe_string(p_ptr[])
        AllocatorFree(api, allocator, pointer(p_ptr[]))
        return ret
    end
end

"""
    $TYPEDSIGNATURES
"""
function SessionGetOutputCount(api::OrtApi, sess::OrtSession)::Csize_t
    out = Ref{Csize_t}()
    GC.@preserve sess begin
        status = @ccall $(api.SessionGetOutputCount)(
            sess.ptr::Ptr{Cvoid},
            out::Ptr{Csize_t},
        )::OrtStatusPtr
    end
    check_and_release(api, status)
    return out[]
end

################################################################################
##### OrtMemoryInfo
################################################################################

"""
    $TYPEDEF
"""
@cenum OrtAllocatorType::Int32 begin
    Invalid = -1
    OrtDeviceAllocator = 0
    OrtArenaAllocator = 1
end

"""
    $TYPEDEF
"""
@cenum OrtMemType::Int32 begin
    OrtMemTypeCPUInput = -2
    OrtMemTypeCPUOutput = -1
    OrtMemTypeCPU = -1
    OrtMemTypeDefault = 0
end

"""
    $TYPEDSIGNATURES
"""
function CreateCpuMemoryInfo(
    api::OrtApi;
    allocator_type = OrtArenaAllocator,
    mem_type = OrtMemTypeDefault,
)::OrtMemoryInfo

    gchandles = Any[api]
    p_ptr = Ref(C_NULL)
    status = @ccall $(api.CreateCpuMemoryInfo)(
        allocator_type::OrtAllocatorType,
        mem_type::OrtMemType,
        p_ptr::Ptr{Ptr{Cvoid}},
    )::OrtStatusPtr
    into_julia(OrtMemoryInfo, api, p_ptr, status, gchandles)
end

"""
    $TYPEDEF
"""
@cenum ONNXTensorElementDataType::UInt32 begin
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED = 0
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT = 1
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 = 2
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 = 3
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 = 4
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 = 5
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 = 6
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 = 7
    ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING = 8
    ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL = 9
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 = 10
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE = 11
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 = 12
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 = 13
    ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64 = 14
    ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128 = 15
    ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 = 16
end

const JULIA_TYPE_FROM_ONNX = Dict{ONNXTensorElementDataType,Type}()
"""
    $TYPEDSIGNATURES
"""
function juliatype(onnx::ONNXTensorElementDataType)::Type
    return JULIA_TYPE_FROM_ONNX[onnx]
end
for (onnx, T) in [
    #(ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED  ,
    (ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, Cfloat),
    (ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, UInt8),
    (ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, Int8),
    (ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, UInt16),
    (ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, Int16),
    (ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, Int32),
    (ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, Int64),
    (ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, Cstring),
    # (ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL       ,
    # (ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16    ,
    (ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, Cdouble),
    (ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32, UInt32),
    (ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64, UInt64),
    # (ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64  ,
    # (ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128 ,
    # (ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16   ,
]
    @eval ONNXTensorElementDataType(::Type{$T}) = $onnx
    JULIA_TYPE_FROM_ONNX[onnx] = T
end

"""
    $TYPEDSIGNATURES
"""
function IsTensor(api::OrtApi, val::OrtValue)::Bool
    out = Ref(Cint(0))
    GC.@preserve val begin
        status = @ccall $(api.IsTensor)(val.ptr::Ptr{Cvoid}, out::Ptr{Cint})::OrtStatusPtr
        check_and_release(api, status)
    end
    return Bool(out[])
end

"""
    $TYPEDSIGNATURES
"""
function GetTensorElementType(
    api::OrtApi,
    o::OrtTensorTypeAndShapeInfo,
)::ONNXTensorElementDataType
    # https://github.com/microsoft/onnxruntime/blob/1886f1a737fb3aa891dea213e076a091002e083f/onnxruntime/core/framework/tensor_type_and_shape.cc#L54
    p_out = Ref{ONNXTensorElementDataType}()
    GC.@preserve o begin
        status = @ccall $(api.GetTensorElementType)(
            o.ptr::Ptr{Cvoid},
            p_out::Ptr{ONNXTensorElementDataType},
        )::OrtStatusPtr
        check_and_release(api, status)
        p_out[]
    end
end

"""
    $TYPEDSIGNATURES
"""
function GetDimensionsCount(api::OrtApi, o::OrtTensorTypeAndShapeInfo)::Csize_t
    p_out = Ref{Csize_t}()
    GC.@preserve o begin
        status = @ccall $(api.GetDimensionsCount)(
            o.ptr::Ptr{Cvoid},
            p_out::Ptr{Csize_t},
        )::OrtStatusPtr
        check_and_release(api, status)
        p_out[]
    end
end

"""
    $TYPEDSIGNATURES
"""
function GetDimensions(
    api::OrtApi,
    o::OrtTensorTypeAndShapeInfo,
    ndims = GetDimensionsCount(api, o),
)::Vector{Int64}
    out = Vector{Int64}(undef, ndims)
    GC.@preserve out o begin
        status = @ccall $(api.GetDimensions)(
            api::OrtApi,
            o.ptr::Ptr{Cvoid},
            pointer(out)::Ptr{Int64},
            ndims::Csize_t,
        )::OrtStatusPtr
        check_and_release(api, status)
        return out
    end
end

"""
    $TYPEDSIGNATURES
"""
function GetTensorTypeAndShape(api::OrtApi, o::OrtValue)::OrtTensorTypeAndShapeInfo
    p_ptr = Ref(C_NULL)
    gchandles = Any[api, o]
    status = @ccall $(api.GetTensorTypeAndShape)(
        o.ptr::Ptr{Cvoid},
        p_ptr::Ptr{Ptr{Cvoid}},
    )::OrtStatusPtr
    into_julia(OrtTensorTypeAndShapeInfo, api, p_ptr, status, gchandles)
end

"""
    $TYPEDSIGNATURES
"""
function CreateTensorWithDataAsOrtValue(
    api::OrtApi,
    memory_info::OrtMemoryInfo,
    data::Array,
)::OrtValue

    shapevec = collect(Int64, size(data))
    onnx_elty = ONNXTensorElementDataType(eltype(data))

    p_ptr = Ref(C_NULL)

    # https://github.com/microsoft/onnxruntime/blob/e2194797a713f19a15ce2afbe0c6a78d5d8c467e/onnxruntime/core/session/onnxruntime_c_api.cc#L207
    # ORT_API_STATUS_IMPL(
    #   OrtApis::CreateTensorWithDataAsOrtValue,
    #   _In_ const OrtMemoryInfo* info,
    #   _Inout_ void* p_data,
    #   size_t p_data_len,
    #   _In_ const int64_t* shape,
    #   size_t shape_len,
    #   ONNXTensorElementDataType type,
    #   _Outptr_ OrtValue** out
    # )
    gchandles = [data, memory_info, shapevec]
    info::Ptr{Cvoid} = memory_info.ptr
    p_data::Ptr{Cvoid} = pointer(data)
    p_data_len::Csize_t = length(data) * sizeof(eltype(data))
    shape::Ptr{Int64} = pointer(shapevec)
    shape_len::Csize_t = length(shapevec)
    status = @ccall $(api.CreateTensorWithDataAsOrtValue)(
        info::Ptr{Cvoid},
        p_data::Ptr{Cvoid},
        p_data_len::Csize_t,
        shape::Ptr{Int64},
        shape_len::Csize_t,
        onnx_elty::ONNXTensorElementDataType,
        p_ptr::Ptr{Ptr{Cvoid}},
    )::OrtStatusPtr
    return into_julia(OrtValue, api, p_ptr, status, gchandles)
end

"""
    $TYPEDSIGNATURES

This function is unsafe, because its output points to memory owned by `tensor`.
"""
function unsafe_GetTensorMutableData(api::OrtApi, tensor::OrtValue)::Array
    p_ptr = Ref(C_NULL)
    GC.@preserve tensor begin
        status = @ccall $(api.GetTensorMutableData)(
            tensor.ptr::Ptr{Cvoid},
            p_ptr::Ptr{Ptr{Cvoid}},
        )::OrtStatusPtr
    end
    check_and_release(api, status)
    info = GetTensorTypeAndShape(api, tensor)
    ONNX_type = GetTensorElementType(api, info)
    T = juliatype(ONNX_type)
    shape = Tuple(GetDimensions(api, info))
    ptrT = Ptr{T}(p_ptr[])
    @check tensor.isalive
    return unsafe_wrap(Array, ptrT, shape, own = false)
end

"""
    $TYPEDSIGNATURES
"""
function GetTensorMutableData!(out::AbstractArray, api::OrtApi, tensor::OrtValue)
    GC.@preserve tensor begin
        data_owned_by_tensor = unsafe_GetTensorMutableData(api, tensor)
        copy!(out, data_owned_by_tensor)
    end
    return out
end

"""
    $TYPEDSIGNATURES
"""
function GetTensorMutableData(api::OrtApi, tensor::OrtValue)::Array
    GC.@preserve tensor begin
        data_owned_by_tensor = unsafe_GetTensorMutableData(api, tensor)
        copy(data_owned_by_tensor)
    end
end

"""
    $TYPEDSIGNATURES
"""
function CreateRunOptions(api::OrtApi)::OrtRunOptions
    p_ptr = Ref(C_NULL)
    gchandles = Any[api]
    status = @ccall $(api.CreateRunOptions)(p_ptr::Ptr{Ptr{Cvoid}})::OrtStatusPtr
    into_julia(OrtRunOptions, api, p_ptr, status, gchandles)
end

"""
    $TYPEDSIGNATURES
"""
function Run(
    api::OrtApi,
    session::OrtSession,
    run_options::Union{Nothing,OrtRunOptions},
    input_names::Vector{String},
    inputs::Vector{OrtValue},
    output_names::Vector{String},
)::Vector{OrtValue}

    @argcheck length(input_names) == length(inputs)

    input_len = length(input_names)
    output_names_len = length(output_names)
    _input_names = Cstring[Base.unsafe_convert(Cstring, s) for s in input_names]
    _output_names = Cstring[Base.unsafe_convert(Cstring, s) for s in output_names]
    _inputs = Ptr{Cvoid}[(inp::OrtValue).ptr for inp in inputs]
    _outputs = Ptr{Cvoid}[C_NULL for _ = 1:output_names_len]

    gchandles = (;
        session,
        run_options,
        input_names,
        inputs,
        output_names,
        _input_names,
        _inputs,
        _outputs,
    )
    run_options_ptr = if run_options === nothing
        C_NULL
    else
        run_options.ptr
    end
    GC.@preserve gchandles begin
        status = @ccall $(api.Run)(
            session.ptr::Ptr{Cvoid},
            run_options_ptr::Ptr{Cvoid}, # TODO
            _input_names::Ptr{Cstring},
            _inputs::Ptr{Ptr{Cvoid}},
            input_len::Csize_t,
            _output_names::Ptr{Cstring},
            output_names_len::Csize_t,
            _outputs::Ptr{Ptr{Cvoid}},
        )::OrtStatusPtr
        check_and_release(api, status)
        outdims = (output_names_len,)
        outputs = map(_outputs) do ptr
            isalive = true
            gchandles = Any[]
            out = OrtValue(ptr, gchandles, isalive)
            finalizer(out) do val
                release(api, val)
            end
            out
        end
        return outputs
    end
end

_collect(::Type{T}, arr::Array{T}) where {T} = arr
_collect(T, itr) = collect(T, itr)
"""
    $TYPEDSIGNATURES
"""
function CreateArenaCfgV2(api::OrtApi, keys, vals)::OrtArenaCfg
    @argcheck length(keys) == length(vals)
    keys = Cstring[Cstring(key) for key in keys]
    vals = _collect(Csize_t, vals)
    num_keys = length(keys)
    p_ptr = Ref(C_NULL)
    GC.@preserve keys vals begin
        status = @ccall $(api.CreateArenaCfgV2)(
            pointer(keys)::Ptr{Cstring},
            pointer(vals)::Ptr{Csize_t},
            num_keys::Csize_t,
            p_ptr::Ptr{Ptr{Cvoid}},
        )::OrtStatusPtr
    end
    gchandles = Any[]
    into_julia(OrtArenaCfg, api, p_ptr, status, gchandles)
end

"""
    $TYPEDEF
"""
@cenum OrtCudnnConvAlgoSearch::UInt32 begin
    EXHAUSTIVE = 0
    HEURISTIC = 1
    DEFAULT = 2
end

"""
    $TYPEDEF
"""
struct OrtCUDAProviderOptions
    device_id::Cint
    cudnn_conv_algo_search::OrtCudnnConvAlgoSearch
    gpu_mem_limit::Csize_t
    arena_extend_strategy::Cint
    do_copy_in_default_stream::Cint
    has_user_compute_stream::Cint
    user_compute_stream::Ptr{Cvoid}
    default_memory_arena_cfg::Ptr{Cvoid} # Ptr{OrtArenaCfg}
end

"""
    $TYPEDSIGNATURES
"""
function OrtCUDAProviderOptions(;
        device_id                 = 0,
        cudnn_conv_algo_search    = EXHAUSTIVE,
        gpu_mem_limit             = typemax(Csize_t),
        arena_extend_strategy     = 0,
        do_copy_in_default_stream = false,
        has_user_compute_stream   = false,
        user_compute_stream       = C_NULL,
        default_memory_arena_cfg  = C_NULL,
    )::OrtCUDAProviderOptions
    OrtCUDAProviderOptions(
        device_id                 ,
        cudnn_conv_algo_search    ,
        gpu_mem_limit             ,
        arena_extend_strategy     ,
        do_copy_in_default_stream ,
        has_user_compute_stream   ,
        user_compute_stream       ,
        default_memory_arena_cfg  ,
   )
end

"""
    $TYPEDSIGNATURES
"""
function SessionOptionsAppendExecutionProvider_CUDA(
    api::OrtApi,
    session_options::OrtSessionOptions,
    cuda_options::OrtCUDAProviderOptions,
)::Nothing
    cuda_options_ptr = Ref(cuda_options)
    GC.@preserve cuda_options_ptr session_options begin
        status = @ccall $(api.SessionOptionsAppendExecutionProvider_CUDA)(
                session_options.ptr::Ptr{Cvoid},
                cuda_options_ptr::Ptr{Cvoid},
            )::OrtStatusPtr
        check_and_release(api, status)
    end
end

################################################################################
##### exports
################################################################################
export OrtApiBase, GetApi, GetVersionString
export OrtApi
export OrtCUDAProviderOptions
export OrtCudnnConvAlgoSearch
export ONNXTensorElementDataType
export release
for f in fieldnames(OrtApi)
    if isdefined(@__MODULE__, f)
        @eval export $f
    end
end

export OrtException

end#module
