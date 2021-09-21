module TestCUDA
import CUDA
using Test
using ONNXRunTime
const OX = ONNXRunTime
using ONNXRunTime: SessionOptionsAppendExecutionProvider_CUDA

#using Libdl
#Libdl.dlopen("/home/jan/.julia/artifacts/e2fd6cdf04b830a1d802fb35a6193788d0a3811a/lib/libcudart.so.11.0")

@testset "CUDA high level" begin
    @testset "increment2x3.onnx" begin
        path = OX.testdatapath("increment2x3.onnx")
        model = OX.load_inference(path, execution_provider=:cuda)
        @test OX.input_names(model) == ["input"]
        @test OX.output_names(model) == ["output"]
        input = randn(Float32, 2,3)
        y = model((;input=input,), ["output"])
        @test y == (output=input .+ 1f0,)
    end
end

using ONNXRunTime.CAPI
@testset "CUDA low level" begin
    api = GetApi(execution_provider=:cuda)
    env = CreateEnv(api, name="myenv")
    path = ONNXRunTime.testdatapath("increment2x3.onnx")
    session_options = CreateSessionOptions(api)
    cuda_options = OrtCUDAProviderOptions()
    SessionOptionsAppendExecutionProvider_CUDA(api, session_options, cuda_options)
    session = CreateSession(api, env, path, session_options)
    @test SessionGetInputCount(api, session) == 1
    @test SessionGetOutputCount(api, session) == 1
    mem = CreateCpuMemoryInfo(api)
    allocator = CreateAllocator(api, session, mem)
    @test SessionGetInputName(api, session, 0, allocator) == "input"
    @test_throws OX.OrtException SessionGetInputName(api, session, 1, allocator)
    @test SessionGetOutputName(api, session, 0, allocator) == "output"
    @test_throws OX.OrtException SessionGetOutputName(api, session, 1, allocator)
    input_array = randn(Float32, 2,3)
    input_tensor = CreateTensorWithDataAsOrtValue(api, mem, input_array)
    run_options = CreateRunOptions(api)
    input_names = ["input"]
    output_names = ["output"]
    inputs = [input_tensor]
    outputs = Run(api, session, run_options, input_names, inputs, output_names)
    @test length(outputs) == 1
    output_tensor = first(outputs)
    @test output_tensor isa OrtValue
    output_array = GetTensorMutableData(api, output_tensor)
    @test typeof(output_array) == Matrix{Float32}
    @test size(output_array) == (2,3)
    @test output_array ≈ 1 .+ input_array
end
end#module

