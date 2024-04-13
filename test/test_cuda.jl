module TestCUDA
import CUDA
import cuDNN
using Test
using ONNXRunTime
const ORT = ONNXRunTime
using ONNXRunTime: SessionOptionsAppendExecutionProvider_CUDA

#using Libdl
#Libdl.dlopen("/home/jan/.julia/artifacts/e2fd6cdf04b830a1d802fb35a6193788d0a3811a/lib/libcudart.so.11.0")

@testset "CUDA high level" begin
    @testset "increment2x3.onnx" begin
        path = ORT.testdatapath("increment2x3.onnx")
        model = ORT.load_inference(path, execution_provider=:cuda)
        @test ORT.input_names(model) == ["input"]
        @test ORT.output_names(model) == ["output"]
        input = randn(Float32, 2,3)
        y = model((;input=input,), ["output"])
        @test y == (output=input .+ 1f0,)
    end

    @testset "provider options" begin
        path = ORT.testdatapath("Conv1d2.onnx")
        input = Array{Float32,3}(undef, (1,2,3))
        input[1,1,1] = 1
        input[1,1,2] = 2
        input[1,1,3] = 3
        input[1,2,1] = 4
        input[1,2,2] = 5
        input[1,2,3] = 6
        inputs=(;input)
        for conv_search in (:DEFAULT, :HEURISTIC, :EXHAUSTIVE)
            cuda_options = (;cudnn_conv_algo_search=conv_search)
            model = ORT.load_inference(path, execution_provider=:cuda,
                                       provider_options=cuda_options)
            out = model(inputs).output
            @test out[1,1,1] == 1
            @test out[1,1,2] == 3
            @test out[1,1,3] == 5
            @test out[1,2,1] == 0
            @test out[1,2,2] == 0
            @test out[1,2,3] == 0
        end
        cuda_options = (;cudnn_conv_algo_search=:NEVER_HEARD_OF_THIS)
        @test_throws ErrorException ORT.load_inference(path, execution_provider=:cuda,
                                                       provider_options=cuda_options)
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
    @test_throws ORT.OrtException SessionGetInputName(api, session, 1, allocator)
    @test SessionGetOutputName(api, session, 0, allocator) == "output"
    @test_throws ORT.OrtException SessionGetOutputName(api, session, 1, allocator)
    input_vec = randn(Float32, 6)
    input_array = [
        input_vec[1] input_vec[2] input_vec[3];
        input_vec[4] input_vec[5] input_vec[6];
    ]
    input_tensor = CreateTensorWithDataAsOrtValue(api, mem, input_vec, (2,3))
    run_options = CreateRunOptions(api)
    input_names = ["input"]
    output_names = ["output"]
    inputs = [input_tensor]
    outputs = Run(api, session, run_options, input_names, inputs, output_names)
    @test length(outputs) == 1
    output_tensor = first(outputs)
    @test output_tensor isa OrtValue
    output_array = GetTensorMutableData(api, output_tensor)
    @test typeof(output_array) <: AbstractMatrix{Float32}
    @test size(output_array) == (2,3)
    @test output_array ≈ 1 .+ input_array
end
end#module

