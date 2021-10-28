module TestCAPI
using Test
using ONNXRunTime.CAPI
import ONNXRunTime as OX

@testset "Session" begin
    api = GetApi()
    env = CreateEnv(api, name="myenv")
    @testset "increment2x3" begin
        path = OX.testdatapath("increment2x3.onnx")
        session_options = CreateSessionOptions(api)
        @test (sprint(show, session_options); true)
        @test_throws Exception CreateSession(api, env, "does_not_exits.onnx", session_options)
        session = CreateSession(api, env, path, session_options)
        @test (sprint(show, session); true)
        @test SessionGetInputCount(api, session) == 1
        @test SessionGetOutputCount(api, session) == 1
        mem = CreateCpuMemoryInfo(api)
        @test (sprint(show, mem); true)
        allocator = CreateAllocator(api, session, mem)
        @test (sprint(show, allocator); true)
        @test SessionGetInputName(api, session, 0, allocator) == "input"
        @test_throws OX.OrtException SessionGetInputName(api, session, 1, allocator)
        @test SessionGetOutputName(api, session, 0, allocator) == "output"
        @test_throws OX.OrtException SessionGetOutputName(api, session, 1, allocator)
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
        @test (sprint(show, output_tensor); true)
        @test output_tensor isa OrtValue
        output_array = GetTensorMutableData(api, output_tensor)
        @test typeof(output_array) <: AbstractMatrix{Float32}
        @test size(output_array) == (2,3)
        @test output_array ≈ 1 .+ input_array
    end
end

@testset "tensor roundtrip" begin
    api = GetApi()
    mem = CreateCpuMemoryInfo(api)
    data = randn(2,3)
    tensor = CreateTensorWithDataAsOrtValue(api, mem, vec(OX.reversedims(data)), size(data))
    @test IsTensor(api, tensor)
    info = GetTensorTypeAndShape(api, tensor)
    onnxelty = GetTensorElementType(api, info)
    @test onnxelty isa ONNXTensorElementDataType
    @test OX.juliatype(onnxelty) == eltype(data)
    @test GetDimensionsCount(api, info) == 2
    @test GetDimensions(api, info) == [2,3]
    data2 = GetTensorMutableData(api, tensor)
    @test length(data2) == length(data)
    @test size(data2) == size(data)
    @test eltype(data2) == eltype(data)
    @test data2 == data
end

end
