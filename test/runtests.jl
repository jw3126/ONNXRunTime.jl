using Test

using ONNXRunTime

using ONNXRunTime: juliatype


@testset "tensor roundtrip" begin
    api = GetApi()
    mem = CreateCpuMemoryInfo(api)
    data = randn(2,3)
    tensor = CreateTensorWithDataAsOrtValue(api, mem, data)
    @test IsTensor(api, tensor)
    info = GetTensorTypeAndShape(api, tensor)
    onnxelty = GetTensorElementType(api, info)
    @test onnxelty isa ONNXTensorElementDataType
    @test juliatype(onnxelty) == eltype(data)
    @test GetDimensionsCount(api, info) == 2
    @test GetDimensions(api, info) == [2,3]
    data2 = GetTensorMutableData(api, tensor)
    @test length(data2) == length(data)
    @test size(data2) == size(data)
    @test eltype(data2) == eltype(data)
    @test data2 == data
end

@testset "Session" begin
    api = GetApi()
    env = CreateEnv(api, name="myenv")
    so = CreateSessionOptions(api)
    @test_throws Exception CreateSession(api, env, "does_not_exits.onnx")
    path = ONNXRunTime.testdatapath("increment2x3.onnx")
    session = CreateSession(api, env, path)
    @test SessionGetInputCount(api, session) == 1
    @test SessionGetOutputCount(api, session) == 1
    mem = CreateCpuMemoryInfo(api)
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
