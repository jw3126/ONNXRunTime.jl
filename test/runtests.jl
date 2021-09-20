using Test

using ONNXRunTime
const OX = ONNXRunTime
using ONNXRunTime: juliatype

@testset "high level" begin
    @testset "increment2x3.onnx" begin
        path = OX.testdatapath("increment2x3.onnx")
        model = OX.load_inference(path)
        @test OX.input_names(model) == ["input"]
        @test OX.output_names(model) == ["output"]
        input = randn(Float32, 2,3)
        #= this works             =# model(Dict("input" => randn(Float32, 2,3)), ["output"])
        @test_throws OX.OrtException model(Dict("nonsense" => input), ["output"])
        @test_throws OX.OrtException model(Dict("input" => input), ["nonsense"])
        @test_throws OX.OrtException model(Dict("input" => input), String[])
        @test_throws OX.OrtException model(Dict("input" => input, "unused"=>input), ["output"])
        @test_throws OX.OrtException model(Dict("input" => input, "unused"=>input), ["output"])
        @test_throws OX.OrtException model(Dict("input" => randn(Float32, 3,2)), ["output"])
        @test_throws Exception       model(Dict("input" => randn(Int, 2,3)    ), ["output"])
        @test_throws OX.OrtException model(Dict("input" => randn(Float64, 2,3)), ["output"])
        y = model(Dict("input" => input), ["output"])
        @test y == Dict("output" => input .+ 1f0)
        y = model(Dict("input" => input))
        @test y == Dict("output" => input .+ 1f0)
    end
    @testset "adder1x2x3.onnx" begin
        path = OX.testdatapath("adder1x2x3.onnx")
        model = OX.load_inference(path)
        @test OX.input_names(model) == ["x", "y"]
        @test OX.output_names(model) == ["sum"]
        x = randn(Float32, 1,2,3)
        y = randn(Float32, 1,2,3)
        d = model(Dict("x" => x, "y"=>y))
        @test d == Dict("sum" => x+y)
    end
    @testset "diagonal1x2x3x4.onnx" begin
        path = OX.testdatapath("diagonal1x2x3x4.onnx")
        model = OX.load_inference(path)
        @test OX.input_names(model) == ["in"]
        @test OX.output_names(model) == ["out1", "out2"]
        x = randn(Float64, 1,2,3,4)
        d = model(Dict("in" => x))
        @test d == Dict("out1" => x, "out2" => x)
    end
end

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
    @testset "increment2x3" begin
        path = ONNXRunTime.testdatapath("increment2x3.onnx")
        session = CreateSession(api, env, path)
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
end
