module TestHighlevel

using Test
using ONNXRunTime
const OX = ONNXRunTime
using ONNXRunTime: juliatype

@testset "high level" begin
    @testset "increment2x3.onnx" begin
        path = OX.testdatapath("increment2x3.onnx")
        model = OX.load_inference(path, execution_provider=:cpu)
        @test OX.input_names(model) == ["input"]
        @test OX.output_names(model) == ["output"]
        input = randn(Float32, 2,3)
        #= this works             =# model(Dict("input" => randn(Float32, 2,3)), ["output"])
        @test_throws OX.ArgumentError model(Dict("nonsense" => input), ["output"])
        @test_throws OX.ArgumentError model(Dict("input" => input), ["nonsense"])
        @test_throws OX.OrtException  model(Dict("input" => input), String[])
        @test_throws OX.ArgumentError model(Dict("input" => input, "unused"=>input), ["output"])
        @test_throws OX.ArgumentError model(Dict("input" => input, "unused"=>input), ["output"])
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
        @test d isa AbstractDict
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
    @testset "swap_x_.onnx" begin
        path = OX.testdatapath("swap_x_.onnx")
        model = OX.load_inference(path)
        @test OX.input_names(model)  == ["in1", "in2"]
        @test OX.output_names(model) == ["out1", "out2"]
        in1 = randn(Float32, 2,3)
        in2 = randn(Float32, 4,5)
        res = model((;in1, in2))
        @test keys(res) === (:out1, :out2)
        @test res isa NamedTuple
        @test res.out1 == in2
        @test res.out2 == in1
    end
    @testset "getindex_12.onnx" begin
        path = OX.testdatapath("getindex_12.onnx")
        model = OX.load_inference(path)
        inputs = (input=collect(reshape(1f0:20, 4,5)),)
        out = model(inputs).output
        @test inputs.input[2,3] == only(out)
    end
    @testset "copy2d.onnx" begin
        path = OX.testdatapath("copy2d.onnx")
        model = OX.load_inference(path)
        inputs = (input=randn(Float32,3,4),)
        out = model(inputs).output
        @test inputs.input == out
    end
end


end#module
