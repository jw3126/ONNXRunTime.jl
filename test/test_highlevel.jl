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

        s = sprint(show, model)
        @test occursin("in1", s)
        @test occursin("in2", s)
        @test occursin("out1", s)
        @test occursin("out2", s)
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
    @testset "matmul.onnx" begin
        path = OX.testdatapath("matmul.onnx")
        model = OX.load_inference(path)
        inputs = (
                  input1 = randn(Float32, 2,3),
                  input2 = randn(Float32, 3,4),
        )
        out = model(inputs).output
        @test out ≈ inputs.input1 * inputs.input2
    end
    @testset "xyz_3x4x5.onnx" begin
        path = OX.testdatapath("xyz_3x4x5.onnx")
        model = OX.load_inference(path)
        inputs = (input=randn(Float32,4,10),)
        out = model(inputs)
        @test out.identity == inputs.input
        @test size(out.X) == size(out.Y) == size(out.Z) == (3,4,5)
        for _ in 1:20
            ix = rand(axes(out.X,1))
            iy = rand(axes(out.X,1))
            iz = rand(axes(out.X,1))
            @test out.X[ix,iy,iz] == ix-1
            @test out.Y[ix,iy,iz] == iy-1
            @test out.Z[ix,iy,iz] == iz-1
        end
    end
    @testset "Conv1d1.onnx" begin
        path = OX.testdatapath("Conv1d1.onnx")
        model = OX.load_inference(path)
        inputs = (input=randn(Float32,4,2,10),)
        out = model(inputs)
        expected = fill(0f0, 4,3,8)
        expected[:,2,:] .= 1
        @test out.output == expected
    end
    @testset "Conv1d2.onnx" begin
        path = OX.testdatapath("Conv1d2.onnx")
        model = OX.load_inference(path)
        input = Array{Float32,3}(undef, (1,2,3))
        input[1,1,1] = 1
        input[1,1,2] = 2
        input[1,1,3] = 3
        input[1,2,1] = 4
        input[1,2,2] = 5
        input[1,2,3] = 6
        inputs=(;input)
        out = model(inputs).output
        @test out[1,1,1] == 1
        @test out[1,1,2] == 3
        @test out[1,1,3] == 5
        @test out[1,2,1] == 0
        @test out[1,2,2] == 0
        @test out[1,2,3] == 0
    end
end


end#module
