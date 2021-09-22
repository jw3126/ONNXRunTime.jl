module TestCArray
using Test
using ONNXRunTime: CArray,unsafe_cwrap

function ==ₜ(x,y)
    (typeof(x) === typeof(y)) && (x == y)
end

@testset "CArray" begin
    o = CArray(1:4, (2,2))
    @test parent(o) ==ₜ [1,2,3,4]
    @test o == [1 2; 3 4]
    @test o[1,1] == 1
    @test o[1,2] == 2
    @test o[2,1] == 3
    @test o[2,2] == 4
    GC.@preserve o begin
        o2 = unsafe_cwrap(pointer(o), (2,2), own=false)
        @test o2 == o
        @test pointer(o2) === pointer(o)
        mem::Vector{Int} = Base.unsafe_wrap(Array, pointer(o), 4, own=false)
        @test mem ==ₜ [1,2,3,4]
    end

    Base.pointer(o::CArray) = pointer(o._data)
    function unsafe_load_C(o::CArray, i::Int)
        @assert 0 <= i <= length(o)
        ptr = pointer(o) + i * sizeof(eltype(o))
        unsafe_load(ptr)
    end
    o = CArray([1 2; 3 4])
    @test unsafe_load_C(o, 0) == 1
    @test unsafe_load_C(o, 1) == 2
    @test unsafe_load_C(o, 2) == 3
    @test unsafe_load_C(o, 3) == 4

    for arr in [
            randn(1),
            randn(2),
            randn(1,2,3),
            randn(1,2,3,4),
            randn(1,2,3,4,5),
        ]
        carr = @inferred CArray(arr)
        @test carr isa CArray
        @test size(arr) == size(carr)
        @test carr == arr
        arr_roundtrip = Array(carr)
        @test arr_roundtrip ==ₜ arr
    end
end

end#module
