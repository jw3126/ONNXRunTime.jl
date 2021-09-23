using ArgCheck
using DocStringExtensions
using TimerOutputs: @timeit

"""
    $TYPEDEF

Array with dense C style memory layout. The same layout as used by `onnxruntime`.
"""
struct CArray{T,N} <: AbstractArray{T,N}
    _data::Vector{T}
    size::NTuple{N,Int}
    """
        CArray(v::Vector{T}, size::NTuple{N,Int}) where {T,N}

    Construct a `CArray`, whose memory is backed by `v` and with shape given by `size`.
    In a typical use, `v` wraps a pointer returned by a call to a C library.
    """
    function CArray(v::Vector{T}, size::NTuple{N,Int}) where {T,N}
        @boundscheck begin
            @argcheck length(v) == prod(size)
        end
        return new{T,N}(v, size)
    end
end
function CArray(v::AbstractVector, size::NTuple{N,Int}) where {N}
    CArray(collect(v), size)
end

"""

See also `Base.unsafe_wrap`.
"""
function unsafe_cwrap(
    ptr::Ptr{T},
    dims::NTuple{N,Integer};
    own::Bool = false
)::CArray{T, N} where {N,T}
    len = prod(dims)
    v::Vector{T} = Base.unsafe_wrap(Array, ptr, (len,); own=own)
    CArray(v, dims)
end
Base.pointer(o::CArray) = Base.pointer(parent(o))

Base.size(o::CArray) = o.size
Base.IndexStyle(::Type{<:CArray}) = Base.IndexCartesian()
Base.@propagate_inbounds function Base.getindex(o::CArray{<:Any, N}, I::CartesianIndex{N}) where{N}
    inds = Tuple(I)
    i = linearindex_C(size(o), inds)
    o._data[i]
end
Base.@propagate_inbounds function linearindex_C(dims::NTuple{N,Int}, inds::NTuple{N,Int})::Int where {N}
    rinds = reverse(inds)
    LinearIndices(reverse(dims))[rinds...]
end

Base.@propagate_inbounds function Base.getindex(o::CArray{<:Any}, inds::Integer...) where {N}
    I = CartesianIndex(inds)
    o[I]
end
Base.@propagate_inbounds function Base.setindex!(o::CArray{<:Any}, val, inds::Integer...) where {N}
    i = linearindex_C(size(o), inds)
    o._data[i] = val
end
Base.parent(o::CArray) = o._data
CArray(o::CArray) = o

layout(o::AbstractArray) = layout(typeof(o))
layout(::Type{<:Array}) = :fortran
layout(::Type{<:CArray}) = :C
function layout(T::Type{<:AbstractArray})
    error(
        """
        Memory layout of array type $T unknown.
        Consider implementing layout(::Type{$T})
        """
    )

end

"""
    CArray(arr::AbstractArray)

Construct a `CArray` from `arr`. Under the hood, this copies each element of `arr`
to a C layout memory region.
"""
@timeit TIMER function CArray(arr::AbstractArray)
    @assert layout(arr) == :fortran
    Base.require_one_based_indexing(arr)
    dims = size(arr)
    T = eltype(arr)
    v = Vector{T}(undef, length(arr))
    @timeit TIMER "loop"  begin
        for I in CartesianIndices(arr)
            i = linearindex_C(dims, Tuple(I))
            v[i] = arr[I]
        end
    end
    CArray(v, dims)
end

@timeit TIMER function collect_CArray(arr::CArray)
    out = similar(arr)
    for I in CartesianIndices(arr)
        out[I] = arr[I]
    end
    out
end

function reversedims(arr::AbstractArray{T,N}) where {N,T}
    rev = ntuple(i->N+1-i, Val(N))
    permutedims(arr, rev)
end

Base.collect(arr::CArray{<:Any,0}) = fill(only(arr._data))
function Base.collect(arr::CArray)
    reversedims(reshape(arr._data, reverse(size(arr))))
end
