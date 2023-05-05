#=
Created on 24/04/2023 15:44:03
Last update: 2023-05-05

@author: Michiel Stock
michielfmstock@gmail.com

Types for hyperdimensional vectors, all subtypes of AbstractVectors.

We provide the following type hierarchy
- `BitHDV`s, which use `BitVector`s as container
    - `BinaryHDV` are binary
    - `BipolarHDV` are bipolar (difference is cosmetic) 
- `SparseHDV`s, which use `SparseVector` as container
    - contains `SparseBinaryHDV` and `SparseBipolarHDV` as subtypes
- `DenseHDV`s, which use regular dense `Vector`s as containers, they are also used as weights
    - Has `HDV{Type}` as concrete instantions
    - `GradedHDV` and `GradedBipolarHDV` as fuzzy versions
    = `RealHDV`
=#

export AbstractHDV, BitHDV, BinaryHDV, BipolarHDV, 
            SparseHDV, DenseHDV, RealHDV, IntHDV,
            GradedHDV, GradedBipolarHDV

abstract type AbstractHDV{T} <: AbstractVector{T} end

Base.size(hdv::AbstractHDV) = (length(hdv.v),)
Base.IndexStyle(::Type{<:AbstractHDV}) = IndexLinear()
Base.getindex(hdv::AbstractHDV, i::Int) = getindex(hdv.v, i)

LinearAlgebra.norm(hdv::AbstractHDV) = norm(hdv.v)
Base.sum(hdv::AbstractHDV) = sum(hdv.v)

# Bitvector HDV
# -------------

abstract type BitHDV{T} <: AbstractHDV{T} end

struct BinaryHDV <: BitHDV{Bool}
    v::BitVector
end

BinaryHDV(;N::Int=10_000) = BinaryHDV(bitrand(N))

struct BipolarHDV <: BitHDV{Int}
    v::BitVector
end

BipolarHDV(;N::Int=10_000) = BipolarHDV(bitrand(N))

Base.getindex(hdv::BipolarHDV, i::Int) = getindex(hdv.v, i) ? 1 : -1
Base.sum(hdv::BipolarHDV) = 2sum(hdv.v) - length(hdv)

# SparseHDV
# ---------

struct SparseHDV{T} <: AbstractHDV{T}
    v::SparseVector{T}
    p::Float64
end

SparseHDV(p; N::Int=10_000) = SparseHDV(sprand(T, N, p), p)

# DenseHDV
# --------

abstract type DenseHDV{T} <: AbstractHDV{T} end

struct RealHDV{T<:Real} <: DenseHDV{T}
    v::Vector{T}
    n::Int
end

RealHDV(v::AbstractVector) = RealHDV(v, 1)
RealHDV(T::Type=Float64; N::Int=10_000) = RealHDV(randn(T, N))

struct IntHDV{T<:Integer} <: DenseHDV{T}
    v::Vector{T}
    n::Int
end

IntHDV(v::AbstractVector) = IntHDV(v, 1)

abstract type AbstractGradedHDV{T} <: DenseHDV{T} end

struct GradedHDV{T<:Real} <: AbstractGradedHDV{T}
    v::Vector{T}
end

function GradedHDV((l, u)=(0, 1); N::Int=10_000)
    @assert 0 ≤ l < u ≤ 1 "Upper and lower bounds for `GradedHDV` have to be `0 ≤ l < u ≤ 1`"
    v = rand(N)
    if (l, u) != (0, 1)
        v .*= u - l
        v .+= l
    end
    return GradedHDV(v)
end

struct GradedBipolarHDV{T<:Real} <: AbstractGradedHDV{T}
    v::Vector{T}
end

function GradedBipolarHDV((l, u)=(-1, 1); N::Int=10_000)
    @assert -1 ≤ l < 0 < u ≤ 1 "Upper and lower bounds for `GradedBipolarHDV` have to be `-1 ≤ l < 0 < u ≤ 1`"
    v = rand(N)
    if (l, u) != (-1, 1)
        v .*= (u - l) / 2
        v .+= (l + 1) / 2
    end
    return GradedBipolarHDV(v)
end

Base.getindex(hdv::GradedBipolarHDV, i::Int) = 2getindex(hdv.v, i) - 1

# CONSTRUCTORS
# ------------
export hdv, binhdv, bphdv, sphdv, realhdv, gradhdv, gradbphdv

"""
    hdv(N::Int=10_000)

Generate a hyperdimensional vector of the `BinaryHDV` type with
a default dimensionality of `N=10_000`. This vector contains random
binary elements encoded effiently as a `BitVector`s
"""
hdv(N::Int=10_000) = BinaryHDV(;N)

"""
    binhdv(N::Int=10_000)

Generate a binary hyperdimensional vector of the `BinaryHDV` type with
a default dimensionality of `N=10_000`. This vector contains random
binary elements encoded effiently as a `BitVector`s.
"""
binhdv(N::Int=10_000) = BinaryHDV(;N)

"""
    binhdv(N::Int=10_000)

Generate a binary hyperdimensional vector of the `BinaryHDV` type with
a default dimensionality of `N=10_000`. This vector contains random
bipolar (-1, 1) elements encoded effiently as a `BitVector`s.
"""
bphdv(N::Int=10_000) = BipolarHDV(;N)

"""
    sphdv(N::Int=10_000, T::Type=Bool; p=0.1)

Generate a sparse hyperdimensional vector of the `SparseHDV` type with
a default dimensionality of `N=10_000` and a sparsity level of `p=0.1`. 
The elements of this vector are randomly set to `true` or `false` with a 
probability of `p`, and are contained in a `SparseVector` with specified
element type `T`.
"""
sphdv(N::Int=10_000, T::Type=Bool; p=0.1) = SparseHDV(sprand(T, N, p), p)

"""
    realhdv(N::Int=10_000, T::Type=Float64)

Generate a hyperdimensional vector of the `RealHDV` type with a 
default dimensionality of `N=10_000`. This vector contains  
standard normal values of type  
`T` values (default `Float64`).
"""
realhdv(N::Int=10_000, T::Type=Float64) = RealHDV(T; N)

"""
    gradhdv(N::Int=10_000; l=0, u=1)

Generate a graded hyperdimensional vector of the `GradedHDV` type with 
a default dimensionality of `N=10_000` and a range between `l` and `u`. 
This vector contains random graded elements, i.e., elements with 
values between `l` and `u` that are uniformly distributed.
"""
gradhdv(N::Int=10_000; l=0, u=1) = GradedHDV((l, u); N)

"""
    gradbphdv(N::Int=10_000; l=-1, u=1)

Generate a graded bipolar hyperdimensional vector of the `GradedBipolarHDV` 
type with a default dimensionality of `N=10_000` and a range between `l` 
and `u`. This vector contains random graded bipolar elements, i.e., 
elements with values between `l` and `u` that are uniformly distributed, 
and is encoded efficiently as an array of `Float64` values.
"""
gradbphdv(N::Int=10_000; l=-1, u=1) = GradedBipolarHDV((l, u); N)


# TRAITS
# ------

# these traits encode the element types that determine the behaviour of the operations
abstract type ElementType end
struct BinaryElements <: ElementType end
struct NumericElements <: ElementType end
struct GradedElements <: ElementType end

# numeric is the default
ElementType(::Type{<:AbstractHDV}) = NumericElements()
ElementType(::Type{<:BinaryHDV}) = BinaryElements()
ElementType(::Type{<:DenseHDV{Bool}}) = BinaryElements()
ElementType(::Type{<:AbstractGradedHDV}) = GradedElements()


# PROMOTION
# ---------

# TODO: implement promotion