#=
Created on 24/04/2023 15:44:03
Last update: -

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

struct GradedHDV{T<:Real} <: DenseHDV{T}
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

struct GradedBipolarHDV{T<:Real} <: DenseHDV{T}
    v::Vector{T}
end

function GradedBipolarHDV((l, u)=(-1, 1); N::Int=10_000)
    @assert -1 ≤ l < 0 < u ≤ 1 "Upper and lower bounds for `GradedBipolarHDV` have to be `-1 ≤ l < 0 < u ≤ 1`"
    v = rand(N)
    if (l, u) != (-1, 1)
        v .*= u - l
        v .+= l
    end
    return GradedHDV(v)
end

# CONSTRUCTORS
# ------------
export hdv, binhdv, bphdv, sphdv, realhdv, gradhdv, gradbphdv

hdv(N::Int=10_000) = BinaryHDV(;N)
binhdv(N::Int=10_000) = BinaryHDV(;N)
bphdv(N::Int=10_000) = BipolarHDV(;N)
sphdv(N::Int=10_000, T::Type=Bool; p=0.1) = SparseHDV(sprand(T, N, p), p)
realhdv(N::Int=10_000, T::Type=Float64) = RealHDV(T; N)
gradhdv(N::Int=10_000; l=0, u=1) = GradedHDV((l, u); N)
gradbphdv(N::Int=10_000; l=-1, u=1) = GradedBipolarHDV((l, u); N)


