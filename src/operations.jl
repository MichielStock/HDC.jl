#=
Created on 02/05/2023 17:14:09
Last update: 2023-05-24

@author: Michiel Stock
michielfmstock@gmail.com

The associated operations for the HDVs
=#


"""
bipol2grad(x::Number)

Maps a bipolar number in [-1, 1] to the [0, 1] interval.
"""
bipol2grad(x::Number) = (x + one(x)) / 2

"""
grad2bipol(x::Number)

Maps a bipolar number in [0, 1] to the [-1, 1] interval.
"""
grad2bipol(x::Number) = 2x - 1

three_pi(x, y) = abs(x-y)==1 ? zero(x) : x * y / (x * y + (one(x) - x) * (one(y) - y))
fuzzy_xor(x, y) = (one(x)-x) * y + x * (one(y)-y)

three_pi_bipol(x, y) = grad2bipol(three_pi(bipol2grad(x), bipol2grad(y)))
fuzzy_xor_bipol(x, y) = grad2bipol(fuzzy_xor(bipol2grad(x), bipol2grad(y)))

inv_fuzzy_xor(z, x) = (z - x) / (1 - 2x)
inv_fuzzy_xor_bipol(z, x) = grad2bipol(inv_fuzzy_xor(bipol2grad(z), bipol2grad(x)))

# BINDING
# -------

Base.bind(u::VT, v::VT) where {VT<:AbstractHDV} = bind(ElementType(VT), u, v)
Base.bind(::BinaryElements, u::VT, v::VT) where {VT<:AbstractHDV} = VT(u.v .⊻ v.v)
Base.bind(::NumericElements, u::VT, v::VT) where {VT<:AbstractHDV} = VT(u.v .* v.v)
Base.bind(::GradedElements, u::VT, v::VT) where {VT<:AbstractHDV} = VT(fuzzy_xor.(u.v, v.v))

Base.bind(vs::AbstractHDV...) = reduce(bind, vs)

Base.:∘(vs::AbstractHDV...) = bind(vs...)

# default binding is its own inverse
unbind(u::VT, v::VT) where {VT<:AbstractHDV} = bind(u, v)
⊘(u::VT, v::VT) where {VT<:AbstractHDV} = unbind(u, v)

# graded
unbind(u::GradedHDV, v::GradedHDV) = GradedHDV(inv_fuzzy_xor.(u.v, v.v))
unbind(u::GradedBipolarHDV, v::GradedBipolarHDV) = GradedHDV(inv_fuzzy_xor_bipol.(u.v, v.v))

# note: might make sense to have a bindingfunction that the elements fall back on

# BUNDLING
# --------

bundle(vs::AbstractHDV...; kwargs...) =  bundle(eltype(vs), vs; kwargs...)
bundle(vs; kwargs...) =  bundle(eltype(vs), vs; kwargs...)
bundle(VT::Type{AbstractHDV}, vs) = error("bundle not implemented for this type")

# binary versions

function bundle(VT::Type{Union{DenseHDV{Bool},BitHDV}}, collection; evenresolve="random")
    r = mapreduce(get_vector, .+, collection)
    if iseven(length(collection))
        if evenresolve == "random"
            r .+= bitrand(N)
        elseif evenresolve == "positive"
            r .+= 1
        elseif evenresolve == "negative"
            nothing
        else
            throw(ArgumentError("evenresolve is either `random`, `positive` or `negative` (got $evenresolve)"))
        end
    end
    return VT(r .> N ÷ 2)
end

# bipolar dense
bundle(VT::Type{IntHDV}, vs) = VT(mapreduce(get_vector, .+, vs))

# real
bundle(VT::Type{IntHDV}, vs) = mapreduce(get_vector, .+, vs) |> normalize! |> VT

# graded
bundle(VT::Type{GradedHDV}, vs) = mapreduce(get_vector, (x,y)->three_pi.(x,y), vs) |> VT
bundle(VT::Type{GradedBipolarHDV}, vs) = mapreduce(get_vector, (x,y)->three_pi_bipol.(x,y), vs) |> VT



# PERMUTATION
# -----------

shift(v::VT, k=1) where {VT<:AbstractHDV} = VT(circshift(v.v, k))
ρ(v::VT, k=1) where {VT<:AbstractHDV} = shift(v, k)

shift!(v::VT, k=1) where {VT<:AbstractHDV} = circshift!(v.v, k)
ρ!(v::VT, k=1) where {VT<:AbstractHDV} = shift!(v, k)

#=
# TODO: fix this, does not take into account lengths % 64 and shifts
# the other way, for some reason
# seems like it shifts the other way
function bitvector_circshift(v::BitVector, k::Int)
    @assert 0 ≤ k ≤ 64 "Only implemented for shifts ≤ 64! (attempted for k=$k)"
    u = similar(v)
    rem = v.chunks[end] >> (64 - k)
    @inbounds for i in 1:length(v.chunks)
        vi = v.chunks[i]
        u.chunks[i] = (rem | (vi << k))
        rem = vi >> (64 - k)
    end
    return u
end
=#

# SIMILARITY
# ----------

sim(u::VT, v::VT) where {VT<:AbstractHDV} = sim(VT, u, v)
τ(u, v) = sim(u, v)

sim(u::AbstractHDV) = v -> sim(u, v)
τ(u) = v -> sim(u, v)

# helpers
Base.count_ones(x::BitArray) = mapreduce(count_ones, +, x.chunks)
sim_cos(x::AbstractVector, y::AbstractVector) = dot(x, y) / (norm(x) * norm(y))

@inline function sim_cos_bp(x::BitArray, y::BitArray)
    N = length(x)
    # matches
    m = mapreduce(p->count_zeros(p[1] ⊻ p[2]), +, zip(x.chunks,y.chunks))
    m -= 64 - N % 64
    return (2m - N) / N
end

sim_tanimoto(x, y) = dot(x, y) / (dot(x, x) + dot(y, y) - dot(x, y))
sim_hamming(x, y) = 1 - mapreduce(p->p[1]==p[2], +, zip(x,y)) / length(x)

@inline function sim_hamming(x::BitVector, y::BitVector)
    N = length(x)
    # matches
    m = mapreduce(p->count_zeros(p[1] ⊻ p[2]), +, zip(x.chunks,y.chunks))
    m -= 64 - N % 64
    return m / N
end

# methods for types
sim(::typeof{BinaryHDV}, u, v) = sim_tanimoto(u.v, v.v)
sim(::typeof{BipolarHDV}, u, v) = sim_cos_bp(u.v, v.v)
sim(::typeof{AbstractHDV}) = sim_cos(u.v, v.v)
sim(::typeof{DenseHDV{Bool}}) = sim_tanimoto(u.v, v.v)