#=
Created on 02/05/2023 17:14:09
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

The associated operations for the HDVs
=#


"""
bipol2grad(x::Number)

Maps a bipolar number in [-1, 1] to the [0, 1] interval.
"""
bipol2grad(x::Number) = (x + one(x)) / 2

three_pi(x, y) = abs(x-y)==1 ? zero(x) : x * y / (x * y + (one(x) - x) * (one(y) - y))
fuzzy_xor(x, y) = (one(x)-x) * y + x * (one(y)-y)

three_pi_bipol(x, y) = grad2bipol(three_pi(bipol2grad(x), bipol2grad(y)))
fuzzy_xor_bipol(x, y) = grad2bipol(fuzzy_xor(bipol2grad(x), bipol2grad(y)))

# BINDING
# -------

Base.bind(u::VT, v::VT) where {VT<:AbstractHDV} = bind(ElementType(VT), u, v)
Base.bind(::BinaryElements, u::VT, v::VT) where {VT<:AbstractHDV} = VT(u.v .⊻ v.v)
Base.bind(::NumericElements, u::VT, v::VT) where {VT<:AbstractHDV} = VT(u.v .* v.v)
Base.bind(::GradedElements, u::VT, v::VT) where {VT<:AbstractHDV} = VT(fuzzy_xor.(u.v, v.v))

Base.:∘(u::VT, v::VT) where {VT<:AbstractHDV} = bind(u, v)

# BUNDLING
# --------


# SHIFTING
# --------


# SIMILARITY
# ----------