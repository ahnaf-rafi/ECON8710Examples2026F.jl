# If user did not supply weights, return 1.
"""
    obswgt(w, i)

Extract weight for unit `i` from weight vector `w`. If `w = nothing`, then
assign w[i] = 1.0.
"""
@inline obswgt(::Nothing, ::Int) = 1.0
@inline obswgt(w::AbstractVector{<:Real}, i::Int) = @inbounds w[i]
export obswgt

abstract type WeightScheme end
