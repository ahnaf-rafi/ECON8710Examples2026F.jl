# If user did not supply weights, return 1.
@inline obswgt(::Nothing, ::Int) = 1.0
@inline obswgt(w::AbstractVector{<:Real}, i::Int) = @inbounds w[i]

export obswgt
