#---------------------------------------------------------------------#
# Mixture of Uniforms weights (non-negative with mean = variance = 1) #
#---------------------------------------------------------------------#
struct MixtureUniformWeights <: WeightScheme end

function munifm1v1(u::T) where {T <: Real}
    return ifelse(u < T(0.75), (T(4) / T(3)) * u, muladd(T(12), u, -T(8)))
end

function mixunifm1v1(
    rng::AbstractRNG, ::Type{T} = Float64
) where {T <: AbstractFloat}
    return munifm1v1(rand(rng, T))
end

function mixunifm1v1!(
    rng::AbstractRNG, x::AbstractVector{T}
) where {T <: AbstractFloat}
    rand!(rng, x)
    map!(munifm1v1, x, x)
    return nothing
end

export munifm1v1, munifm1v1_rand, munifm1v1_rand!

@inline wgts!(rng::AbstractRNG, w::Vector{Float64}, ::MixtureUniformWeights) = mixunifm1v1!(rng, w)
