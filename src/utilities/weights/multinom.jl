#---------------------#
# Multinomial weights #
#---------------------#
struct MultinomialWeights <: WeightScheme end

function wgts!(
    rng::AbstractRNG, w::AbstractVector{T}, s::Random.Sampler , ::MultinomialWeights
) where T
    # This method assumes a precomputed sampler s = Random.Sampler(rng, 1:n) has
    # been passed.

    n = length(w)
    # Don't do anything if n == 0.
    n == 0 && return w

    fill!(w, zero(T))
    @inbounds for _ in 1:n
        w[rand(rng, s)] += one(T)
    end
    return w
end

function wgts!(
    rng::AbstractRNG, w::AbstractVector{T}, ::MultinomialWeights
) where T
    n = length(w)
    # Don't do anything if n == 0.
    n == 0 && return w

    fill!(w, zero(T))
    s = Random.Sampler(rng, 1:n)  # Precompute the sampler.
    @inbounds for _ in 1:n
        w[rand(rng, s)] += one(T)
    end
    return w
end

export MultinomialWeights
