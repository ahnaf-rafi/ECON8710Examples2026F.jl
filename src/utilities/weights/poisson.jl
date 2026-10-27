#-----------------#
# Poisson weights #
#-----------------#
struct PoissonWeights <: WeightScheme end

function wgts!(
    rng::AbstractRNG, w::AbstractVector{T}, s::Random.Sampler, ::PoissonWeights
) where T
    n = length(w)
    # Don't do anything if n == 0.
    n == 0 && return w

    fill!(w, 0)
    m = rand(rng, Poisson(n))
    @inbounds for _ in 1:m
        w[rand(rng, s)] += one(T)
    end
    return w
end

function wgts!(rng::AbstractRNG, w::AbstractVector{T}, ::PoissonWeights) where T
    n = length(w)
    # Don't do anything if n == 0.
    n == 0 && return w

    fill!(w, 0)
    m = rand(rng, Poisson(n))
    s = Random.Sampler(rng, 1:n)
    @inbounds for _ in 1:m
        w[rand(rng, s)] += one(T)
    end
    return w
end
