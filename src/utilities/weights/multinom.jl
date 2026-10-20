#---------------------#
# Multinomial weights #
#---------------------#
struct MultinomialWeights <: WeightScheme end

function wgts!(rng::AbstractRNG, w::Vector{Int}, ::MultinomialWeights)
    n = length(w)
    # Don't do anything if n == 0.
    n == 0 && return w

    fill!(w, 0)
    s = Random.Sampler(rng, 1:n)  # Precompute the sampler.
    for _ in 1:n
        w[rand(rng, s)] += 1
    end
    return w
end
