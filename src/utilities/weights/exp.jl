#---------------------#
# Exponential weights #
#---------------------#
struct ExponentialWeights <: WeightScheme end
wgts!(rng::AbstractRNG, w::Vector{Float64}, ::ExponentialWeights) = randexp!(rng, w)

export ExponentialWeights
