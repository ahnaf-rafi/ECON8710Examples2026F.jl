#------------------#
# Gaussian weights #
#------------------#
struct GaussianWeights <: WeightScheme end
function wgts!(rng::AbstractRNG, w::Vector{Float64}, ::GaussianWeights)
    randn!(rng,w)
    w .+= 1.0
    return w
end

export GaussianWeights
