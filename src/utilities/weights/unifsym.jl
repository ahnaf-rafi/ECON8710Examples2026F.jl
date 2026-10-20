#------------------------------------------------#
# Symmetric uniform distribution: Uniform[-a, a] #
#------------------------------------------------#
struct SymmetricUniformWeights <: WeightScheme end

function unifsym(u::T, a::T) where {T<:AbstractFloat}
    return a * muladd(T(2), u, -one(T))
end

function unifsym_rand(
    rng::AbstractRNG, a::T=1.0
) where {T<:AbstractFloat}
    return unifsym(rand(rng, T), a)
end

function unifsym_rand!(
    rng::AbstractRNG, x::AbstractVector{T}, a::Real=one(T)
) where {T<:AbstractFloat}
    scale = T(a)
    rand!(rng, x)
    map!(u -> unifsym(u, scale), x, x)
    return nothing
end

export unifsym, unifsym_rand, unifsym_rand!

#----------------------------------------------------#
# Symmetric uniform weights with mean = variance = 1 #
#----------------------------------------------------#

function wgts!(rng::AbstractRNG, w::Vector{Float64}, ::SymmetricUniformWeights)
    unifsym_rand!(rng, w, sqrt(3.0))
    w .+= 1.0
end
