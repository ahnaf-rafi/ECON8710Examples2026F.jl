function ll_fg!(G, θ, data::Matrix{Float64}, ::Probit, wgts=nothing)
    n = size(data, 2)
    k = length(θ)
    half_LOG2PI = LOG2PI / 2

    llval = 0.0
    fill!(G, 0.0)
    @inbounds for i in 1:n
        wi = obswgt(wgts, i)
        # Skip computation if observation weight is zero.
        iszero(wi) && continue

        yval = data[end, i]
        Xi = @views data[1:k, i]
        cdfx = yval * dot(θ, Xi)
        logcdfi = normlogcdf(cdfx)
        llval += wi * logcdfi

        logpdfi = -(cdfx^2 / 2) - half_LOG2PI # Avoid underflow.
        ri = wi * yval * exp(logpdfi - logcdfi)
        for j in 1:k
            G[j] -= ri * Xi[j]
        end
    end

    G ./= n
    return -llval / n
end

function ll_g!(G, θ, data::Matrix{Float64}, ::Probit, wgts=nothing)
    n = size(data, 2)
    k = length(θ)
    half_LOG2PI = LOG2PI / 2

    fill!(G, 0.0)
    @inbounds for i in 1:n
        wi = obswgt(wgts, i)
        # Skip computation if observation weight is zero.
        iszero(wi) && continue

        yval = data[end, i]
        Xi = @views data[1:k, i]
        cdfx = yval * dot(θ, Xi)
        logcdfi = normlogcdf(cdfx)

        logpdfi = -(cdfx^2 / 2) - half_LOG2PI  # Avoid underflow.
        ri = wi * yval * exp(logpdfi - logcdfi)
        for j in 1:k
            G[j] -= ri * Xi[j]
        end
    end

    G ./= n
    return nothing
end
