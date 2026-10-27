function ll(θ, data::Matrix{Float64}, ::Probit, wgts=nothing)
    n = size(data, 2)
    k = length(θ)

    llval = 0.0
    @inbounds for i in 1:n
        wi = obswgt(wgts, i)
        # Skip computation if observation weight is zero.
        iszero(wi) && continue

        yval = data[end, i]
        Xi = @views data[1:k, i]
        llval += wi * normlogcdf(yval * dot(θ, Xi))
    end

    return -llval / n
end
