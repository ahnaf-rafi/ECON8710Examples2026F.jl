function gen!(
    rng :: AbstractRNG, data :: EndogProbitData, dgp :: EndogProbitGaussianDGP
)
    # Generate standard Gaussian errors.
    randn!(rng, data.ξs)
    # Scale by lower triangular cholesky decomp of covariance.
    lmul!(dgp.CU0, data.ξs)

    # Column-by-column random generation of ξ.
    for i in axes(data.ξs, 2)
        _rand_endogprobit_single!(data, dgp, i)
    end

    return nothing
end

export gen!

@inline function _rand_endogprobit_single!(
    data :: EndogProbitData, dgp :: EndogProbitGaussianDGP, i :: Int
)
    Xβ0ᵢ = 0.0
    Xγ0ᵢ = 0.0
    row_num = 1
    @inbounds for j in 1:data.k
        xval = data.ξs[row_num,i]
        Xβ0ᵢ += xval * dgp.β0[j]
        Xγ0ᵢ += xval * dgp.γ0[j]
        row_num += 1
    end

    Zπ0ᵢ = 0.0
    @inbounds for j in 1:data.l
        zval = data.ξs[row_num,i]
        Zπ0ᵢ += zval * dgp.π0[j]
        row_num += 1
    end

    @inbounds begin
        D = Xγ0ᵢ + Zπ0ᵢ + data.ξs[row_num, i]
        data.ξs[row_num, i] = D
        row_num += 1
        u = Xβ0ᵢ + (dgp.α0 * D) - data.ξs[row_num, i]
        data.ξs[row_num, i] = ifelse(u ≥ 0, 1.0, -1.0)
    end
    return nothing
end

function gen1!(
    rng :: AbstractRNG, data :: EndogProbitData, dgp :: EndogProbitGaussianDGP
)
    # Generate standard Gaussian errors.
    randn!(rng, data.ξs)
    # Scale by lower triangular cholesky decomp of covariance.
    lmul!(dgp.CU0, data.ξs)

    k = data.k
    l = data.l

    @inbounds for i in axes(data.ξs, 2)
        Xβ0ᵢ = 0.0
        Xγ0ᵢ = 0.0
        row_num = 1
        for j in 1:k
            xval = data.ξs[row_num,i]
            Xβ0ᵢ += xval * dgp.β0[j]
            Xγ0ᵢ += xval * dgp.γ0[j]
            row_num += 1
        end

        Zπ0ᵢ = 0.0
        for j in 1:l
            zval = data.ξs[row_num,i]
            Zπ0ᵢ += zval * dgp.π0[j]
            row_num += 1
        end

        D = Xγ0ᵢ + Zπ0ᵢ + data.ξs[row_num, i]
        data.ξs[row_num, i] = D
        row_num += 1
        u = Xβ0ᵢ + (dgp.α0 * D) - data.ξs[row_num, i]
        data.ξs[row_num, i] = ifelse(u ≥ 0, 1.0, -1.0)
    end
    return nothing
end

export gen1!
