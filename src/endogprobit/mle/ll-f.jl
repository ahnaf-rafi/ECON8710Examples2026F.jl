"""
    ll(ϑ, data::EndogProbitData, wgts=nothing)

Return the negative average loglikelihood, omitting the normal-density
constant. Observations are columns of `data.ξs`, with rows ordered as
`[X; Z; D; Y]` and outcomes `Y` coded as `-1` or `1`.

The parameter vector is ordered as `ϑ = [β; γ; π; α; τ; ρ]`, with length
`2data.k + data.l + 3`.

`wgts` is either `nothing` for unit weights or a real vector with one weight
per observation. Weighted contributions are divided by the number of
observations, not the sum of the weights.
"""
function ll(ϑ, data::EndogProbitData, wgts=nothing)
    n = size(data.ξs, 2)
    k = data.k
    l = data.l

    # Iterate once through back and extract params.
    rownumϑ = (2k) + l + 1
    α = ϑ[rownumϑ]
    rownumϑ += 1
    τ = ϑ[rownumϑ]
    rownumϑ += 1
    ρ = ϑ[rownumϑ]

    # The regular likelihood requires positive variance and |ρ| < 1.
    if !(isfinite(τ) && τ > 0 && abs(ρ) < 1)
        return Inf
    end

    # Avoid repeating some computations.
    invσ = inv(sqrt(τ))
    ρinvσ = ρ * invσ
    cdfxdenom = inv(sqrt((1 - ρ) * (1 + ρ)))  # More accurate if |ρ| is close to 1.
    halflogτ = log(τ) / 2
    inv2τ = 0.5 / τ

    # Initialize log likelihood value.
    llval = 0.0
    @inbounds for i in 1:n
        wi = obswgt(wgts, i)
        # Skip computation if observation weight is zero.
        iszero(wi) && continue

        # Initialize row counters for ξ and ϑ.
        rownumξ = 1
        rownumϑ = 1

        # Calculate X'β.
        Xβi = 0.0
        for j in 1:k
            Xβi += data.ξs[j,i] * ϑ[rownumϑ]
            rownumϑ += 1
        end

        # Calculate X'γ.
        Xγi = 0.0
        for _ in 1:k
            Xγi += data.ξs[rownumξ,i] * ϑ[rownumϑ]
            rownumξ += 1
            rownumϑ += 1
        end

        # Calculate Z'π.
        Zπi = 0.0
        for _ in 1:l
            Zπi += data.ξs[rownumξ,i] * ϑ[rownumϑ]
            rownumξ += 1
            rownumϑ += 1
        end

        # Extract D and Y.
        Di = data.ξs[rownumξ,i]
        rownumξ += 1
        Yi = data.ξs[rownumξ,i]

        # Calculate first stage residual
        νi = Di - Xγi - Zπi
        # Calculation evaluation point for standard normal cdf for part of
        # likelihood associated with Y.
        cdfx = Yi * cdfxdenom * (Xβi + (α * Di) - (ρinvσ * νi))
        # Add likelihood increment.
        llval += wi * (normlogcdf(cdfx) - halflogτ - ((νi^2) * inv2τ))
    end
    return -llval / n
end
