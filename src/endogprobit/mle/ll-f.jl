"""
    ll(ϑ, ws::EndogProbitMLEWorkspace, wgts=nothing)

Return the negative average loglikelihood, omitting the normal-density
constant. Observations are columns of `ws.ξs`, with rows ordered as
`[X; Z; D; Y]` and outcomes `Y` coded as `-1` or `1`.

The parameter vector is ordered as `ϑ = [β; γ; π; α; τ; ρ]`, with length
`2ws.k + ws.l + 3`.

`wgts` is either `nothing` for unit weights or a real vector with one weight
per observation. Weighted contributions are divided by the number of
observations, not the sum of the weights.
"""
function ll(ϑ, ws::EndogProbitMLEWorkspace, wgts=nothing)
    n = size(ws.ξs, 2)
    k = ws.k
    l = ws.l

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
            Xβi += ws.ξs[j,i] * ϑ[rownumϑ]
            rownumϑ += 1
        end

        # Calculate X'γ.
        Xγi = 0.0
        for _ in 1:k
            Xγi += ws.ξs[rownumξ,i] * ϑ[rownumϑ]
            rownumξ += 1
            rownumϑ += 1
        end

        # Calculate Z'π.
        Zπi = 0.0
        for _ in 1:l
            Zπi += ws.ξs[rownumξ,i] * ϑ[rownumϑ]
            rownumξ += 1
            rownumϑ += 1
        end

        # Extract D and Y.
        Di = ws.ξs[rownumξ,i]
        rownumξ += 1
        Yi = ws.ξs[rownumξ,i]

        # Calculate first stage residual
        νi = Di - Xγi - Zπi
        # Calculate evaluation point for standard normal cdf for part of
        # likelihood associated with Y.
        cdfx = Yi * cdfxdenom * (Xβi + (α * Di) - (ρinvσ * νi))
        # Add likelihood increment.
        llval += wi * (normlogcdf(cdfx) - halflogτ - ((νi^2) * inv2τ))
    end
    return -llval / n
end

export ll
