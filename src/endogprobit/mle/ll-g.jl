"""
    ll_g!(G, ϑ, ws::EndogProbitMLEWorkspace, wgts=nothing)

Overwrite `G` with the gradient of [`ll`](@ref) and return `nothing`.
Note that this is the gradient of the negative average log-likelihood.
Parameter vector is ordered as `ϑ = [β; γ; π; α; τ; ρ]`, where `τ = Var(ν) > 0`
and `ρ = Corr(ν, ϵ) ∈ (-1, 1)` lies strictly between -1 and 1.
Observations are columns of `ws.ξs`, with rows ordered as `[X; Z; D; Y]`
and outcomes `Y` coded as `-1` or `1`.

`wgts` is either `nothing` for unit weights or a real vector with one weight
per observation. Zero-weight observations are skipped.
"""
function ll_g!(G, ϑ, ws::EndogProbitMLEWorkspace, wgts=nothing)
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
        fill!(G, NaN)
        return nothing
    end

    # Avoid repeating some computations.
    invσ = inv(sqrt(τ))
    ρinvσ = ρ * invσ
    cdfxdenom = inv(sqrt((1 - ρ) * (1 + ρ)))  # More accurate if |ρ| is close to 1.
    inv2τ = 0.5 / τ
    invτ = inv(τ)
    invh² = cdfxdenom^2
    half_LOG2PI = LOG2PI / 2

    # Initialize the requested outputs.
    fill!(G, 0.0)
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
        logcdfi = normlogcdf(cdfx)

        # Evaluate the inverse Mills ratio in logs for negative cdf arguments.
        logpdfi = -(cdfx^2 / 2) - half_LOG2PI
        ri = Yi * exp(logpdfi - logcdfi)

        # Common factors in the β, γ, and π score components.
        scoreβi = ri * cdfxdenom
        scoreδi = νi * invτ + scoreβi * ρinvσ

        # Initialize row counters and fill the negative-loglikelihood gradient.
        rownumξ = 1
        rownumϑ = 1
        for j in 1:k
            G[rownumϑ] -= wi * scoreβi * ws.ξs[j,i]
            rownumϑ += 1
        end

        for _ in 1:k
            G[rownumϑ] -= wi * scoreδi * ws.ξs[rownumξ,i]
            rownumξ += 1
            rownumϑ += 1
        end

        for _ in 1:l
            G[rownumϑ] -= wi * scoreδi * ws.ξs[rownumξ,i]
            rownumξ += 1
            rownumϑ += 1
        end

        G[rownumϑ] -= wi * scoreβi * Di
        rownumϑ += 1

        # Normal-density and probit contributions to the τ score.
        scoreτi = ((νi^2) * invτ - 1 + scoreβi * ρinvσ * νi) * inv2τ
        G[rownumϑ] -= wi * scoreτi
        rownumϑ += 1

        # The normal density of ν does not depend on ρ.
        scoreρi = scoreβi * invh² * (ρ * (Xβi + α * Di) - invσ * νi)
        G[rownumϑ] -= wi * scoreρi
    end

    G ./= n
    return nothing
end

"""
    ll_fg!(G, ϑ, ws::EndogProbitMLEWorkspace, wgts=nothing)

Return the negative average loglikelihood [`ll`](@ref) and overwrite `G` with
its gradient in a single pass.
Parameter vector is ordered as `ϑ = [β; γ; π; α; τ; ρ]`, where `τ = Var(ν) > 0`
and `ρ = Corr(ν, ϵ) ∈ (-1, 1)`.
Observations are columns of `ws.ξs`, with rows ordered as `[X; Z; D; Y]`
and outcomes `Y` coded as `-1` or `1`.

`wgts` is either `nothing` for unit weights or a real vector with one weight
per observation. Zero-weight observations are skipped.
"""
function ll_fg!(G, ϑ, ws::EndogProbitMLEWorkspace, wgts=nothing)
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
        fill!(G, NaN)
        return Inf
    end

    # Avoid repeating some computations.
    invσ = inv(sqrt(τ))
    ρinvσ = ρ * invσ
    cdfxdenom = inv(sqrt((1 - ρ) * (1 + ρ)))  # More accurate if |ρ| is close to 1.
    halflogτ = log(τ) / 2
    inv2τ = 0.5 / τ
    invτ = inv(τ)
    invh² = cdfxdenom^2
    half_LOG2PI = LOG2PI / 2

    # Initialize the requested outputs.
    llval = 0.0
    fill!(G, 0.0)
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
        logcdfi = normlogcdf(cdfx)
        # Add likelihood increment.
        llval += wi * (logcdfi - halflogτ - ((νi^2) * inv2τ))

        # Evaluate the inverse Mills ratio in logs for negative cdf arguments.
        logpdfi = -(cdfx^2 / 2) - half_LOG2PI
        ri = Yi * exp(logpdfi - logcdfi)

        # Common factors in the β, γ, and π score components.
        scoreβi = ri * cdfxdenom
        scoreδi = νi * invτ + scoreβi * ρinvσ

        # Initialize row counters and fill the negative-loglikelihood gradient.
        rownumξ = 1
        rownumϑ = 1
        for j in 1:k
            G[rownumϑ] -= wi * scoreβi * ws.ξs[j,i]
            rownumϑ += 1
        end

        for _ in 1:k
            G[rownumϑ] -= wi * scoreδi * ws.ξs[rownumξ,i]
            rownumξ += 1
            rownumϑ += 1
        end

        for _ in 1:l
            G[rownumϑ] -= wi * scoreδi * ws.ξs[rownumξ,i]
            rownumξ += 1
            rownumϑ += 1
        end

        G[rownumϑ] -= wi * scoreβi * Di
        rownumϑ += 1

        # Normal-density and probit contributions to the τ score.
        scoreτi = ((νi^2) * invτ - 1 + scoreβi * ρinvσ * νi) * inv2τ
        G[rownumϑ] -= wi * scoreτi
        rownumϑ += 1

        # The normal density of ν does not depend on ρ.
        scoreρi = scoreβi * invh² * (ρ * (Xβi + α * Di) - invσ * νi)
        G[rownumϑ] -= wi * scoreρi
    end

    G ./= n
    return -llval / n
end

export ll_g!, ll_fg!
