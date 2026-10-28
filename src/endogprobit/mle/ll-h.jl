"""
    ll_h!(H, ϑ, ws::EndogProbitMLEWorkspace, wgts=nothing)

Overwrite `H` with the Hessian of [`ll`](@ref) and return `nothing`.
Note that this is the Hessian of the negative average log-likelihood.
Parameter vector is ordered as `ϑ = [β; γ; π; α; τ; ρ]`, where `τ = Var(ν) > 0`
and `ρ = Corr(ν, ϵ) ∈ (-1, 1)` lies strictly between -1 and 1.
Observations are columns of `ws.ξs`, with rows ordered as `[X; Z; D; Y]`
and outcomes `Y` coded as `-1` or `1`.

`wgts` is either `nothing` for unit weights or a real vector with one weight
per observation. Zero-weight observations are skipped.
"""
function ll_h!(H, ϑ, ws::EndogProbitMLEWorkspace, wgts=nothing)
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
        fill!(H, NaN)
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
    fill!(H, 0.0)
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

        # Derivatives of the unsigned probit index q with respect to
        # regression indices, variance τ, and correlation ρ.
        qi = cdfxdenom * (Xβi + α * Di - ρinvσ * νi)
        rqi = -ri * (qi + ri)
        dqβi = cdfxdenom
        dqδi = ρinvσ * cdfxdenom
        dqτi = dqδi * νi * inv2τ
        dqρi = cdfxdenom * invh² * (ρ * (Xβi + α * Di) - invσ * νi)

        # Common factors in the loglikelihood Hessian blocks. δ denotes
        # the combined first-stage coefficients [γ; π].
        curvββi = rqi * dqβi^2
        curvδβi = rqi * dqδi * dqβi
        curvδδi = rqi * dqδi^2 - invτ
        curvτβi = rqi * dqτi * dqβi
        curvρβi = rqi * dqρi * dqβi + ri * ρ * cdfxdenom * invh²
        curvτδi = rqi * dqτi * dqδi - ri * dqδi * inv2τ - νi * invτ^2
        curvρδi = rqi * dqρi * dqδi + ri * invσ * cdfxdenom * invh²

        # Iterate and fill columns of the lower triangle of H.
        colnumξ = 1
        colnumϑ = 1

        for j in 1:k
            rownumξ = 1
            rownumϑ = colnumϑ

            for a in j:k
                H[rownumϑ,colnumϑ] -= wi * curvββi * ws.ξs[a,i] * ws.ξs[j,i]
                rownumϑ += 1
            end

            for _ in 1:k
                H[rownumϑ,colnumϑ] -= wi * curvδβi *
                    ws.ξs[rownumξ,i] * ws.ξs[j,i]
                rownumξ += 1
                rownumϑ += 1
            end

            for _ in 1:l
                H[rownumϑ,colnumϑ] -= wi * curvδβi *
                    ws.ξs[rownumξ,i] * ws.ξs[j,i]
                rownumξ += 1
                rownumϑ += 1
            end

            H[rownumϑ,colnumϑ] -= wi * curvββi * Di * ws.ξs[j,i]
            rownumϑ += 1
            H[rownumϑ,colnumϑ] -= wi * curvτβi * ws.ξs[j,i]
            rownumϑ += 1
            H[rownumϑ,colnumϑ] -= wi * curvρβi * ws.ξs[j,i]
            colnumϑ += 1
        end

        for j in 1:k
            rownumξ = colnumξ
            rownumϑ = colnumϑ

            for _ in j:k
                H[rownumϑ,colnumϑ] -= wi * curvδδi *
                    ws.ξs[rownumξ,i] * ws.ξs[colnumξ,i]
                rownumξ += 1
                rownumϑ += 1
            end

            for _ in 1:l
                H[rownumϑ,colnumϑ] -= wi * curvδδi *
                    ws.ξs[rownumξ,i] * ws.ξs[colnumξ,i]
                rownumξ += 1
                rownumϑ += 1
            end

            H[rownumϑ,colnumϑ] -= wi * curvδβi * Di * ws.ξs[colnumξ,i]
            rownumϑ += 1
            H[rownumϑ,colnumϑ] -= wi * curvτδi * ws.ξs[colnumξ,i]
            rownumϑ += 1
            H[rownumϑ,colnumϑ] -= wi * curvρδi * ws.ξs[colnumξ,i]
            colnumξ += 1
            colnumϑ += 1
        end

        for j in 1:l
            rownumξ = colnumξ
            rownumϑ = colnumϑ

            for _ in j:l
                H[rownumϑ,colnumϑ] -= wi * curvδδi *
                    ws.ξs[rownumξ,i] * ws.ξs[colnumξ,i]
                rownumξ += 1
                rownumϑ += 1
            end

            H[rownumϑ,colnumϑ] -= wi * curvδβi * Di * ws.ξs[colnumξ,i]
            rownumϑ += 1
            H[rownumϑ,colnumϑ] -= wi * curvτδi * ws.ξs[colnumξ,i]
            rownumϑ += 1
            H[rownumϑ,colnumϑ] -= wi * curvρδi * ws.ξs[colnumξ,i]
            colnumξ += 1
            colnumϑ += 1
        end

        rownumϑ = colnumϑ
        H[rownumϑ,colnumϑ] -= wi * curvββi * Di^2
        rownumϑ += 1
        H[rownumϑ,colnumϑ] -= wi * curvτβi * Di
        rownumϑ += 1
        H[rownumϑ,colnumϑ] -= wi * curvρβi * Di
        colnumϑ += 1

        # Normal-density and probit contributions to the τ and ρ blocks.
        ddqττi = -3 * dqτi * inv2τ
        ddqρτi = νi * invσ * cdfxdenom * invh² * inv2τ
        ddqρρi = 3 * ρ * invh² * dqρi +
            (Xβi + α * Di) * cdfxdenom * invh²
        curvττi = (0.5 - (νi^2) * invτ) * invτ^2 +
            rqi * dqτi^2 + ri * ddqττi
        curvρτi = rqi * dqρi * dqτi + ri * ddqρτi
        curvρρi = rqi * dqρi^2 + ri * ddqρρi

        rownumϑ = colnumϑ
        H[rownumϑ,colnumϑ] -= wi * curvττi
        rownumϑ += 1
        H[rownumϑ,colnumϑ] -= wi * curvρτi
        colnumϑ += 1
        H[rownumϑ,colnumϑ] -= wi * curvρρi
    end

    # Copy the lower triangle to the upper triangle.
    @inbounds for j in 1:((2k) + l + 3)
        for a in j:((2k) + l + 3)
            H[j,a] = H[a,j]
        end
    end

    H ./= n
    return nothing
end

"""
    ll_fgh!(G, H, ϑ, ws::EndogProbitMLEWorkspace, wgts=nothing)

Return [`ll`](@ref) and overwrite `G` and `H` with its gradient and
Hessian in a single pass through the observations.
Note that this returns the negative average log-likelihood, and the gradient and
Hessian thereof.
Parameter vector is ordered as `ϑ = [β; γ; π; α; τ; ρ]`, where `τ = Var(ν) > 0`
and `ρ = Corr(ν, ϵ) ∈ (-1, 1)`.
Observations are columns of `ws.ξs`, with rows ordered as `[X; Z; D; Y]`
and outcomes `Y` coded as `-1` or `1`.

`wgts` is either `nothing` for unit weights or a real vector with one weight
per observation. Zero-weight observations are skipped.
"""
function ll_fgh!(G, H, ϑ, ws::EndogProbitMLEWorkspace, wgts=nothing)
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
        fill!(H, NaN)
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
    fill!(H, 0.0)
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

        # Derivatives of the unsigned probit index q with respect to
        # regression indices, variance τ, and correlation ρ.
        qi = cdfxdenom * (Xβi + α * Di - ρinvσ * νi)
        rqi = -ri * (qi + ri)
        dqβi = cdfxdenom
        dqδi = ρinvσ * cdfxdenom
        dqτi = dqδi * νi * inv2τ
        dqρi = cdfxdenom * invh² * (ρ * (Xβi + α * Di) - invσ * νi)

        # Common factors in the loglikelihood Hessian blocks. δ denotes
        # the combined first-stage coefficients [γ; π].
        curvββi = rqi * dqβi^2
        curvδβi = rqi * dqδi * dqβi
        curvδδi = rqi * dqδi^2 - invτ
        curvτβi = rqi * dqτi * dqβi
        curvρβi = rqi * dqρi * dqβi + ri * ρ * cdfxdenom * invh²
        curvτδi = rqi * dqτi * dqδi - ri * dqδi * inv2τ - νi * invτ^2
        curvρδi = rqi * dqρi * dqδi + ri * invσ * cdfxdenom * invh²

        # Iterate and fill columns of the lower triangle of H.
        colnumξ = 1
        colnumϑ = 1

        for j in 1:k
            rownumξ = 1
            rownumϑ = colnumϑ

            for a in j:k
                H[rownumϑ,colnumϑ] -= wi * curvββi * ws.ξs[a,i] * ws.ξs[j,i]
                rownumϑ += 1
            end

            for _ in 1:k
                H[rownumϑ,colnumϑ] -= wi * curvδβi *
                    ws.ξs[rownumξ,i] * ws.ξs[j,i]
                rownumξ += 1
                rownumϑ += 1
            end

            for _ in 1:l
                H[rownumϑ,colnumϑ] -= wi * curvδβi *
                    ws.ξs[rownumξ,i] * ws.ξs[j,i]
                rownumξ += 1
                rownumϑ += 1
            end

            H[rownumϑ,colnumϑ] -= wi * curvββi * Di * ws.ξs[j,i]
            rownumϑ += 1
            H[rownumϑ,colnumϑ] -= wi * curvτβi * ws.ξs[j,i]
            rownumϑ += 1
            H[rownumϑ,colnumϑ] -= wi * curvρβi * ws.ξs[j,i]
            colnumϑ += 1
        end

        for j in 1:k
            rownumξ = colnumξ
            rownumϑ = colnumϑ

            for _ in j:k
                H[rownumϑ,colnumϑ] -= wi * curvδδi *
                    ws.ξs[rownumξ,i] * ws.ξs[colnumξ,i]
                rownumξ += 1
                rownumϑ += 1
            end

            for _ in 1:l
                H[rownumϑ,colnumϑ] -= wi * curvδδi *
                    ws.ξs[rownumξ,i] * ws.ξs[colnumξ,i]
                rownumξ += 1
                rownumϑ += 1
            end

            H[rownumϑ,colnumϑ] -= wi * curvδβi * Di * ws.ξs[colnumξ,i]
            rownumϑ += 1
            H[rownumϑ,colnumϑ] -= wi * curvτδi * ws.ξs[colnumξ,i]
            rownumϑ += 1
            H[rownumϑ,colnumϑ] -= wi * curvρδi * ws.ξs[colnumξ,i]
            colnumξ += 1
            colnumϑ += 1
        end

        for j in 1:l
            rownumξ = colnumξ
            rownumϑ = colnumϑ

            for _ in j:l
                H[rownumϑ,colnumϑ] -= wi * curvδδi *
                    ws.ξs[rownumξ,i] * ws.ξs[colnumξ,i]
                rownumξ += 1
                rownumϑ += 1
            end

            H[rownumϑ,colnumϑ] -= wi * curvδβi * Di * ws.ξs[colnumξ,i]
            rownumϑ += 1
            H[rownumϑ,colnumϑ] -= wi * curvτδi * ws.ξs[colnumξ,i]
            rownumϑ += 1
            H[rownumϑ,colnumϑ] -= wi * curvρδi * ws.ξs[colnumξ,i]
            colnumξ += 1
            colnumϑ += 1
        end

        rownumϑ = colnumϑ
        H[rownumϑ,colnumϑ] -= wi * curvββi * Di^2
        rownumϑ += 1
        H[rownumϑ,colnumϑ] -= wi * curvτβi * Di
        rownumϑ += 1
        H[rownumϑ,colnumϑ] -= wi * curvρβi * Di
        colnumϑ += 1

        # Normal-density and probit contributions to the τ and ρ blocks.
        ddqττi = -3 * dqτi * inv2τ
        ddqρτi = νi * invσ * cdfxdenom * invh² * inv2τ
        ddqρρi = 3 * ρ * invh² * dqρi +
            (Xβi + α * Di) * cdfxdenom * invh²
        curvττi = (0.5 - (νi^2) * invτ) * invτ^2 +
            rqi * dqτi^2 + ri * ddqττi
        curvρτi = rqi * dqρi * dqτi + ri * ddqρτi
        curvρρi = rqi * dqρi^2 + ri * ddqρρi

        rownumϑ = colnumϑ
        H[rownumϑ,colnumϑ] -= wi * curvττi
        rownumϑ += 1
        H[rownumϑ,colnumϑ] -= wi * curvρτi
        colnumϑ += 1
        H[rownumϑ,colnumϑ] -= wi * curvρρi
    end

    # Copy the lower triangle to the upper triangle.
    @inbounds for j in 1:((2k) + l + 3)
        for a in j:((2k) + l + 3)
            H[j,a] = H[a,j]
        end
    end

    G ./= n
    H ./= n
    return -llval / n
end

"""
    ll_gh!(G, H, ϑ, ws::EndogProbitMLEWorkspace, wgts=nothing)

Overwrite `G` and `H` with the gradient and Hessian of [`ll`](@ref)
in a single pass through the observations and return `nothing`.
Note that this returns the negative average log-likelihood, and the gradient and
Hessian thereof.

Parameter vector is ordered as `ϑ = [β; γ; π; α; τ; ρ]`, where `τ = Var(ν) > 0`
and `ρ = Corr(ν, ϵ) ∈ (-1, 1)`.
Observations are columns of `ws.ξs`, with rows ordered as `[X; Z; D; Y]`
and outcomes `Y` coded as `-1` or `1`.

`wgts` is either `nothing` for unit weights or a real vector with one weight
per observation. Zero-weight observations are skipped.
"""
function ll_gh!(G, H, ϑ, ws::EndogProbitMLEWorkspace, wgts=nothing)
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
        fill!(H, NaN)
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
    fill!(H, 0.0)
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

        # Derivatives of the unsigned probit index q with respect to
        # regression indices, variance τ, and correlation ρ.
        qi = cdfxdenom * (Xβi + α * Di - ρinvσ * νi)
        rqi = -ri * (qi + ri)
        dqβi = cdfxdenom
        dqδi = ρinvσ * cdfxdenom
        dqτi = dqδi * νi * inv2τ
        dqρi = cdfxdenom * invh² * (ρ * (Xβi + α * Di) - invσ * νi)

        # Common factors in the loglikelihood Hessian blocks. δ denotes
        # the combined first-stage coefficients [γ; π].
        curvββi = rqi * dqβi^2
        curvδβi = rqi * dqδi * dqβi
        curvδδi = rqi * dqδi^2 - invτ
        curvτβi = rqi * dqτi * dqβi
        curvρβi = rqi * dqρi * dqβi + ri * ρ * cdfxdenom * invh²
        curvτδi = rqi * dqτi * dqδi - ri * dqδi * inv2τ - νi * invτ^2
        curvρδi = rqi * dqρi * dqδi + ri * invσ * cdfxdenom * invh²

        # Iterate and fill columns of the lower triangle of H.
        colnumξ = 1
        colnumϑ = 1

        for j in 1:k
            rownumξ = 1
            rownumϑ = colnumϑ

            for a in j:k
                H[rownumϑ,colnumϑ] -= wi * curvββi * ws.ξs[a,i] * ws.ξs[j,i]
                rownumϑ += 1
            end

            for _ in 1:k
                H[rownumϑ,colnumϑ] -= wi * curvδβi *
                    ws.ξs[rownumξ,i] * ws.ξs[j,i]
                rownumξ += 1
                rownumϑ += 1
            end

            for _ in 1:l
                H[rownumϑ,colnumϑ] -= wi * curvδβi *
                    ws.ξs[rownumξ,i] * ws.ξs[j,i]
                rownumξ += 1
                rownumϑ += 1
            end

            H[rownumϑ,colnumϑ] -= wi * curvββi * Di * ws.ξs[j,i]
            rownumϑ += 1
            H[rownumϑ,colnumϑ] -= wi * curvτβi * ws.ξs[j,i]
            rownumϑ += 1
            H[rownumϑ,colnumϑ] -= wi * curvρβi * ws.ξs[j,i]
            colnumϑ += 1
        end

        for j in 1:k
            rownumξ = colnumξ
            rownumϑ = colnumϑ

            for _ in j:k
                H[rownumϑ,colnumϑ] -= wi * curvδδi *
                    ws.ξs[rownumξ,i] * ws.ξs[colnumξ,i]
                rownumξ += 1
                rownumϑ += 1
            end

            for _ in 1:l
                H[rownumϑ,colnumϑ] -= wi * curvδδi *
                    ws.ξs[rownumξ,i] * ws.ξs[colnumξ,i]
                rownumξ += 1
                rownumϑ += 1
            end

            H[rownumϑ,colnumϑ] -= wi * curvδβi * Di * ws.ξs[colnumξ,i]
            rownumϑ += 1
            H[rownumϑ,colnumϑ] -= wi * curvτδi * ws.ξs[colnumξ,i]
            rownumϑ += 1
            H[rownumϑ,colnumϑ] -= wi * curvρδi * ws.ξs[colnumξ,i]
            colnumξ += 1
            colnumϑ += 1
        end

        for j in 1:l
            rownumξ = colnumξ
            rownumϑ = colnumϑ

            for _ in j:l
                H[rownumϑ,colnumϑ] -= wi * curvδδi *
                    ws.ξs[rownumξ,i] * ws.ξs[colnumξ,i]
                rownumξ += 1
                rownumϑ += 1
            end

            H[rownumϑ,colnumϑ] -= wi * curvδβi * Di * ws.ξs[colnumξ,i]
            rownumϑ += 1
            H[rownumϑ,colnumϑ] -= wi * curvτδi * ws.ξs[colnumξ,i]
            rownumϑ += 1
            H[rownumϑ,colnumϑ] -= wi * curvρδi * ws.ξs[colnumξ,i]
            colnumξ += 1
            colnumϑ += 1
        end

        rownumϑ = colnumϑ
        H[rownumϑ,colnumϑ] -= wi * curvββi * Di^2
        rownumϑ += 1
        H[rownumϑ,colnumϑ] -= wi * curvτβi * Di
        rownumϑ += 1
        H[rownumϑ,colnumϑ] -= wi * curvρβi * Di
        colnumϑ += 1

        # Normal-density and probit contributions to the τ and ρ blocks.
        ddqττi = -3 * dqτi * inv2τ
        ddqρτi = νi * invσ * cdfxdenom * invh² * inv2τ
        ddqρρi = 3 * ρ * invh² * dqρi +
            (Xβi + α * Di) * cdfxdenom * invh²
        curvττi = (0.5 - (νi^2) * invτ) * invτ^2 +
            rqi * dqτi^2 + ri * ddqττi
        curvρτi = rqi * dqρi * dqτi + ri * ddqρτi
        curvρρi = rqi * dqρi^2 + ri * ddqρρi

        rownumϑ = colnumϑ
        H[rownumϑ,colnumϑ] -= wi * curvττi
        rownumϑ += 1
        H[rownumϑ,colnumϑ] -= wi * curvρτi
        colnumϑ += 1
        H[rownumϑ,colnumϑ] -= wi * curvρρi
    end

    # Copy the lower triangle to the upper triangle.
    @inbounds for j in 1:((2k) + l + 3)
        for a in j:((2k) + l + 3)
            H[j,a] = H[a,j]
        end
    end

    G ./= n
    H ./= n
    return nothing
end

export ll_h!, ll_gh!, ll_fgh!
