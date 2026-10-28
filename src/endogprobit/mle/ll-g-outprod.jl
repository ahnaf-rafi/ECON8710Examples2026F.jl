"""
    ll_ggp!(GGp, ϑ::Vector{Float64}, ws::EndogProbitMLEWorkspace, wgts=nothing)

Overwrite `GGp` with the average outer product of observation-level gradients.
If `gi` is the gradient of observation i's unweighted negative loglikelihood,
the result is `sum(wi * gi * gi') / n`.

Parameter vector is ordered as `ϑ = [β; γ; π; α; τ; ρ]`, where `τ = Var(ν) > 0`
and `ρ = Corr(ν, ϵ) ∈ (-1, 1)`.
Observations are columns of `ws.ξs`, with rows ordered as `[X; Z; D; Y]`
and outcomes `Y` coded as `-1` or `1`.

`wgts` is either `nothing` for unit weights or a real vector with one weight
per observation. Zero-weight observations are skipped.
"""
function ll_ggp!(GGp, ϑ::Vector{Float64}, ws::EndogProbitMLEWorkspace, wgts=nothing)
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
        fill!(GGp, NaN)
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
    fill!(GGp, 0.0)
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

        # The outer product is unchanged by reversing the score's sign.
        scoreτi = ((νi^2) * invτ - 1 + scoreβi * ρinvσ * νi) * inv2τ
        scoreρi = scoreβi * invh² * (ρ * (Xβi + α * Di) - invσ * νi)

        # Common factors in the outer-product blocks. δ denotes [γ; π].
        prodββi = scoreβi^2
        prodδβi = scoreδi * scoreβi
        prodδδi = scoreδi^2
        prodτβi = scoreτi * scoreβi
        prodρβi = scoreρi * scoreβi
        prodτδi = scoreτi * scoreδi
        prodρδi = scoreρi * scoreδi

        # Iterate and fill columns of the lower triangle of GGp.
        colnumξ = 1
        colnumϑ = 1

        for j in 1:k
            rownumξ = 1
            rownumϑ = colnumϑ

            for a in j:k
                GGp[rownumϑ,colnumϑ] += wi * prodββi * ws.ξs[a,i] * ws.ξs[j,i]
                rownumϑ += 1
            end

            for _ in 1:k
                GGp[rownumϑ,colnumϑ] += wi * prodδβi *
                    ws.ξs[rownumξ,i] * ws.ξs[j,i]
                rownumξ += 1
                rownumϑ += 1
            end

            for _ in 1:l
                GGp[rownumϑ,colnumϑ] += wi * prodδβi *
                    ws.ξs[rownumξ,i] * ws.ξs[j,i]
                rownumξ += 1
                rownumϑ += 1
            end

            GGp[rownumϑ,colnumϑ] += wi * prodββi * Di * ws.ξs[j,i]
            rownumϑ += 1
            GGp[rownumϑ,colnumϑ] += wi * prodτβi * ws.ξs[j,i]
            rownumϑ += 1
            GGp[rownumϑ,colnumϑ] += wi * prodρβi * ws.ξs[j,i]
            colnumϑ += 1
        end

        for j in 1:k
            rownumξ = colnumξ
            rownumϑ = colnumϑ

            for _ in j:k
                GGp[rownumϑ,colnumϑ] += wi * prodδδi *
                    ws.ξs[rownumξ,i] * ws.ξs[colnumξ,i]
                rownumξ += 1
                rownumϑ += 1
            end

            for _ in 1:l
                GGp[rownumϑ,colnumϑ] += wi * prodδδi *
                    ws.ξs[rownumξ,i] * ws.ξs[colnumξ,i]
                rownumξ += 1
                rownumϑ += 1
            end

            GGp[rownumϑ,colnumϑ] += wi * prodδβi * Di * ws.ξs[colnumξ,i]
            rownumϑ += 1
            GGp[rownumϑ,colnumϑ] += wi * prodτδi * ws.ξs[colnumξ,i]
            rownumϑ += 1
            GGp[rownumϑ,colnumϑ] += wi * prodρδi * ws.ξs[colnumξ,i]
            colnumξ += 1
            colnumϑ += 1
        end

        for j in 1:l
            rownumξ = colnumξ
            rownumϑ = colnumϑ

            for _ in j:l
                GGp[rownumϑ,colnumϑ] += wi * prodδδi *
                    ws.ξs[rownumξ,i] * ws.ξs[colnumξ,i]
                rownumξ += 1
                rownumϑ += 1
            end

            GGp[rownumϑ,colnumϑ] += wi * prodδβi * Di * ws.ξs[colnumξ,i]
            rownumϑ += 1
            GGp[rownumϑ,colnumϑ] += wi * prodτδi * ws.ξs[colnumξ,i]
            rownumϑ += 1
            GGp[rownumϑ,colnumϑ] += wi * prodρδi * ws.ξs[colnumξ,i]
            colnumξ += 1
            colnumϑ += 1
        end

        rownumϑ = colnumϑ
        GGp[rownumϑ,colnumϑ] += wi * prodββi * Di^2
        rownumϑ += 1
        GGp[rownumϑ,colnumϑ] += wi * prodτβi * Di
        rownumϑ += 1
        GGp[rownumϑ,colnumϑ] += wi * prodρβi * Di
        colnumϑ += 1

        # Fill the final variance/correlation block.
        rownumϑ = colnumϑ
        GGp[rownumϑ,colnumϑ] += wi * scoreτi^2
        rownumϑ += 1
        GGp[rownumϑ,colnumϑ] += wi * scoreρi * scoreτi
        colnumϑ += 1
        GGp[rownumϑ,colnumϑ] += wi * scoreρi^2
    end

    # Copy the lower triangle to the upper triangle.
    @inbounds for j in 1:((2k) + l + 3)
        for a in j:((2k) + l + 3)
            GGp[j,a] = GGp[a,j]
        end
    end

    GGp ./= n
    return nothing
end

export ll_ggp!
