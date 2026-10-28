function vcov_mle!(ϑhat :: Vector{Float64}, ws :: EndogProbitMLEWorkspace, wgts = nothing)
    n = size(ws.ξs, 2)
    d = (2ws.k) + ws.l + 3

    sumw = isnothing(wgts) ?  n : sum(wgts)

    # Both routines use linear weights and divide by the original n.
    ll_h!(ws.H, ϑhat, ws, wgts)
    ll_ggp!(ws.Ω, ϑhat, ws, wgts)

    # Solve H * Hinv = I using preallocated storage.
    fill!(ws.Hinv, 0.0)
    @inbounds for j in 1:d
        ws.Hinv[j, j] = 1.0
    end

    FH = cholesky!(Symmetric(ws.H, :U))
    ldiv!(FH, ws.Hinv)

    mul!(ws.tmp, ws.Hinv, ws.Ω)
    mul!(ws.V, ws.tmp, transpose(ws.Hinv))

    # Correct for raw weight normalization and estimator covariance scaling.
    ws.V .*= (sumw / n) / n

    # Remove numerical asymmetry.
    @inbounds for j in 1:d
        for k in 1:j-1
            val = (ws.V[k, j] + ws.V[j, k]) / 2
            ws.V[k, j] = val
            ws.V[j, k] = val
        end
    end

    return nothing
end

export vcov_mle!
