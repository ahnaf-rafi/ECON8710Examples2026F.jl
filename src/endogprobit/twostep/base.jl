struct EndogProbit2StepWorkspace
    ξhat :: Matrix{Float64} # [νhat, D, X', Y]
    ξols :: Matrix{Float64} # [X', Z', D]
    δhat :: Vector{Float64}

    # Blocks of the stacked moment Jacobian.
    SS   :: Matrix{Float64} # p × p.
    HH   :: Matrix{Float64} # q × q.
    CS   :: Matrix{Float64} # q × p.
    v    :: Vector{Float64} # p.

    # Full sandwich covariance, ordered [δ; τ; θ].
    Jinv :: Matrix{Float64} # Inverse of summed moment Jacobian.
    Ω    :: Matrix{Float64} # Sum of moment outer products.
    V    :: Matrix{Float64} # Covariance matrix of [δhat; τhat; θhat].
    tmp  :: Matrix{Float64} # Matrix multiplication workspace.
    m    :: Vector{Float64} # Observation-level moments.

    # Delta-method workspace for alpha.
    gα   :: Vector{Float64} # Gradient of alpha.
    Vgα  :: Vector{Float64} # V * gα.
end

function EndogProbit2StepWorkspace(n::Int, k::Int, l::Int)
    p = k + l
    q = k + 2
    d = p + 1 + q

    return EndogProbit2StepWorkspace(
        Matrix{Float64}(undef, q + 1, n),
        Matrix{Float64}(undef, p + 1, n),
        Vector{Float64}(undef, p),
        Matrix{Float64}(undef, p, p),
        Matrix{Float64}(undef, q, q),
        Matrix{Float64}(undef, q, p),
        Vector{Float64}(undef, p),
        Matrix{Float64}(undef, d, d),
        Matrix{Float64}(undef, d, d),
        Matrix{Float64}(undef, d, d),
        Matrix{Float64}(undef, d, d),
        Vector{Float64}(undef, d),
        Vector{Float64}(undef, d),
        Vector{Float64}(undef, d)
    )
end

function convert!(ws2step::EndogProbit2StepWorkspace, wsmle::EndogProbitMLEWorkspace)
    k = wsmle.k
    l = wsmle.l

    @inbounds for i in axes(wsmle.ξs, 2)
        # OLS data: [X; Z; D].
        for j in 1:k+l+1
            ws2step.ξols[j, i] = wsmle.ξs[j, i]
        end

        # Second-stage data: [νhat; D; X; Y].
        ws2step.ξhat[1, i] = 0.0  # Filled by residualize! after OLS.
        ws2step.ξhat[2, i] = wsmle.ξs[k+l+1, i]
        for j in 1:k
            ws2step.ξhat[j+2, i] = wsmle.ξs[j, i]
        end
        ws2step.ξhat[k+3, i] = wsmle.ξs[k+l+2, i]
    end

    return nothing
end

export convert!

function residualize!(ws::EndogProbit2StepWorkspace)
    @inbounds for i in axes(ws.ξhat, 2)
        Si = @view ws.ξols[1:end-1, i]
        ws.ξhat[1, i] = ws.ξols[end, i] - dot(ws.δhat, Si)
    end

    return nothing
end

export residualize!

export EndogProbit2StepWorkspace
