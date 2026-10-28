function vcov_gmm!(
    θhat :: Vector{Float64},
    ws   :: EndogProbit2StepWorkspace,
    wgts = nothing
)
    n = size(ws.ξhat, 2)
    p = length(ws.δhat)
    q = length(θhat)
    d = p + 1 + q
    itau = p + 1

    c = θhat[1]
    half_LOG2PI = LOG2PI / 2

    # Estimate first-stage residual variance.
    sumw = 0.0
    τhat = 0.0
    @inbounds for i in 1:n
        wi = obswgt(wgts, i)
        iszero(wi) && continue

        νi = ws.ξhat[1, i]
        sumw += wi
        τhat += wi * νi^2
    end
    τhat /= sumw

    fill!(ws.SS, 0.0)
    fill!(ws.HH, 0.0)
    fill!(ws.CS, 0.0)
    fill!(ws.v, 0.0)
    fill!(ws.Ω, 0.0)

    @inbounds for i in 1:n
        wi = obswgt(wgts, i)
        iszero(wi) && continue

        νi = ws.ξhat[1, i]
        yi = ws.ξhat[q+1, i]

        indexi = 0.0
        for j in 1:q
            indexi += θhat[j] * ws.ξhat[j, i]
        end
        zi = yi * indexi

        λi = exp(-(zi^2 / 2) - half_LOG2PI - normlogcdf(zi))
        ri = yi * λi
        hi = λi * (zi + λi)

        # First-stage moments and Jacobian blocks.
        for j in 1:p
            Sji = ws.ξols[j, i]
            ws.m[j] = Sji * νi
            ws.v[j] -= 2 * wi * νi * Sji

            for k in 1:j
                ws.SS[k, j] += wi * ws.ξols[k, i] * Sji
            end
        end

        # Residual-variance moment.
        ws.m[itau] = νi^2 - τhat

        # Probit moments and Jacobian blocks.
        for j in 1:q
            Wji = ws.ξhat[j, i]
            ws.m[itau+j] = ri * Wji

            for k in 1:j
                ws.HH[k, j] += wi * hi * ws.ξhat[k, i] * Wji
            end

            crossji = c * hi * Wji
            if j == 1
                crossji -= ri
            end
            for k in 1:p
                ws.CS[j, k] += wi * crossji * ws.ξols[k, i]
            end
        end

        # Upper triangle of the linearly weighted sandwich meat.
        for j in 1:d
            for k in 1:j
                ws.Ω[k, j] += wi * ws.m[k] * ws.m[j]
            end
        end
    end

    # Fill lower triangle of the meat.
    @inbounds for j in 1:d
        for k in 1:j-1
            ws.Ω[j, k] = ws.Ω[k, j]
        end
    end

    # Invert the block lower-triangular summed Jacobian:
    #
    # J = [-SS       0        0;
    #       v'   -sumw        0;
    #       CS       0      -HH].
    #
    # Solve J * Jinv = I one block of rows at a time.
    fill!(ws.Jinv, 0.0)

    # First-stage rows.
    @inbounds for j in 1:p
        ws.Jinv[j, j] = -1.0
    end

    FS = cholesky!(Symmetric(ws.SS, :U))
    ldiv!(FS, @view(ws.Jinv[1:p, :]))

    # Residual-variance row.
    @inbounds for j in 1:d
        val = 0.0
        for k in 1:p
            val += ws.v[k] * ws.Jinv[k, j]
        end
        if j == itau
            val -= 1.0
        end
        ws.Jinv[itau, j] = val / sumw
    end

    # Probit rows.
    @inbounds for j in 1:d
        for k in 1:q
            val = 0.0
            for h in 1:p
                val += ws.CS[k, h] * ws.Jinv[h, j]
            end
            if j == itau + k
                val -= 1.0
            end
            ws.Jinv[itau+k, j] = val
        end
    end

    FH = cholesky!(Symmetric(ws.HH, :U))
    ldiv!(FH, @view(ws.Jinv[itau+1:d, :]))

    # Form the sandwich using the summed Jacobian and meat.
    mul!(ws.tmp, ws.Jinv, ws.Ω)
    mul!(ws.V, ws.tmp, transpose(ws.Jinv))

    # Normalize weights implicitly, retaining n as the sample size.
    ws.V .*= sumw / n

    # Remove numerical asymmetry.
    @inbounds for j in 1:d
        for k in 1:j-1
            val = (ws.V[k, j] + ws.V[j, k]) / 2
            ws.V[k, j] = val
            ws.V[j, k] = val
        end
    end

    return τhat
end

export vcov_gmm!
