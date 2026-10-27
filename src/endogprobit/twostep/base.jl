struct TwoStepEndogProbitWorkspace
    ξhat :: Matrix{Float64} # [νhat, D, X', Y]
    ξols :: Matrix{Float64} # [X', Z', D]
    δhat :: Vector{Float64}
    SS   :: Matrix{Float64}
end

function convert!(ws::TwoStepEndogProbitWorkspace, data::EndogProbitData)
    k = data.k
    l = data.l

    @inbounds for i in axes(data.ξs, 2)
        # OLS data: [X; Z; D].
        for j in 1:k+l+1
            ws.ξols[j, i] = data.ξs[j, i]
        end

        # Second-stage data: [νhat; D; X; Y].
        ws.ξhat[1, i] = 0.0  # Filled by residualize! after OLS.
        ws.ξhat[2, i] = data.ξs[k+l+1, i]
        for j in 1:k
            ws.ξhat[j+2, i] = data.ξs[j, i]
        end
        ws.ξhat[k+3, i] = data.ξs[k+l+2, i]
    end

    return nothing
end

export convert!

function residualize!(ws::TwoStepEndogProbitWorkspace)
    @inbounds for i in axes(ws.ξhat, 2)
        Si = @view ws.ξols[1:end-1, i]
        ws.ξhat[1, i] = ws.ξols[end, i] - dot(ws.δhat, Si)
    end

    return nothing
end

export residualize!
