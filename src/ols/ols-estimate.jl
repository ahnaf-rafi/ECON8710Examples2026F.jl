"""
    ols!(βhat, SS, S, D, wgts)

Regress D on observation-column regressors S with observation weights.
Overwrites βhat and the square workspace SS.
"""
function ols!(βhat::Vector{Float64}, SS::Matrix{Float64},
              S::AbstractMatrix{Float64}, D::AbstractVector{Float64}, wgts)
    fill!(SS, 0.0)
    fill!(βhat, 0.0)
    p = size(S, 1)

    @inbounds for i in axes(S, 2)
        w_i = obswgt(wgts,i)
        iszero(w_i) && continue

        for j in 1:p
            wS_ji = w_i * S[j, i]
            βhat[j] += wS_ji * D[i]

            for k in 1:j
                SS[k, j] += S[k, i] * wS_ji
            end
        end
    end

    F = cholesky!(Symmetric(SS, :U))
    ldiv!(F, βhat)
    return βhat
end

"""
    ols!(βhat, SS, A, wgts)

Fit from A = [S; D']; its last row is the outcome.
"""
function ols!(βhat::Vector{Float64}, SS::Matrix{Float64},
              A::AbstractMatrix{Float64}, wgts=nothing)
    S = @view A[1:end-1, :]
    D = @view A[end, :]
    return ols!(βhat, SS, S, D, wgts)
end

export ols!
