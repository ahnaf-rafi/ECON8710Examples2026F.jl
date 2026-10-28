struct EndogProbitMLEWorkspace
    k    :: Int
    l    :: Int
    ξs   :: Matrix{Float64}
    H    :: Matrix{Float64} # Average Hessian; overwritten by Cholesky.
    Hinv :: Matrix{Float64} # Inverse of average Hessian.
    Ω    :: Matrix{Float64} # Average score outer product.
    V    :: Matrix{Float64} # Covariance matrix of ϑhat.
    tmp  :: Matrix{Float64} # Matrix multiplication workspace.
end

function EndogProbitMLEWorkspace(n::Int, k::Int, l::Int)
    d = (2k) + l + 3

    return EndogProbitMLEWorkspace(
        k,
        l,
        Matrix{Float64}(undef, k + l + 2, n),
        Matrix{Float64}(undef, d, d),
        Matrix{Float64}(undef, d, d),
        Matrix{Float64}(undef, d, d),
        Matrix{Float64}(undef, d, d),
        Matrix{Float64}(undef, d, d)
    )
end

export EndogProbitMLEWorkspace

# Abstract super-type for more DGPs for this model for future use.
abstract type EndogProbitDGP end

"""
Gaussian DGP with scalar treatment coefficient `α0` and lower Cholesky
factor `CU0` for errors ordered `(X..., Z..., V, U)`.
Pass `cholesky(Σ).L` as `CU0`.
"""
struct EndogProbitGaussianDGP <: EndogProbitDGP
    α0  :: Float64
    β0  :: Vector{Float64}
    γ0  :: Vector{Float64}
    π0  :: Vector{Float64}
    CU0 :: LowerTriangular{Float64,Matrix{Float64}}
end

export EndogProbitGaussianDGP

covardim(dgp::EndogProbitDGP) = (length(dgp.γ0), length(dgp.π0))
