"""
    EndogProbitData

Data container for a probit model with a single endogenous covariate.

Fields:
- `ξs::Matrix{Float64}`: observation columns, with `k + l + 2` rows.
- `k::Int`: number of exogenous regressors.
- `l::Int`: number of excluded instruments.

`EndogProbitData(n, k, l)` allocates uninitialized storage for `n` observations.
"""
struct EndogProbitData
    ξs :: Matrix{Float64}
    k  :: Int
    l  :: Int
end

function EndogProbitData(n::Int, k::Int, l::Int)
    return EndogProbitData(Matrix{Float64}(undef, k + l + 2, n), k, l)
end

export EndogProbitData

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
    π0  :: Vector{Float64}
    γ0  :: Vector{Float64}
    CU0 :: LowerTriangular{Float64,Matrix{Float64}}
end

export EndogProbitGaussianDGP
