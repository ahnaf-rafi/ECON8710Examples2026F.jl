include("../src/ECON8710Examples2026F.jl")

using .ECON8710Examples2026F
using LinearAlgebra
using Random
using StatsFuns: normcdf, normpdf,  normlogcdf
using Optim

#-----------#
# DGP setup #
#-----------#

k = 2
l = 2
dS = k+l
dξ = dS + 2

α0 = 1.0
β0 = ones(k) / k
γ0 = ones(k) / dS
π0 = ones(l) / dS
τ0 = 4.0  # Variance of ν.
σ0 = sqrt(τ0)
ρ0 = 0.5 # Correlation
ρ0S = 0.1

# Generate a matrix of zeros to be filled with entries of covariance matrix of
# U = (X', Z', ν, ϵ)'.
ΣU0 = zeros(dξ, dξ)
for j in 1:dS
    for i in 1:dS
        ΣU0[i,j] = ifelse(j != i, ρ0S, 1.0)
    end
end
ΣU0[dS + 1, dS + 1] = τ0
ΣU0[dS + 1, dξ] = ρ0
ΣU0[dξ, dS + 1] = ρ0
ΣU0[dξ, dξ] = 1.0
CU0 = cholesky(ΣU0).L

# Various parameter vectors
ϑ0 = [β0; γ0; π0; α0; ρ0; τ0]
dϑ = length(ϑ0)
θ0 = [β0; α0]
dθ = length(θ0)
η0 = [γ0; π0; τ0]
dη = length(η0)
