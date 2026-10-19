#--------------------------------#
# External libraries and modules #
#--------------------------------#
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
dS = k+l     # Dimension of exogenous part of ξ.
dξ = dS + 2  # Dimension of ξ.

α0 = 1.0
β0 = ones(k) / k
γ0 = ones(k) / dS
π0 = ones(l) / dS
τ0 = 4.0           # Variance of ν.
σ0 = sqrt(τ0)      # Std. dev. of ν.
ρ0 = 0.5           # Correlation between ϵ and ν.
ρ0S = 0.1

# Generate a matrix of zeros to be filled with entries of covariance matrix of
# U = (X', Z', ν, ϵ)'. For this simulation, we will draw U as jointly normal.
# Then, we will compute ξ (with ordering below) from slides from it.
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
# For generating U, it will be a good idea to
CU0 = cholesky(ΣU0).L

# Various parameter vectors.
ϑ0 = [β0; γ0; π0; α0; ρ0; τ0]  # Fundamental structural parameter.
dϑ = length(ϑ0)                # Fundamental structural parameter: length.
θ0 = [β0; α0]                  # Second stage parameter.
dθ = length(θ0)                # Second stage parameter: length.
η0 = [γ0; π0; τ0]              # First stage parameter.
dη = length(η0)                # First stage parameter: length.

#----------------------#
# Simulation workspace #
#----------------------#
# Set a sample size to do pilot runs with.
n = 200

# SET SEED! YES. EVEN FOR PILOT AND TRIAL STUFF. GET INTO THE HABIT.
rng = Xoshiro(0)

# Allocate data storage. We will order things as ξ = (X', Z', D, Y)'.
ξs = Matrix{Float64}(undef, dξ, n)

# Note that U and ξ have the same dimension. By allocating ξ, we can first
# generate `U' into ξs and then mutate the last two rows column-by-column!
# To that end, we first generate independent standard normals using randn!.
# Then, we multiply by the Cholesky decomposition to get correlated components
# of U.
randn!(rng, ξs)

# To understand lmul!, run:
# help?> lmul!
# Note that typing `?` at a `julia>` prompt will turn it into the help interface
# with prompt `help?>`.
# lmul! basically does in-place left-multiplication of a matrix B by either a
# scalar a or a "special" matrix A. RTFM.
lmul!(CU0, ξs)
# Quick correctness check: compare (ξs * ξs') / n to ΣU0.

# Then, loop over columns, and generate D and Y.
for i in axes(ξs, 2)
    # Second stage linear component associated with X.
    Xβ0i = 0.0
    # First stage linear component associated with X.
    Xγ0i = 0.0

    # Start at row 1 of ξs.
    row_num = 1
    for j in 1:k
        # Extract value of x in column.
        xval = ξs[row_num,i]

        # Update linear components associated with X.
        Xβ0i += xval * β0[j]
        Xγ0i += xval * γ0[j]

        # Update row number.
        row_num += 1
    end

    # First stage linear component associated with instruments Z.
    Zπ0i = 0.0
    for j in 1:l
        Zπ0i += ξs[row_num,i] * π0[j]
        row_num += 1
    end

    # Generate D.
    D = Xγ0i + Zπ0i + ξs[row_num, i]
    # Update row for D in-place.
    ξs[row_num, i] = D

    # Update row number.
    row_num += 1

    # Generate utility --- currently, ξs[row_num, i] = ϵ.
    u = Xβ0i + (α0 * D) - ξs[row_num, i]

    # Encode Y ∈ {-1, 1} instead of 0-1 for easier handling of probit later.
    ξs[row_num, i] = ifelse(u >= 0, 1.0, -1.0)
end
