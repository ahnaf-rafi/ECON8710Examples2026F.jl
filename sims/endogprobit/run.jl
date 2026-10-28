# Run from the package root with:
# julia --project=. sims/endogprobit/run.jl [n [r_start [r_end [B]]]]
# Defaults: n=1000, r_start=1, r_end=20, B=1000.

using ECON8710Examples2026F
using LinearAlgebra

BLAS.set_num_threads(1)

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
τ0 = 4.0           # Variance of ν.
σ0 = sqrt(τ0)      # Std. dev. of ν.
ρ0 = 0.5           # Correlation between ϵ and ν.
ρ0S = 0.1

# Covariance matrix of U = (X', Z', ν, ϵ)'.
ΣU0 = zeros(dξ, dξ)
for j in 1:dS
    for i in 1:dS
        ΣU0[i,j] = ifelse(j != i, ρ0S, 1.0)
    end
end
ΣU0[dS + 1, dS + 1] = τ0
ΣU0[dS + 1, dξ] = ρ0 * σ0
ΣU0[dξ, dS + 1] = ρ0 * σ0
ΣU0[dξ, dξ] = 1.0
CU0 = cholesky(ΣU0).L

dgp = EndogProbitGaussianDGP(α0, β0, γ0, π0, CU0)
ϑinit = [β0; γ0; π0; α0; τ0; ρ0]

#---------------------#
# Simulation settings #
#---------------------#

length(ARGS) <= 4 || throw(ArgumentError("Expected at most four arguments: n r_start r_end B."))

n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1000
r_start = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
r_end = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 20
B = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1000

n > 0 || throw(ArgumentError("n must be positive."))
B > 0 || throw(ArgumentError("B must be positive."))
1 <= r_start <= r_end || throw(ArgumentError("Require 1 <= r_start <= r_end."))

mseed = 42

ρlo = -1.0+1e-12
ρhi = 1.0-1e-12
tolf = 1e-12
tolx = 1e-12
tolg = 1e-12
maxiter = 1000
maxgrads = 100

verbose = true
overwrite = false
logdir = joinpath(@__DIR__, "logs")
outdir = joinpath(@__DIR__, "results-raw")

#-----------------#
# Run simulations #
#-----------------#

run_sims(
    MLENLoptBackend(), n, dgp, r_start, r_end, ϑinit;
    ρlo=ρlo, ρhi=ρhi, B=B, mseed=mseed,
    tolf=tolf, tolx=tolx, tolg=tolg,
    maxiter=maxiter, maxgrads=maxgrads,
    verbose=verbose, overwrite=overwrite,
    logdir=logdir, outdir=outdir
)
