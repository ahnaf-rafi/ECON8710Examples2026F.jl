include("../src/ECON8710Examples2026F.jl")

using .ECON8710Examples2026F
using LinearAlgebra
using Random
using StatsFuns: normcdf, normpdf,  normlogcdf
using Optim
using NLopt
using BenchmarkTools

include("pg-likelihood-fit-func.jl")

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

# Generate a matrix of zeros to be filled with entries of covariance matrix of
# U = (X', Z', ν, ϵ)'.
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

# Various parameter vectors
ϑ0 = [β0; γ0; π0; α0; τ0; ρ0]  # Fundamental structural parameter.
dϑ = length(ϑ0)                # Fundamental structural parameter: length.
θ0 = [β0; α0]                  # Second stage parameter.
dθ = length(θ0)                # Second stage parameter: length.
η0 = [γ0; π0; τ0]              # First stage parameter.
dη = length(η0)                # First stage parameter: length.

#----------------------#
# Simulation workspace #
#----------------------#
# Set up dgp.
dgp = EndogProbitGaussianDGP(α0, β0, π0, γ0, CU0)

# Set a sample size to do pilot runs with.
n = 500

# Struct for data storage.
data = EndogProbitData(n, k, l)

# SET SEED! YES. EVEN FOR PILOT AND TRIAL STUFF. GET INTO THE HABIT.
rng = Xoshiro(0)

# Generate data.
gen!(rng, data, dgp)

wgts = nothing

# MLE code here.
ϑinit = copy(ϑ0)
ϑhat  = copy(ϑ0)
ϑwork = copy(ϑ0)

# Optimization params.
tolf  = 0.0     # 1e-12
tolx  = 1e-12
tolg = 1e-12
maxiter = 1000
maxgrads = 100
mlelb = fill(-Inf, dϑ)
mleub = fill(Inf, dϑ)
mlelb[dϑ] = -0.9
mleub[dϑ] = 0.9

ρlo = -1.0+1e-12
ρhi = 1.0-1e-12
nρgrid = 201

opt = NLopt.Opt(:LD_LBFGS, dϑ)

ρgrid = collect(range(ρlo, ρhi; length=nρgrid))

fit_mle!(opt, ϑhat, ϑinit, ϑwork, mlelb, mleub, ρgrid, data, wgts)

@benchmark fit_mle!(
    $opt, $ϑhat, $ϑinit, $ϑwork, $mlelb, $mleub, $ρgrid, $data, $wgts
)



evals=1


# Set optimization objective
opt = NLopt.Opt(:LD_LBFGS, dϑ)
min_objective!(opt, (ϑ, G) -> ll_nlopt_fg!(ϑ, G, data, wgts))

# Set bounds
mlelb = fill(-Inf, dϑ)
mleub = fill(Inf, dϑ)
mlelb[dϑ] = -0.9
mleub[dϑ] = 0.9

NLopt.lower_bounds!(opt, mlelb)
NLopt.upper_bounds!(opt, mleub)

NLopt.ftol_abs!(opt, tolf)
NLopt.xtol_abs!(opt, tolx)
NLopt.nlopt_set_param(opt, "tolg", tolg)
NLopt.maxeval!(opt, maxiter)
NLopt.vector_storage!(opt, maxgrads)

# optf, optx, ret = optimize!(opt, ϑinit)
@benchmark optimize!($opt, $ϑwork) setup=(
    copyto!($ϑwork, $ϑinit)
)

optimize!(opt, ϑinit)
niters = NLopt.numevals(opt)

# Profile likelihood: endpoints are undefined.
ρgrid = collect(range(-1.0+eps(Float64), 1.0-eps(Float64); length=201))
profile_ll = fill(NaN, length(ρgrid))
profile_ϑ = fill(NaN, dϑ, length(ρgrid))
profile_niters = fill(0, length(ρgrid))
profile_status = fill(:NOT_RUN, length(ρgrid))

popt = NLopt.Opt(:LD_LBFGS, dϑ)
NLopt.min_objective!(
    popt, (ϑ, G) -> ll_nlopt_fg!(ϑ, G, data, wgts)
)
NLopt.ftol_abs!(popt, tolf)
NLopt.xtol_abs!(popt, tolx)
popt.params["tolg"] = tolg
NLopt.maxeval!(popt, maxiter)
NLopt.vector_storage!(popt, maxgrads)

lb = fill(-Inf, dϑ)
ub = fill(Inf, dϑ)
lb[dϑ - 1] = eps(Float64)  # Positive variance.

for j in eachindex(ρgrid)
    ρ = ρgrid[j]
    abs(ρ) == 1 && continue

    # Equal bounds fix the final coordinate at ρ.
    lb[dϑ] = ρ
    ub[dϑ] = ρ
    NLopt.lower_bounds!(popt, lb)
    NLopt.upper_bounds!(popt, ub)

    start = copy(ϑinit)
    start[dϑ] = ρ
    f, x, status = NLopt.optimize(popt, start)

    profile_ll[j] = f
    profile_ϑ[:, j] = x
    profile_niters[j] = NLopt.numevals(popt)
    profile_status[j] = status
end

mknaninf(x::Real) = ifelse(isnan(x), Inf, x)

# Gridded box-constrained optimization.
boxed_ll = fill(NaN, length(ρgrid) - 1)
boxed_ϑ = fill(NaN, dϑ, length(boxed_ll))
boxed_niters = zeros(Int, length(boxed_ll))
boxed_status = fill(:NOT_RUN, length(boxed_ll))

bopt = NLopt.Opt(:LD_LBFGS, dϑ)
NLopt.min_objective!(
    bopt, (ϑ, G) -> ll_nlopt_fg!(ϑ, G, data, wgts)
)
NLopt.ftol_abs!(bopt, tolf)
NLopt.xtol_abs!(bopt, tolx)
bopt.params["tolg"] = tolg
NLopt.maxeval!(bopt, maxiter)
NLopt.vector_storage!(bopt, maxgrads)

blb = fill(-Inf, dϑ)
bub = fill(Inf, dϑ)
blb[dϑ - 1] = eps(Float64)

for j in eachindex(boxed_ll)
    blb[dϑ] = ρgrid[j]
    bub[dϑ] = ρgrid[j + 1]
    NLopt.lower_bounds!(bopt, blb)
    NLopt.upper_bounds!(bopt, bub)

    start = copy(ϑinit)
    start[dϑ] = (ρgrid[j] + ρgrid[j + 1]) / 2
    f, x, status = NLopt.optimize(bopt, start)

    boxed_ll[j] = f
    boxed_ϑ[:, j] = x
    boxed_niters[j] = NLopt.numevals(bopt)
    boxed_status[j] = status
end

minimum(mknaninf, boxed_ll)
boxed_niters

res1[1]
minimum(mknaninf, profile_ll)
minimum(mknaninf, boxed_ll)

niters
maximum(profile_niters)
minimum(profile_niters)
maximum(boxed_niters)
minimum(boxed_niters)

jprofile = argmin(mknaninf.(profile_ll))
jboxed = argmin(mknaninf.(boxed_ll))

ϑdirect = res1[2]
ϑprofile = profile_ϑ[:, jprofile]
ϑboxed = boxed_ϑ[:, jboxed]

# Columns: true, direct, profile, boxed.
hcat(ϑ0, ϑdirect, ϑprofile, ϑboxed)

# Corresponding objectives and return statuses.
(res1[1], profile_ll[jprofile], boxed_ll[jboxed])
(res1[3], profile_status[jprofile], boxed_status[jboxed])
