module ECON8710Examples2026F

using LinearAlgebra
using Random
using Dates

using StatsFuns: normcdf, normpdf, normlogcdf

# Useful constants.
const LOG2PI = log(2π)
const SQRT2 = sqrt(2.0)
const SQRT2PI_INV = sqrt(2 / π)

#--------------------#
# Utilities: general #
#--------------------#
# Logging.
include("utilities/general/logging.jl")
# Naming output files.
include("utilities/general/outfilenamer.jl")

# Simulation utility: different seeds for different runs.
include("utilities/rng/mix64.jl")

# Different weighting schemes for multiplier bootstraps.
export wgts!
include("utilities/weights/base-obswgt.jl")
include("utilities/weights/multinom.jl")
include("utilities/weights/poisson.jl")
include("utilities/weights/unifmix.jl")
include("utilities/weights/unifsym.jl")
include("utilities/weights/exp.jl")
include("utilities/weights/gauss.jl")

#-----------------------------------------------#
# Probit model with single endogenous regressor #
#-----------------------------------------------#
# Structs for endogenous probit.
include("endogprobit/types.jl")

# MLE
include("endogprobit/mle/ll-f.jl")
include("endogprobit/mle/ll-g.jl")
include("endogprobit/mle/ll-h.jl")
include("endogprobit/mle/ll-g-outprod.jl")
include("endogprobit/mle/ll-wrap-nlopt.jl")
include("endogprobit/mle/ll-wrap-optim.jl")
# include("endogprobit/mle/ll-wrap-jso.jl")

# Simulation setup.
# RNG constructor for split simulations.
include("endogprobit/simulation/mkrng.jl")
# Data generator.
include("endogprobit/simulation/gen.jl")
# Simulation runner(s).
include("endogprobit/simulation/runner.jl")

end
