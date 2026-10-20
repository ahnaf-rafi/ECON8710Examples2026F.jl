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
include("utilities/general/logging.jl")
include("utilities/general/outfilenamer.jl")

include("utilities/rng/mix64.jl")

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
include("endogprobit/types.jl")
include("endogprobit/simulation/mkrng.jl")
include("endogprobit/simulation/gen.jl")

end
