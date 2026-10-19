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
include("utilities/rng/mxi64.jl")


end
