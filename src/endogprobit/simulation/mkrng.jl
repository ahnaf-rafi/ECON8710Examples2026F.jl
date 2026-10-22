abstract type AbstractDispatchedSimRNG end

# Placeholder struct to help mkrng distinguish use case.
struct EndogProbitSimRNG <: AbstractDispatchedSimRNG end
export EndogProbitSimRNG

"""
    mkrng(::EndogProbitSimRng, mseed::Int, n::Int, r::Int, b::Int) -> Xoshiro

Construct a reproducible RNG from master seed `mseed`, sample size `n`,
replication index `r`, and bootstrap index `b`. Reserve `b = 0` for sample
generation and `b ≥ 1` for bootstrap draws. Integer inputs must be nonnegative.
"""
function mkrng(::EndogProbitSimRNG, mseed::Int, n::Int, r::Int, b::Int) :: Xoshiro
    seed = UInt64(mseed)
    seed = mix64(seed ⊻ mix64(UInt64(n)))
    seed = mix64(seed ⊻ mix64(UInt64(r)))
    seed = mix64(seed ⊻ mix64(UInt64(b)))
    return Xoshiro(seed)
end

export mkrng
