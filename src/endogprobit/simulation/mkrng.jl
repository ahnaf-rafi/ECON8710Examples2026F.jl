# Placeholder struct to help mk_rng distinguish use case.
struct EndogProbitSimRng end
export EndogProbitSimRng

"""
    mk_rng(::EndogProbitSimRng, mseed::Int, n::Int, r::Int, b::Int) -> Xoshiro

Construct a reproducible RNG from master seed `mseed`, sample size `n`,
replication index `r`, and bootstrap index `b`. Reserve `b = 0` for sample
generation and `b ≥ 1` for bootstrap draws. Integer inputs must be nonnegative.
"""
function mk_rng(::EndogProbitSimRng, mseed::Int, n::Int, r::Int, b::Int) :: Xoshiro
    seed = UInt64(mseed)
    seed = mix64(seed ⊻ mix64(UInt64(n)))
    seed = mix64(seed ⊻ mix64(UInt64(r)))
    seed = mix64(seed ⊻ mix64(UInt64(b)))
    return Xoshiro(seed)
end

export mk_rng
