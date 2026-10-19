"""
    mix64(x::UInt64) -> UInt64

SplitMix64-style bit mixer.

This is used only to turn structured simulation indices into well-scrambled
64-bit seeds. It is not the simulation RNG itself.
"""
function mix64(x::UInt64)::UInt64
    # Add a large fixed odd constant.
    # This constant is related to the golden ratio and is commonly used in
    # SplitMix-style generators to move the input away from simple low-entropy
    # values like 0, 1, 2, ...
    # Since x is UInt64, arithmetic wraps around modulo 2^64.
    x += 0x9e3779b97f4a7c15

    # First avalanche step.
    # x >> 30 shifts the bits of x right by 30 positions.
    # x ⊻ (x >> 30) mixes high bits into lower bits.
    # Multiplication by a large odd constant further scrambles the bits.
    # Again, multiplication wraps modulo 2^64.
    x = (x ⊻ (x >> 30)) * 0xbf58476d1ce4e5b9

    # Second avalanche step.
    # This repeats the same idea with a different right shift and a different
    # large odd multiplier.
    # The goal is that changing one input bit changes many output bits.
    x = (x ⊻ (x >> 27)) * 0x94d049bb133111eb

    # Final avalanche step.
    # This last xor-shift mixes the remaining high-bit information down
    # into the lower bits.
    return x ⊻ (x >> 31)
end

export mix64
