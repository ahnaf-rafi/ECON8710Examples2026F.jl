"""
    SimJLD2Namer

Functor for constructing JLD2 output filenames for simulation results.

Filenames are of the form `<prefix>r<rrrr>.jld2`, where the replication
number is zero-padded to `r_pad` digits.

# Fields
- `prefix::String` : fixed filename prefix encoding all invariant simulation
                     parameters (e.g. DGP index, sample size)
- `r_pad::Int`     : zero-padding width for the replication number

# Example
```julia
namer = SimJLD2Namer("dgp1_n0800_", 4)
namer(1)    # "dgp1_n0800_r0001.jld2"
namer(2000) # "dgp1_n0800_r2000.jld2"
```
"""
struct SimJLD2Namer
    prefix :: String
    r_pad  :: Int
end

"""
    (namer::SimJLD2Namer)(r)

Construct a JLD2 output filename for replication `r`.
"""
function (namer::SimJLD2Namer)(r::Int)
    return namer.prefix * "r$(lpad(r, namer.r_pad, '0')).jld2"
end

export SimJLD2Namer
