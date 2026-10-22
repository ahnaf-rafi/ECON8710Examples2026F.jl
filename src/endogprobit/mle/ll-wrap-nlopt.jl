"""
    ll_nlopt_fg!(ϑ::Vector{Float64}, G, data::EndogProbitData, wgts=nothing)

Evaluate the components of [`ll`](@ref) for an NLopt objective callback.
If `G` is nonempty, overwrite it with the gradient using [`ll_fg!`](@ref).
An empty `G` requests only the objective value.
"""
@inline function ll_nlopt_fg!(
    ϑ::Vector{Float64}, G, data::EndogProbitData, wgts=nothing
)
    if isempty(G)
        return ll(ϑ, data, wgts)
    end
    return ll_fg!(G, ϑ, data, wgts)
end

export ll_nlopt_fg!
