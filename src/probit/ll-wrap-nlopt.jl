@inline function ll_nlopt_fg!(
    ϑ::Vector{Float64}, G, data::Matrix{Float64}, probtype::Probit, wgts=nothing
)
    if isempty(G)
        return ll(ϑ, data, probtype, wgts)
    end
    return ll_fg!(G, ϑ, data, probtype, wgts)
end
