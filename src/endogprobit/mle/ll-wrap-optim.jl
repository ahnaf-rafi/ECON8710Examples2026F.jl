"""
    ll_optim_only_fg!(F, G, ϑ::Vector{Float64}, data::EndogProbitData, wgts=nothing)

Evaluate the components of [`ll`](@ref) requested by a callback by
`NLSolversBase.only_fg!`.
"""
function ll_optim_only_fg!(F, ::Nothing, ϑ::Vector{Float64}, data::EndogProbitData, wgts=nothing)
    return ll(ϑ, data, wgts)
end

function ll_optim_only_fg!(::Nothing, G, ϑ::Vector{Float64}, data::EndogProbitData, wgts=nothing)
    ll_g!(G, ϑ, data, wgts)
end

@inline function ll_optim_only_fg!(F, G, ϑ::Vector{Float64}, data::EndogProbitData, wgts=nothing)
    return ll_fg!(G, ϑ, data, wgts)
end

"""
    ll_optim_only_fgh!(F, G, H, ϑ::Vector{Float64}, data::EndogProbitData, wgts=nothing)
Evaluate the components of [`ll`](@ref) requested by a callback by
`NLSolversBase.only_fgh!`.
"""
@inline function ll_optim_only_fgh!(
    F, ::Nothing, ::Nothing, ϑ::Vector{Float64}, data::EndogProbitData, wgts=nothing
)
    return ll(ϑ, data, wgts)
end

@inline function ll_optim_only_fgh!(
    ::Nothing, G, ::Nothing, ϑ::Vector{Float64}, data::EndogProbitData, wgts=nothing
)
    ll_g!(G, ϑ, data, wgts)
end

@inline function ll_optim_only_fgh!(
    ::Nothing, ::Nothing, H, ϑ::Vector{Float64}, data::EndogProbitData, wgts=nothing
)
    ll_h!(H, ϑ, data, wgts)
end

@inline function ll_optim_only_fgh!(
    F, G, ::Nothing, ϑ::Vector{Float64}, data::EndogProbitData, wgts=nothing
)
    return ll_fg!(G, ϑ, data, wgts)
end

@inline function ll_optim_only_fgh!(
    ::Nothing, G, H, ϑ::Vector{Float64}, data::EndogProbitData, wgts=nothing
)
    ll_gh!(G, H, ϑ, data, wgts)
end

@inline function ll_optim_only_fgh!(F, G, H, ϑ::Vector{Float64}, data::EndogProbitData, wgts=nothing)
    return ll_fgh!(G, H, ϑ, data, wgts)
end
