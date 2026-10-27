function fit_gmm!(
    optprb::NLopt.Opt, θinit::Vector{Float64}, ws::TwoStepEndogProbitWorkspace, wgts
)
    ols!(ws.δhat, ws.SS, ws.ξols, wgts)
    residualize!(ws)
    min_objective!(optprb, (θ, G) -> ll_nlopt_fg!(θ, G, ws.ξhat, Probit(), wgts))
    llprbval, _, llprbstatus = optimize!(optprb, θinit)

    return llprbval, llprbstatus
end

export fit_gmm!
