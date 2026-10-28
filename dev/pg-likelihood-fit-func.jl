function fit_mle!(
    opt :: NLopt.Opt, ϑhat :: Vector{Float64}, ϑinit :: Vector{Float64},
    ϑwork :: Vector{Float64}, mlelb :: Vector{Float64}, mleub :: Vector{Float64},
    ρgrid :: Vector{Float64}, ws :: EndogProbitMLEWorkspace, wgts = nothing;
    tolf :: Float64 = 1e-12, tolx :: Float64 = 1e-12, tolg :: Float64 = 1e-12,
    maxiter :: Int = 1000, maxgrads :: Int = 100
)
    dϑ = length(ϑinit)
    NLopt.min_objective!(opt, (ϑ, G) -> ll_nlopt_fg!(ϑ, G, ws, wgts))
    NLopt.ftol_abs!(opt, tolf)
    NLopt.xtol_abs!(opt, tolx)
    NLopt.nlopt_set_param(opt, "tolg", tolg)
    NLopt.maxeval!(opt, maxiter)
    NLopt.vector_storage!(opt, maxgrads)

    bestll = Inf # Tracker for best likelihood candidate
    beststatus = :NOT_RUN
    copyto!(ϑhat, ϑinit)

    @inbounds for j in 1:length(ρgrid)-1
        # Extract endpoints and set bounds.
        mlelb[dϑ] = ρgrid[j]
        mleub[dϑ] = ρgrid[j+1]
        NLopt.lower_bounds!(opt, mlelb)
        NLopt.upper_bounds!(opt, mleub)

        # Reset starting values; place ρ at the box midpoint.
        copyto!(ϑwork, ϑinit)
        ϑwork[dϑ] = (mlelb[dϑ] + mleub[dϑ]) / 2

        # optimize! overwrites ϑwork.
        llval, _, status = NLopt.optimize!(opt, ϑwork)

        if isfinite(llval) && llval < bestll
            bestll = llval
            beststatus = status
            copyto!(ϑhat, ϑwork)
        end
    end
    return bestll, beststatus
end
