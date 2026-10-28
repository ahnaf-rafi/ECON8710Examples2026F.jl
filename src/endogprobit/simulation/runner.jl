abstract type MLEBackend end
struct MLEOptimBackend <: MLEBackend end
struct MLENLoptBackend <: MLEBackend end

function run_sims(
              :: MLENLoptBackend,
    n         :: Int,
    dgp       :: EndogProbitDGP,
    r_start   :: Int,
    r_end     :: Int,
    ϑinit     :: Vector{Float64};
    ρlo       :: Float64 = -1.0+1e-12,
    ρhi       :: Float64 = 1.0-1e-12,
    B         :: Int     = 1000,
    mseed     :: Int     = 42,
    tolf      :: Float64 = 1e-12,
    tolx      :: Float64 = 1e-12,
    tolg      :: Float64 = 1e-12,
    maxiter   :: Int     = 1000,
    maxgrads  :: Int     = 100,
    verbose   :: Bool    = false,
    overwrite :: Bool    = false,
    endline1  :: String  = "\n",
    endline2  :: String  = "\r",
    logdir    :: String  = "logs",
    outdir    :: String  = "results-raw",
    n_pad     :: Int     = 3,
    r_pad     :: Int     = 4
)
    # Dimensions.
    k, l = covardim(dgp)
    dϑ = (2k) + l + 3
    dθ = k + 2

    # MLE parameter bounds.
    mlelb = fill(-Inf, dϑ)
    mleub = fill(Inf, dϑ)
    mlelb[dϑ-1] = eps(Float64)
    mlelb[dϑ] = ρlo
    mleub[dϑ] = ρhi

    # Create storage.
    wsmle = EndogProbitMLEWorkspace(n, k, l)
    wsgmm = EndogProbit2StepWorkspace(n, k, l)

    ϑmle   = Vector{Float64}(undef, dϑ)
    θgmm   = Vector{Float64}(undef, dθ)
    ϑmle_b = Vector{Float64}(undef, dϑ)
    θgmm_b = Vector{Float64}(undef, dθ)

    αs_mle_bm    = Vector{Float64}(undef, B)
    αs_se_mle_bm = Vector{Float64}(undef, B)
    αs_gmm_bm    = Vector{Float64}(undef, B)
    αs_se_gmm_bm = Vector{Float64}(undef, B)
    αs_mle_be    = Vector{Float64}(undef, B)
    αs_se_mle_be = Vector{Float64}(undef, B)
    αs_gmm_be    = Vector{Float64}(undef, B)
    αs_se_gmm_be = Vector{Float64}(undef, B)

    wgts = Vector{Float64}(undef, n)

    # Construct optimizers.
    optmle = NLopt.Opt(:LD_LBFGS, dϑ)
    optprb = NLopt.Opt(:LD_LBFGS, dθ)

    NLopt.lower_bounds!(optmle, mlelb)
    NLopt.upper_bounds!(optmle, mleub)

    for opt in (optmle, optprb)
        NLopt.ftol_abs!(opt, tolf)
        NLopt.xtol_abs!(opt, tolx)
        NLopt.nlopt_set_param(opt, "tolg", tolg)
        NLopt.maxeval!(opt, maxiter)
        NLopt.vector_storage!(opt, maxgrads)
    end

    # Make logging and output directories if needed.
    ispath(logdir) || mkpath(logdir)
    ispath(outdir) || mkpath(outdir)

    # Create logger path and output namer.
    pad_n = lpad(n, n_pad, '0')
    pad_R_start = lpad(r_start, r_pad, '0')
    log_file = joinpath(logdir, "n$(pad_n)_r$(pad_R_start).txt")
    namer = SimJLD2Namer(joinpath(outdir, "n$(pad_n)_"), r_pad)

    # Start logging.
    io = open(log_file, "w")
    log_progress!(
        io,
        "Endog. Probit Gaussian Reg.s, n=$n, r=$(r_start)--$r_end";
        verbose=verbose, endline=endline1
    )

    for r in r_start:r_end
        outpath = namer(r)
        # Skip iteration if overwrite=false and output file exists.
        if !overwrite && isfile(outpath)
            # In principle, one should double check here if the saved output is
            # actually valid:
            # - Is it from the same DGP (dists+params)?
            # - Is it running with the same optimization params?
            # - This can be done by saving some of this info in JLD2 files as an
            #   architectural choice, and then checking if the saved file from
            #   matches on those features.
            # I am however, a lazy person and not so worried about that.
            # Not too lazy to let myself know what happened though!
            log_progress!(io, "  r=$r skipped (exists)."; verbose=verbose)

            # This is how you skip an iteration in a loop in Julia.
            continue
        end

        # Reset RNG and starting values for this replication.
        rng = mkrng(EndogProbitSimRNG(), mseed, n, r)
        s = Random.Sampler(rng, 1:n)
        copyto!(ϑmle, ϑinit)

        # Crank out estimation!
        run_sim_single!(
            rng, ϑmle, θgmm, ϑmle_b, θgmm_b,
            αs_mle_bm, αs_se_mle_bm, αs_gmm_bm, αs_se_gmm_bm,
            αs_mle_be, αs_se_mle_be, αs_gmm_be, αs_se_gmm_be,
            optmle, optprb, wsmle, wsgmm, dgp, wgts, s, r, B, outpath
        )

        # Log that current iter finished.
        log_progress!(
            io, "  r=$r completed."; verbose=verbose, endline=endline2
        )
    end

    log_progress!(io, "  Done."; verbose=verbose, endline=endline1)
    close(io)
    return nothing
end

export run_sims

function run_sim_single!(
    rng          :: AbstractRNG,
    ϑmle         :: Vector{Float64},
    θgmm         :: Vector{Float64},
    ϑmle_b       :: Vector{Float64},
    θgmm_b       :: Vector{Float64},
    αs_mle_bm    :: Vector{Float64},
    αs_se_mle_bm :: Vector{Float64},
    αs_gmm_bm    :: Vector{Float64},
    αs_se_gmm_bm :: Vector{Float64},
    αs_mle_be    :: Vector{Float64},
    αs_se_mle_be :: Vector{Float64},
    αs_gmm_be    :: Vector{Float64},
    αs_se_gmm_be :: Vector{Float64},
    optmle       :: NLopt.Opt,
    optprb       :: NLopt.Opt,
    wsmle        :: EndogProbitMLEWorkspace,
    wsgmm        :: EndogProbit2StepWorkspace,
    dgp          :: EndogProbitDGP,
    wgts         :: Vector{Float64},
    s            :: Random.Sampler,
    r            :: Int,
    B            :: Int,
    outpath      :: String
)
    # Generate data.
    gen!(rng, wsmle, dgp)

    # Fit MLE.
    min_objective!(optmle, (ϑ, G) -> ll_nlopt_fg!(ϑ, G, wsmle, nothing))
    llval, _, status = optimize!(optmle, ϑmle)

    # Extract structural alpha and its standard error.
    vcov_mle!(ϑmle, wsmle, nothing)
    ialpha = (2wsmle.k) + wsmle.l + 1
    αmle = ϑmle[ialpha]
    seαmle = sqrt(wsmle.V[ialpha, ialpha])

    # Fit GMM.
    convert!(wsgmm, wsmle)
    fill!(θgmm, 0.0)
    llprbval, llprbstatus = fit_gmm!(optprb, θgmm, wsgmm, nothing)

    # Extract structural alpha and its standard error.
    τgmm = vcov_gmm!(θgmm, wsgmm, nothing)
    αgmm, seαgmm = extract_alpha_gmm!(θgmm, τgmm, wsgmm)

    # Bootstrap MLE objective uses the current contents of wgts.
    min_objective!(
        optmle, (ϑ, G) -> ll_nlopt_fg!(ϑ, G, wsmle, wgts)
    )

    # Bootstrap loop.
    for b in 1:B
        # Multinomial bootstrap.
        wgts!(rng, wgts, s, MultinomialWeights())

        copyto!(ϑmle_b, ϑmle)
        llval_bm, _, status_bm = optimize!(optmle, ϑmle_b)
        vcov_mle!(ϑmle_b, wsmle, wgts)
        αs_mle_bm[b] = ϑmle_b[ialpha]
        αs_se_mle_bm[b] = sqrt(wsmle.V[ialpha, ialpha])

        copyto!(θgmm_b, θgmm)
        llprbval_bm, llprbstatus_bm = fit_gmm!(
            optprb, θgmm_b, wsgmm, wgts
        )
        τgmm_bm = vcov_gmm!(θgmm_b, wsgmm, wgts)
        αgmm_bm, seαgmm_bm = extract_alpha_gmm!(θgmm_b, τgmm_bm, wsgmm)
        αs_gmm_bm[b] = αgmm_bm
        αs_se_gmm_bm[b] = seαgmm_bm

        # Exponential bootstrap.
        wgts!(rng, wgts, ExponentialWeights())

        copyto!(ϑmle_b, ϑmle)
        llval_be, _, status_be = optimize!(optmle, ϑmle_b)
        vcov_mle!(ϑmle_b, wsmle, wgts)
        αs_mle_be[b] = ϑmle_b[ialpha]
        αs_se_mle_be[b] = sqrt(wsmle.V[ialpha, ialpha])

        copyto!(θgmm_b, θgmm)
        llprbval_be, llprbstatus_be = fit_gmm!(
            optprb, θgmm_b, wsgmm, wgts
        )
        τgmm_be = vcov_gmm!(θgmm_b, wsgmm, wgts)
        αgmm_be, seαgmm_be = extract_alpha_gmm!(θgmm_b, τgmm_be, wsgmm)
        αs_gmm_be[b] = αgmm_be
        αs_se_gmm_be[b] = seαgmm_be
    end

    # Store.
    jldsave(
        outpath;
        αmle = αmle,
        seαmle = seαmle,
        αgmm = αgmm,
        seαgmm = seαgmm,
        αs_mle_bm = αs_mle_bm,
        αs_se_mle_bm = αs_se_mle_bm,
        αs_gmm_bm = αs_gmm_bm,
        αs_se_gmm_bm = αs_se_gmm_bm,
        αs_mle_be = αs_mle_be,
        αs_se_mle_be = αs_se_mle_be,
        αs_gmm_be = αs_gmm_be,
        αs_se_gmm_be = αs_se_gmm_be
    )

    return nothing
end

export run_sim_single!

export MLENLoptBackend
