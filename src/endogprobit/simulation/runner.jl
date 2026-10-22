function run_sims(
    n         :: Int,
    dgp       :: EndogProbitDGP,
    r_start   :: Int,
    r_end     :: Int;
    mseed     :: Int    = 42,
    verbose   :: Bool   = false,
    overwrite :: Bool   = false,
    endline1  :: String = "\n",
    endline2  :: String = "\r",
    logdir    :: String = "logs",
    outdir    :: String = "results-raw",
    n_pad     :: Int    = 3,
    r_pad     :: Int    = 4
)
    ispath(logdir) || mkpath(logdir)  # Make logging directory if needed.
    ispath(outdir) || mkpath(outdir)  # Make results directory if needed.

    # Create logger and output namer.
    pad_n = lpad(n, n_pad, '0')
    pad_R_start = lpad(r_start, r_pad, '0')
    log_file = joinpath(logdir_dgp, "n$(pad_n)_r$(pad_R_start).txt")
    namer = SimJLD2Namer(joinpath(outdir, "n$(pad_n)_"), r_pad)

    # Start logging.
    io = open(log_file, "w")
    log_progress!(
        io,
        "Endog. Probit Gaussian Reg.s, n=$n, r=$(r_start)--$r_end";
        verbose=verbose, endline=endline1
    )

    # Create storage (structs of) arrays.
    k, l = covardim(dgp)
    data = EndogProbitData(n, k, l)

    for r in r_start:r_end
        # TODO: Check if being asked to overwrite. If not, exit early.
        # !overwrite && <check-output-existence> && return nothing
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

        # TODO: Crank out the estimation!
        # run_sim_single(...)

        # Log that current iter finished.
        log_progress!(
            io, "  r=$r completed."; verbose=verbose, endline=endline
        )
    end
end

export run_sims

function run_sim_single(
    rng       :: AbstractRNG,
    data      :: EndogProbitData,
    dgp       :: EndogProbitDGP,
    r         :: Int,
    B         :: Int,
    outpath   :: String,
    io        :: IO;
    overwrite :: Bool   = true,
    verbose   :: Bool   = false,
    endline   :: String ="\n"
)
    # Generate data.
    gen!(rng, data, dgp)

    # TODO: Fit MLE.
    # TODO: Fit GMM.
    # TODO: Calculate estimates of asymptotic variance matrix.
    # TODO: Conduct bootstraps.
    # TODO: Store.

    # TODO: Write to logfile
    # log_progress!(io, <msg>; verbose=verbose, endline=endline)

    return nothing
end

export run_sims_single
