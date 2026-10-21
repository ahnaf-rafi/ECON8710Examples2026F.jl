function run_sims(
    n::Int
)
    ispath(logdir) || mkpath(logdir)  # Make logging directory if needed.
    ispath(outdir) || mkpath(outdir)  # Make results directory if needed.

    a
end

function run_sim_single(
    rng::AbstractRNG, dgp::EndogProbitDGP, data::EndogProbitData, io::IO;
    overwrite::Bool=true,
    verbose::Bool=false,
    endline="\n"
)
    # TODO: Check if being asked to overwrite. If not, exit early.
    # !overwrite && <check-output-existence> && return nothing

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
