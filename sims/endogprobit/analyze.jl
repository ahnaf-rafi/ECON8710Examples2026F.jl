# Run from the package root with:
# julia --project=. sims/endogprobit/analyze.jl [run directory]
# With no argument, use the latest date-time-stamped parallel run.

using JLD2
using Statistics
using Distributions
using Printf

#-------------------#
# Analysis settings #
#-------------------#

ns = [1000, 2000, 4000]
r_start = 1
r_end = 500
B = 500
α0 = 1.0
level = 0.95

#--------------------#
# Analyze one sample #
#--------------------#

function analyze_alpha(n, indir, io, r_start, r_end, B, α0, level)
    R = r_end - r_start + 1
    αs = fill(NaN, R, 2)
    counts = zeros(Int, 2, 5)
    covers = zeros(Int, 2, 5)
    missing = 0
    p = (1.0 - level) / 2
    z = quantile(Normal(), 1.0 - p)
    ds = zeros(B)
    ts = zeros(B)

    for r in r_start:r_end
        path = joinpath(indir, "n$(lpad(n, 3, '0'))_r$(lpad(r, 4, '0')).jld2")
        if !isfile(path)
            missing += 1
            println(io, "Missing: n=$n, r=$r")
            continue
        end
        data = load(path)

        for (j, est) in enumerate(("mle", "gmm"))
            α = data["α$est"]
            seα = data["seα$est"]
            if isfinite(α)
                αs[r-r_start+1,j] = α
            else
                println(io, "Invalid estimate: n=$n, r=$r, estimator=$est")
            end

            # Gaussian interval.
            if isfinite(α) && isfinite(seα) && seα > 0.0
                counts[j,1] += 1
                covers[j,1] += α - z * seα <= α0 <= α + z * seα
            else
                println(io, "Invalid Gaussian interval: n=$n, r=$r, estimator=$est")
            end

            for (h, kind) in enumerate(("bm", "be"))
                αs_b = data["αs_$(est)_$kind"]
                ses_b = data["αs_se_$(est)_$kind"]
                αs_b isa AbstractVector || error("Expected bootstrap vector: $path")
                ses_b isa AbstractVector || error("Expected bootstrap SE vector: $path")
                length(αs_b) == B || error("Unexpected bootstrap length: $path")
                length(ses_b) == B || error("Unexpected bootstrap SE length: $path")
                valid_b = isfinite(α)
                valid_t = isfinite(seα) && seα > 0.0
                for b in 1:B
                    ds[b] = αs_b[b] - α
                    ts[b] = ds[b] / ses_b[b]
                    valid_b &= isfinite(ds[b])
                    valid_t &= isfinite(ts[b]) && isfinite(ses_b[b]) && ses_b[b] > 0.0
                end

                # Basic interval: reverse quantiles of the centered bootstrap.
                if valid_b
                    lo = α - quantile(ds, 1.0 - p)
                    hi = α - quantile(ds, p)
                    counts[j,h+1] += 1
                    covers[j,h+1] += lo <= α0 <= hi
                else
                    println(io, "Invalid basic interval: n=$n, r=$r, estimator=$est, weights=$kind")
                end

                # Bootstrap-t interval: use bootstrap SEs in the pivot,
                # and the original SE when constructing the interval.
                if valid_b && valid_t
                    lo = α - quantile(ts, 1.0 - p) * seα
                    hi = α - quantile(ts, p) * seα
                    counts[j,h+3] += 1
                    covers[j,h+3] += lo <= α0 <= hi
                else
                    println(io, "Invalid studentized interval: n=$n, r=$r, estimator=$est, weights=$kind")
                end
            end
        end
    end

    # Use denominator N for variance, so MSE = variance + bias^2.
    moments = fill(NaN, 2, 3)
    usable = zeros(Int, 2)
    for j in 1:2
        total = 0.0
        for r in 1:R
            if isfinite(αs[r,j])
                usable[j] += 1
                total += αs[r,j]
            end
        end
        N = usable[j]
        if N > 0
            meanα = total / N
            bias = meanα - α0
            variance = 0.0
            mse = 0.0
            for r in 1:R
                if isfinite(αs[r,j])
                    variance += (αs[r,j] - meanα)^2
                    mse += (αs[r,j] - α0)^2
                end
            end
            moments[j,1] = bias
            moments[j,2] = variance / N
            moments[j,3] = mse / N
        end
    end
    println(io, "n=$n: expected=$R, saved=$(R-missing), missing=$missing")
    return moments, usable, covers, counts, missing
end

function write_analysis(runroot, ns, r_start, r_end, B, α0, level)
    1 <= r_start <= r_end || error("Require 1 <= r_start <= r_end.")
    B > 0 || error("B must be positive.")
    0.0 < level < 1.0 || error("Require 0 < level < 1.")
    indir = joinpath(runroot, "results-raw")
    isdir(indir) || error("Missing results-raw directory: $runroot")
    outdir = joinpath(runroot, "analysis")
    mkpath(outdir)
    R = r_end - r_start + 1
    results = open(joinpath(outdir, "alpha-checks.txt"), "w") do io
        println(io, "Missing files and invalid estimates/intervals are listed below.")
        results = []
        for n in ns
            push!(results, analyze_alpha(n, indir, io, r_start, r_end, B, α0, level))
        end
        results
    end

    open(joinpath(outdir, "alpha-moments.txt"), "w") do io
        println(io, "Alpha: bias, empirical variance, and mean squared error")
        println(io, "Truth=$α0; expected R=$R per n; bootstrap B=$B.")
        println(io, "Variance uses denominator N. MSE = variance + bias^2.")
        println(io, "Excluded = expected R - usable N, including missing files.")
        println(io, "See alpha-checks.txt for individual missing/invalid results.\n")
        rule = repeat('-', 87)
        println(io, rule)
        @printf(io, "%6s  %-9s  %5s  %8s  %8s  %12s  %12s  %12s\n",
                "n", "Estimator", "N", "Excluded", "Missing", "Bias", "Variance", "MSE")
        println(io, rule)
        for (i, n) in enumerate(ns)
            moments, usable, covers, counts, missing = results[i]
            for (j, est) in enumerate(("MLE", "Two-step"))
                @printf(io, "%6d  %-9s  %5d  %8d  %8d  %12.6g  %12.6g  %12.6g\n",
                        n, est, usable[j], R-usable[j], missing,
                        moments[j,1], moments[j,2], moments[j,3])
            end
        end
        println(io, rule)
    end

    open(joinpath(outdir, "alpha-coverage.txt"), "w") do io
        println(io, "Alpha: equal-tailed $(100level)% confidence interval coverage")
        println(io, "Each cell is percent coverage (usable N). Excluded count = $R - N.")
        println(io, "Coverage is conditional on an available, numerically valid interval.")
        println(io, "All $B bootstrap draws must be valid; no draws are dropped.")
        println(io, "BM = multinomial weights; BE = exponential weights.")
        println(io, "Basic uses centered bootstrap differences (not percentile intervals).")
        println(io, "Studentized uses (alpha_b-alpha)/SE_b and the original SE.")
        println(io, "Quantiles use Julia's default linear interpolation.")
        println(io, "Saved files do not contain optimizer statuses or gradient norms.")
        println(io, "These numerical checks do not establish convergence.\n")
        rule = repeat('-', 108)
        println(io, rule)
        @printf(io, "%6s  %-9s", "n", "Estimator")
        for label in ("Gaussian", "Basic BM", "Basic BE", "Student BM", "Student BE")
            @printf(io, "  %16s", label)
        end
        println(io)
        println(io, rule)
        for (i, n) in enumerate(ns)
            moments, usable, covers, counts, missing = results[i]
            for (j, est) in enumerate(("MLE", "Two-step"))
                @printf(io, "%6d  %-9s", n, est)
                for h in 1:5
                    N = counts[j,h]
                    value = N > 0 ? 100covers[j,h] / N : NaN
                    cell = @sprintf("%.2f (%d)", value, N)
                    @printf(io, "  %16s", cell)
                end
                println(io)
            end
        end
        println(io, rule)
        println(io, "See alpha-checks.txt for individual missing/invalid results.")
    end
    println("Alpha analysis: $outdir")
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) <= 1 || error("Supply at most one run directory.")
    if isempty(ARGS)
        rundir = joinpath(@__DIR__, "parallel-runs")
        runroot = ""
        for name in readdir(rundir)
            path = joinpath(rundir, name)
            if occursin(r"^\d{8}-\d{6}$", name) && isdir(path)
                global runroot = path
            end
        end
        isempty(runroot) && error("No date-time-stamped parallel runs found.")
    else
        runroot = abspath(ARGS[1])
    end
    write_analysis(runroot, ns, r_start, r_end, B, α0, level)
end
