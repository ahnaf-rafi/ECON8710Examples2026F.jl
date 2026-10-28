using Distributed
using Dates

#---------------------#
# Simulation settings #
#---------------------#

ns = [1000, 2000, 4000]
r_start = 1
r_end = 500
B = 500
mseed = 42

batch_size = 10
reserve_cores = 1
core_budget = nothing  # Override with usable physical cores if needed.

outroot = joinpath(
    @__DIR__, "parallel-runs", Dates.format(now(), "yyyymmdd-HHMMSS")
)

settings = (
    B=B, mseed=mseed,
    k=2, l=2, α0=1.0, τ0=4.0, ρ0=0.5, ρ0S=0.1,
    ρlo=-1.0+1e-12, ρhi=1.0-1e-12,
    tolf=1e-12, tolx=1e-12, tolg=1e-12,
    maxiter=1000, maxgrads=100, outroot=outroot
)

#----------------#
# CPU allocation #
#----------------#

function available_cores()
    if Sys.islinux()
        # Count physical cores in this process's allowed CPU set.
        status = read("/proc/self/status", String)
        match_cpu = match(r"(?m)^Cpus_allowed_list:\s*(.+)$", status)
        isnothing(match_cpu) && error("Set core_budget explicitly.")

        cores = Set{Tuple{Int,Int}}()
        for part in split(strip(match_cpu.captures[1]), ',')
            ends = parse.(Int, split(part, '-'))
            for cpu in first(ends):last(ends)
                path = "/sys/devices/system/cpu/cpu$cpu/topology"
                socket = parse(
                    Int, strip(read(joinpath(path, "physical_package_id"), String))
                )
                core = parse(
                    Int, strip(read(joinpath(path, "core_id"), String))
                )
                push!(cores, (socket, core))
            end
        end
        ncores = length(cores)

        # Respect a visible cgroup-v2 CPU quota.
        quota_file = "/sys/fs/cgroup/cpu.max"
        if isfile(quota_file)
            quota = split(read(quota_file, String))
            if quota[1] != "max"
                ncores = min(
                    ncores, fld(parse(Int, quota[1]), parse(Int, quota[2]))
                )
            end
        end
    elseif Sys.isapple()
        ncores = parse(Int, strip(read(`sysctl -n hw.physicalcpu`, String)))
    elseif Sys.iswindows()
        command = "(Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum"
        ncores = parse(
            Int, strip(read(`powershell -NoProfile -Command $command`, String))
        )
    else
        error("Set core_budget to the usable physical-core allocation.")
    end

    if haskey(ENV, "SLURM_CPUS_PER_TASK")
        ncores = min(ncores, parse(Int, ENV["SLURM_CPUS_PER_TASK"]))
    end
    return ncores
end

all(>(0), ns) || error("Sample sizes must be positive.")
length(unique(ns)) == length(ns) || error("Sample sizes must be distinct.")
1 <= r_start <= r_end || error("Require 1 <= r_start <= r_end.")
B > 0 && batch_size > 0 || error("B and batch_size must be positive.")
reserve_cores >= 1 || error("Reserve at least one core.")

ncores = isnothing(core_budget) ? available_cores() : core_budget
nworkers_use = ncores - reserve_cores
nworkers_use >= 1 || error("No worker cores remain after reserving cores.")

jobs = Tuple{Int,Int,Int}[]
for n in ns
    for r in r_start:batch_size:r_end
        push!(jobs, (n, r, min(r + batch_size - 1, r_end)))
    end
end
nworkers_use = min(nworkers_use, length(jobs))

mkpath(dirname(outroot))
mkdir(outroot)  # Refuse to reuse an existing run directory.

open(joinpath(outroot, "settings.txt"), "w") do io
    println(io, settings)
    println(io, "ns=$ns r_start=$r_start r_end=$r_end batch_size=$batch_size")
    println(io, "core_budget=$ncores reserve_cores=$reserve_cores workers=$nworkers_use")
end

#---------------#
# Start workers #
#---------------#

projectdir = dirname(Base.active_project())
worker_ids = addprocs(
    nworkers_use;
    exeflags=`--project=$projectdir --threads=1 --gcthreads=1`,
    enable_threaded_blas=false
)

try
    @everywhere worker_ids begin
        using ECON8710Examples2026F
        using LinearAlgebra

        BLAS.set_num_threads(1)

        function run_batch(job, settings)
            n, r_start, r_end = job

            # DGP setup.
            k = settings.k
            l = settings.l
            dS = k+l
            dξ = dS + 2

            α0 = settings.α0
            β0 = ones(k) / k
            γ0 = ones(k) / dS
            π0 = ones(l) / dS
            τ0 = settings.τ0
            σ0 = sqrt(τ0)
            ρ0 = settings.ρ0
            ρ0S = settings.ρ0S

            ΣU0 = zeros(dξ, dξ)
            for j in 1:dS
                for i in 1:dS
                    ΣU0[i,j] = ifelse(j != i, ρ0S, 1.0)
                end
            end
            ΣU0[dS + 1, dS + 1] = τ0
            ΣU0[dS + 1, dξ] = ρ0 * σ0
            ΣU0[dξ, dS + 1] = ρ0 * σ0
            ΣU0[dξ, dξ] = 1.0
            CU0 = cholesky(ΣU0).L

            dgp = EndogProbitGaussianDGP(α0, β0, γ0, π0, CU0)
            ϑinit = [β0; γ0; π0; α0; τ0; ρ0]

            logdir = joinpath(
                settings.outroot, "logs", "n$(n)_r$(r_start)"
            )
            outdir = joinpath(settings.outroot, "results-raw")
            mkpath(logdir)

            # Run this batch and record its outcome.
            t_start = time_ns()
            status = "OK"
            try
                run_sims(
                    MLENLoptBackend(), n, dgp, r_start, r_end, ϑinit;
                    ρlo=settings.ρlo, ρhi=settings.ρhi,
                    B=settings.B, mseed=settings.mseed,
                    tolf=settings.tolf, tolx=settings.tolx, tolg=settings.tolg,
                    maxiter=settings.maxiter, maxgrads=settings.maxgrads,
                    verbose=false, overwrite=false,
                    logdir=logdir, outdir=outdir
                )
            catch err
                err isa InterruptException && rethrow()
                status = "FAILED"
                open(joinpath(logdir, "error.txt"), "w") do io
                    showerror(io, err, catch_backtrace())
                end
            end

            elapsed = (time_ns() - t_start) / 1e9
            println(
                "$status: n=$n, r=$r_start:$r_end, " *
                "$(round(elapsed; digits=3)) s."
            )
            return (n, r_start, r_end, myid(), status, elapsed)
        end
    end

    # One batch at a time per worker. Faster workers take more batches.
    results = pmap(WorkerPool(worker_ids), jobs; batch_size=1) do job
        run_batch(job, settings)
    end

    open(joinpath(outroot, "batches.tsv"), "w") do io
        println(io, "n\tr_start\tr_end\tworker\tstatus\tseconds")
        for result in results
            println(io, join(result, '\t'))
        end
    end

    println("Outputs: $outroot")
    any(result -> result[5] != "OK", results) &&
        error("Some batches failed; see batch logs.")
finally
    rmprocs(worker_ids)
end
