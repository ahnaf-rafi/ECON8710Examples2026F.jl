#----------------#
# Logging helper #
#----------------#

"""
    log_progress!(io::IO, msg::String; verbose::Bool=false, endline="\\n")

Write a timestamped message followed by a newline to `io`, then flush `io`.
If `verbose` is true, also write the message to stdout followed by `endline`.

# Arguments
- `io`      : Log file handle; mutated by each call.
- `msg`     : Message to log.
- `verbose` : If true, also print to stdout. Default false.
- `endline` : Text appended to the stdout message. Defaults to a newline;
              does not affect the newline written to `io`.
"""
function log_progress!(io::IO, msg::String; verbose::Bool=false, endline="\n")
    ts   = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    line = "[$ts] $msg"
    println(io, line)
    flush(io)
    verbose && print(line*endline)
end

export log_progress!
