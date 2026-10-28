"""
    extract_alpha_gmm!(θhat, τhat, ws)

Extract structural alpha and its standard error using ws.V from vcov_gmm!.

Fills ws.gα and ws.Vgα. Returns (αhat, seα).
"""
function extract_alpha_gmm!(
    θhat :: Vector{Float64},
    τhat :: Float64,
    ws   :: EndogProbit2StepWorkspace
)
    p = length(ws.δhat)
    itau = p + 1

    c = θhat[1]
    a = θhat[2]

    scale2 = 1 + c^2 * τhat
    invscale = inv(sqrt(scale2))
    invscale3 = invscale / scale2

    αhat = a * invscale

    fill!(ws.gα, 0.0)
    ws.gα[itau] = -0.5 * a * c^2 * invscale3
    ws.gα[itau+1] = -a * c * τhat * invscale3
    ws.gα[itau+2] = invscale

    mul!(ws.Vgα, ws.V, ws.gα)
    varα = dot(ws.gα, ws.Vgα)

    return αhat, sqrt(varα)
end

export extract_alpha_gmm!
