using Flux
using Pipe: @pipe

struct PositionalEncoder
    L::Int
end

(p::PositionalEncoder)(x) = cat(
    [
        reshape(sin.(2 .^ i .* π .* x), size(x)[1:end-1]..., 1, size(x)[end]) for
        i = 0:(p.L-1)
    ]...,
    [
        reshape(cos.(2 .^ i .* π .* x), size(x)[1:end-1]..., 1, size(x)[end]) for
        i = 0:(p.L-1)
    ]...;
    dims = length(size(x)),
)

flat(p::PositionalEncoder, x) =
    reshape(p(x), 2 * p.L * prod(size(x)[1:end-1]), size(x)[end])

struct Split{T}
    paths::T
end

Split(paths...) = Split(paths)

Flux.@functor Split

(m::Split)(x::AbstractArray) = tuple(map(f -> f(x), m.paths))

struct NeRFNet
    trunk::Any
    head::Any
end

function NeRFNet(L_o::Int, L_d::Int)
    o_encoder = PositionalEncoder(L_o)
    d_encoder = PositionalEncoder(L_d)
    γ_o(x) = flat(o_encoder, x)
    γ_d(x) = flat(d_encoder, x)
    trunk = Chain(
        Parallel(
            vcat,
            γ_o,
            Chain(γ_o, Dense(60, 256, relu), [Dense(256, 256, relu) for i = 1:4]...),
        ),
        Dense(316, 256, relu),
        Dense(256, 256, relu),
        Dense(256, 256, relu),
        Dense(256, 256),
        Split(Dense(256, 1, sigmoid), (x) -> x),
    )
    rgb_head =
        Chain(Parallel(vcat, γ_d, x -> x), Dense(280, 128, relu), Dense(128, 3, sigmoid))
    return NeRFNet(trunk, rgb_head)
end

Flux.@functor NeRFNet


function (nn::NeRFNet)(x, d)
    σ, h = nn.trunk(x)[1]
    rgb = nn.head((d, h))
    return σ, rgb
end


struct NeRFNODE
    nn::NeRFNet
end

NeRFNODE(L_o::Int, L_d::Int) = NeRFNODE(NeRFNet(L_o, L_d))

Flux.@functor NeRFNODE

function (nnode::NeRFNODE)(x)
    p, d, viewdir, near, far, C, T = view(x, 1:3, :),
    view(x, 4:6, :),
    view(x, 7:9, :),
    view(x, 10, :) |> permutedims,
    view(x, 11, :) |> permutedims,
    view(x, 12, :) |> permutedims,
    view(x, 13:15, :)
    dpdt = (far .- near) .* d
    dddt = @pipe similar(d) |> fill!(_, 0)
    dviewdirdt = @pipe similar(viewdir) |> fill!(_, 0)
    dneardt = @pipe similar(near) |> fill!(_, 0)
    dfardt = @pipe similar(far) |> fill!(_, 0)
    σ, rgb = nnode.nn(p, viewdir)
    dCdt = exp.(-T) .* σ .* rgb
    dTdt = σ
    return cat(dpdt, dddt, dviewdirdt, dneardt, dfardt, dCdt, dTdt; dims = 1)
end

function raw_to_state_space(x)
    bs = size(x)[end]
    CTinitial = similar(x, (4, bs))
    fill!(CTinitial, 0)
    return cat(x, CTinitial; dims = 1)
end

function DiffEqArray_to_Array(x, T)
    xarr = T(x)
    return reshape(xarr, size(xarr)[1:end-1])
end
