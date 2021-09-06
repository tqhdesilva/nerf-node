using Flux

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
    dims = length(size(x))
)

struct NeRFNet

end

function (nn::NeRFNet)() end

Flux.@functor NeRFNet
