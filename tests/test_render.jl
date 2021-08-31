using Test
using PyCall

include(joinpath(@__DIR__, "../render.jl"))
include(joinpath(@__DIR__, "../helpers.jl"))
include(joinpath(@__DIR__, "../pymodules.jl"))

np = pyimport("numpy")


@testset "get_rays" begin
    c2w = Float32[
        0.99359834 0.003297717 0.112923324
        -0.0012722292 0.9998371 -0.018004214
        -0.112964295 0.01774529 0.99344057
        -0.25719586 0.03902717 0.034052096
    ]
    H, W, focal = 504, 378, 407.5658f0
    c2w_reversed = reverse_dims(c2w)
    expected_rays_o, expected_rays_d =
        PyModules.run_nerf_helpers.get_rays_np(H, W, focal, c2w_reversed)
    expected_rays_o, expected_rays_d =
        reverse_dims(np.array(expected_rays_o)), reverse_dims(expected_rays_d)
    got_rays_o, got_rays_d = get_rays(H, W, focal, c2w)
    @test got_rays_o ≈ expected_rays_o atol = 0.5
    @test got_rays_d ≈ expected_rays_d atol = 0.5
end;
