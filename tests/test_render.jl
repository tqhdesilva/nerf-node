using Test
using PyCall

include(joinpath(@__DIR__, "../render.jl"))
include(joinpath(@__DIR__, "../helpers.jl"))
include(joinpath(@__DIR__, "../pymodules.jl"))

np = pyimport("numpy")


@testset "get_rays" begin
    c2w = rand(4, 3)
    H, W, focal = 256, 512, 5.0
    c2w_reversed = reverse_dims(c2w)
    expected_rays_o, expected_rays_d =
        PyModules.run_nerf_helpers.get_rays_np(H, W, focal, c2w_reversed)
    expected_rays_o, expected_rays_d =
        reverse_dims(np.array(expected_rays_o)), reverse_dims(expected_rays_d)
    got_rays_o, got_rays_d = get_rays(H, W, focal, c2w)
    @test got_rays_o ≈ expected_rays_o
    @test got_rays_d ≈ expected_rays_d
end;
