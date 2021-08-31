using Test

include(joinpath(@__DIR__, "../render.jl"))
include(joinpath(@__DIR__, "../helpers.jl"))
include(joinpath(@__DIR__, "../pymodules.jl"))


@testset "get_rays" begin
    c2w = rand(4, 3)
    H, W, focal = 256.0, 512.0, 5.0
    c2w_reversed = reverse_dims(c2w)
    expected =
        PyModules.run_nerf_helpers.get_rays_np(H, W, focal, c2w_reversed) |> reverse_dims
    got = get_rays(H, W, focal, c2w)
    @test got ≈ expected
end;
