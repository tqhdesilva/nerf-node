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
        reverse_dims(np.array(expected_rays_o)), reverse_dims(np.array(expected_rays_d))
    got_rays_o, got_rays_d = get_rays(H, W, focal, c2w)
    @test got_rays_o ≈ expected_rays_o rtol = 0.01
    @test got_rays_d ≈ expected_rays_d rtol = 0.01
end;

@testset "ndc_rays" begin
    rays_o = Float32[
        -0.25719586 -0.25719586 -0.25719586 -0.25719586 -0.25719586
        0.03902717 0.03902717 0.03902717 0.03902717 0.03902717
        0.034052096 0.034052096 0.034052096 0.034052096 0.034052096
    ]
    rays_d = Float32[
        -0.49953154 -0.4970937 -0.49465576 -0.49221793 -0.48978004
        0.4414239 0.441432 0.44144008 0.44144818 0.44145626
        -1.0712894 -1.0710124 -1.0707353 -1.0704583 -1.0701811
    ]
    H, W, focal = 504, 378, 407.5658f0
    expected_ndc_o, expected_ndc_d = PyModules.run_nerf_helpers.ndc_rays(
        H,
        W,
        focal,
        1.0f0,
        reverse_dims(rays_o),
        reverse_dims(rays_d),
    )
    expected_ndc_o, expected_ndc_d =
        reverse_dims(np.array(expected_ndc_o)), reverse_dims(np.array(expected_ndc_d))

    got_ndc_o, got_ndc_d = ndc_rays(H, W, focal, 1.0f0, rays_o, rays_d)
    @test isa(got_ndc_o, AbstractArray{Float32})
    @test isa(got_ndc_d, AbstractArray{Float32})
    @test got_ndc_o ≈ expected_ndc_o rtol = 0.01
    @test got_ndc_d ≈ expected_ndc_d rtol = 0.01
end
