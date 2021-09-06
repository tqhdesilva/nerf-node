using Test
using Pipe: @pipe

include(joinpath(@__DIR__, "../model.jl"))


@testset "positional encoder L = 10" begin
    x = rand(3, 1000)
    pe = PositionalEncoder(10)
    y = @pipe pe(x) |> reshape(_, :, size(_)[end])
    @test size(y) == (60, 1000)
end

@testset "positional encoder L = 4" begin
    x = rand(3, 1000)
    pe = PositionalEncoder(4)
    y = @pipe pe(x) |> reshape(_, :, size(_)[end])
    @test size(y) == (24, 1000)
end

@testset "arbitrary shape" begin
    x = rand(3, 16, 32, 100)
    pe = PositionalEncoder(4)
    y = pe(x)
    @test size(y) == (3, 16, 32, 8, 100)
end
