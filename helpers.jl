function reverse_dims(A::AbstractArray)
    n = length(size(A))
    permutedims(A, n:-1:1)
end



function get_rays(H, W, focal, c2w::AbstractMatrix)
    i, j = [i for i in 1:W, j in 1:H], [j for i in 1:W, j in 1:H]
    dirs = cat(
        reshape(i .- W * 0.5 / focal, 1, size(i)...),
        reshape(- j .- H * 0.5 / focal, 1, size(j)...),
        reshape([-1 for _ in i], 1, size(i)...);
        dims=1
    ); # 3 x W x H
    # left to right broadcasting instead of right to left from numpy
    rays_d = sum(reshape(dirs, size(dirs)[1], 1, size(dirs)[2:end]...) .* c2w[1:3, :], dims=1) # 3 x W x H
    rays_o = repeat(c2w[end, :], 1, W, H)
    rays_o, rays_d
end