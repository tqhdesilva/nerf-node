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


function ndc_rays(H, W, focal, near, rays_o::AbstractArray, rays_d::AbstractMatrix)
    # Shift rays to near plane
    t = -(near .+ rays_o[3, :]) ./ rays_d[3, :] # 1 x N
    rays_o = rays_o .+ t .* rays_d # 3 x N

    # Project onto NDC coordinates
    o_x, o_y, o_z = selectdim(rays_o, 1, 1), selectdim(rays_o, 1, 2), selectdim(rays_o, 1, 3)
    d_x, d_y, d_z = selectdim(rays_d, 1, 1), selectdim(rays_d, 1, 2), selectdim(rays_d, 1, 3)

    o1 = - focal ./ (W / 2) .* o_x ./ o_z,
    o2 = - focal ./ (H / 2) .* o_y ./ o_z,
    o3 = 1 .+ 2 .* near ./ o_z

    d1 = - focal ./ (W / 2) .* (d_x ./ d_z .- o_x ./ o_z)
    d2 = - focal ./ (H / 2) .* (d_y ./ d_z .- o_y ./ o_z)
    d3 = -2 .* n ./ o_z

    o_prime = [o1 o2 o3]
    d_prime = [d1 d2 d3]

    return o_prime, d_prime
end

struct RenderResult
    rgb::AbstractArray{T} where T <: Real
    disparity::AbstractArray{U} where U <: Real
    function RenderResult(rgb, disparity)
        if size(rgb, 1) != 3
            error("rgb first dim must be 3.")
        end
        if size(rgb)[2:end] != size(disparity)
            error("rgb and disparity must match on the last dimensions.")
        end
        return new(rgb, disparity)
    end
end

function combine_renderresults(render_results::Vector{RenderResult})
    d = size(render_results[1].rgb) |> length
    return RenderResult(
        cat((rr.rgb for rr in render_results)...; dims=d),
        cat((rr.disparity for rr in render_results)...; dims=d - 1)
    )
end

function render_rays(rays::AbstractMatrix{T}) where T <: Real # rays is 8(or 11 with viewdir) x n
    # rays_o, rays_d, view_dir, near, far
end

function batch_render_rays(rays::AbstractMatrix{T}, chunksize; kwargs...)
    render_results = Vector{RenderResult}()
    n = last(size(rays))
    for i in 1:chunksize:n
        rays_batch = rays[:, i:min(i + chunksize, n)]
        render_result = render_rays(rays_batch, kwargs...)
        push!(render_results, render_result)
    end
    # combine render_results vector into single render result

end

function render(H, W, focal, c2w::AbstractMatrix{T,2}; c2w_staticcam::Union{AbstractMatrix{T,2},Nothing}=nothing, near=0, far=1, ndc=true, chunksize=1024 * 32) where T <: Real
    rays_o, rays_d = get_rays(H, W, focal, c2w)
    viewdirs = rays_d
    if ! isnothing(c2w_staticcam)
        rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
    end
    if ndc
        rays_o, rays_d = ndc_rays(H, W, focal, near, rays_o, rays_d)
    end
    rays = cat(reshape(rays_o, 3, :, 1), reshape(rays_d, 3, :, 1), reshape(viewdirs, 3, :, 1); dims=3)
    return 
end
