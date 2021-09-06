# %%
# constants config
llff_hold = 8 # size of holdout/test set

# %%
using DifferentialEquations, DiffEqFlux
using Pipe: @pipe
using Flux
using Flux.Data: DataLoader
using DiffEqFlux, DifferentialEquations

include("helpers.jl")
include("render.jl")
include("pymodules.jl")

# %%
images, poses, bounds, render_poses, i_test = PyModules.load_llff.load_llff_data(
    "nerf/data/nerf_llff_data/fern/",
    8;
    recenter = true,
    bd_factor = 0.75,
    spherify = false,
);
images, poses, bounds, render_poses = reverse_dims(images),
reverse_dims(poses),
reverse_dims(bounds),
reverse_dims(render_poses);

hwf = poses[end, :, 1]; # height, width, focal length
poses = poses[1:4, :, :]; # camera-to-world
i_test = [convert(Integer, i_test)];
i_test = collect(1:(size(images)|>last))[1:llff_hold];
i_train = [i for i in 1:(size(images)|>last) if !(i in i_test)];

# not really necessary since we still get a float vector out...
near, far = 1.0, 1.0;
H, W, f = hwf;
H, W = convert(Integer, H), convert(Integer, W);
hwf = [H, W, f];

train_poses = poses[:, :, i_train];
test_poses = poses[:, :, i_test];

train_images = images[:, :, :, i_train];
test_images = images[:, :, :, i_test];

# %%
get_features(c2w) = get_features(H, W, f, c2w, 0.0f0, 1.0f0)
# train_rays = get_features(H, W, f, train_poses[:, :, 1], 0, 1);
train_ray_features =
    @pipe train_poses |> mapslices(get_features, _, dims = [1, 2]) |> reshape(_, 11, :);
train_rgb = reshape(train_images, 3, :);

# %%
train_dataloader = DataLoader(
    (features = train_ray_features, rgb = train_rgb);
    batchsize = 32,
    shuffle = true,
);
# train_dataloader |> first

# %%
# define nn
function get_model()

end

# define a function that takes nn and outputs rgb
# call train! with Flux.params(nn)

# ray_results = [get_rays(H, W, f, train_poses[:, :, i]) for i in 1:last(size(train_poses))];
# rays_o = reshape(cat([rr[1] for rr in ray_results]...; dims=[2, 3]), 3, :);
# rays_d = reshape(cat([rr[2] for rr in ray_results]...; dims=[2,3]), 3, :);


# render_path -> render -> batchify_rays -> render_rays
# render_path takes a sequence of camera poses
# render is used for training
# it takes either c2w poses to generate rays, or already generated rays
# batchify rays renders rays in samller mini-batches to avoid OOM
# render_rays does the actual hard lifting of rendering
# it only takes ray minibatches
