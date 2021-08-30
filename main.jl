# %%
# constants config
llff_hold = 8 # size of holdout/test set

# %%
using PyCall
using DifferentialEquations, DiffEqFlux

include("helpers.jl")
include("render.jl")

# use PyCall for import load_nerf
py"""
import sys

sys.path.append("nerf")
"""
load_llff = pyimport("load_llff");

# %%
images, poses, bounds, render_poses, i_test = load_llff.load_llff_data(
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

test_poses = poses[:, :, i_test];
train_poses = poses[:, :, i_train];

# %%
rays = get_features(H, W, f, train_poses[:, :, 1], 0, 1);
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
