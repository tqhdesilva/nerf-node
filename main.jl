# %%
# config
llff_hold = 8

# %%
using PyCall
using DifferentialEquations, DiffEqFlux

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
	recenter=true,
	bd_factor=0.75,
	spherify=false
);

hwf = poses[1, 1:3, end]; # height, width, focal length
poses = poses[:, 1:3, 1:4]; # camera-to-world
i_test = [convert(Integer, i_test)]
i_test = collect(1:(size(images) |> first))[1:llff_hold];
i_train = [i for i in 1:(size(images) |> first) if !(i in i_test)];

# not really necessary since we still get a float vector out...
near, far = 1., 1.
H, W, f = hwf
H, W = convert(Integer, H), convert(Integer, W)
hwf = [H, W, f];

render_poses = poses[i_test, :, :];

# %%
# render_path -> render -> batchify_rays -> render_rays
# render_path takes a sequence of camera poses
# render is used for training
# it takes either c2w poses to generate rays, or already generated rays
# batchify rays renders rays in samller mini-batches to avoid OOM
# render_rays does the actual hard lifting of rendering
# it only takes ray minibatches
