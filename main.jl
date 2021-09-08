# %%
# constants config
llff_hold = 8 # size of holdout/test set

# %%
using Printf
using DifferentialEquations, DiffEqFlux
using Pipe: @pipe
using Flux
using Flux.Data: DataLoader
using DiffEqFlux, DifferentialEquations

include("helpers.jl")
include("render.jl")
include("pymodules.jl")
include("model.jl")

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
near, far = 0.0f0, 1.0f0;
H, W, f = hwf;
H, W = convert(Integer, H), convert(Integer, W);
hwf = [H, W, f];

train_poses = poses[:, :, i_train];
test_poses = poses[:, :, i_test];

train_images = images[:, :, :, i_train];
test_images = images[:, :, :, i_test];

# %%
get_features(c2w) = get_features(H, W, f, c2w, near, far)
train_ray_features =
    @pipe train_poses |> mapslices(get_features, _, dims = [1, 2]) |> reshape(_, 11, :);
train_rgb = reshape(train_images, 3, :);
train_dataloader =
    DataLoader((train_ray_features, train_rgb); batchsize = 32, shuffle = true);

# %%
# define nn

nn = NeRFNODE(10, 4);
nn_ode = NeuralODE(
    nn,
    (0.0f0, 1.0f0),
    Tsit5(),
    save_everystep = false,
    reltol = 1e-3,
    abstol = 1e-3,
    save_start = false,
);

model = Chain(
    raw_to_state_space,
    nn_ode,
    x -> x.u[end], # get final timestep
    x -> x[end-2:end, :], # get rgb only
);

loss(x, y) = sum((model(x) .- y) .^ 2);
opt = Flux.Optimiser(ExpDecay(1, 0.1, 250 * 1000), ADAM(5e-4, (0.9, 0.999)));
params = Flux.params(model);

# %%
iter = 0

cb() = begin
    global iter += 1
    if iter % 10 == 1
        @printf("Iter: %3d", iter)
    end
    if iter == 20
        Flux.stop()
    end
end
Flux.train!(loss, params, train_dataloader, opt; cb = cb)

# define a function that takes nn and outputs rgb
# call train! with Flux.params(nn)

# render_path -> render -> batchify_rays -> render_rays
# render_path takes a sequence of camera poses
# render is used for training
# it takes either c2w poses to generate rays, or already generated rays
# batchify rays renders rays in samller mini-batches to avoid OOM
# render_rays does the actual hard lifting of rendering
# it only takes ray minibatches
