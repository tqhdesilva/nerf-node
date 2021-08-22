using NPZ

# https://github.com/Fyusion/LLFF#using-your-own-poses-without-running-colmap
# http://www.cse.psu.edu/~rtc12/CSE486/lecture12.pdf
# Julia arrays are column major(np and torch are row major)
# colexicographic vs lexicographic
# might be worth just using pycall?
# how to transfer in-memory np to jl?

"Load the data from npy files, with a scaling factor."
function load_data(basedir::String, width::UInt, height::UInt; factor::UInt=1)
	poses_bounds_arr = npzread(joinpath(basedir, "poses_bounds.npy"))
	poses = poses_bounds_arr[:, 1:12] |>
		(x) -> reshape(x, size(poses_bounds_arr, 1), 3, 4) |>
		(x) -> permutedims(x, (2, 3, 1))
	hwf = poses_bounds_arr[:, 13:15] |>
		(x) -> permutedims(x, (2, 1))
	bounds = poses_bounds_arr[:, 16:17] |>
		(x) -> permutedims(x, (2, 1))
	correct_coords!(poses)
end

"Swaps two columns of a matrix in-place."
function swapcols!(X::AbstractMatrix, i::Integer, j::Integer)
    @inbounds for k = 1:size(X, 1)
        X[k,i], X[k,j] = X[k,j], X[k,i]
    end
end

"Correct rotation coordinates in-place from [-y, x, z] to [x, y, z]."
function correct_coords!(poses::AbstractArray{T,3}) where T <: Any
	for i in 1:size(poses, 3)
		rotation = @view poses[:, :, i]
		rotation[:, 1] = - rotation[:, 1]
		swapcols!(rotation, 1, 2)
	end
end
