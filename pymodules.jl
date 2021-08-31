module PyModules
using PyCall
function __init__()
    py"""
    import sys
    """

    py"""sys.path.append"""(joinpath(@__DIR__, "nerf"))

    global load_llff = pyimport("load_llff")
    global run_nerf_helpers = pyimport("run_nerf_helpers")
    global run_nerf = pyimport("run_nerf")
end
end
