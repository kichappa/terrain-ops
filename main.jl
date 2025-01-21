using CUDA
using Random
using PlotlyJS, Plots
include("terrain.jl")

# Constants for setting up the simulation
L = 200
seed = 758
altPs = 7
max_height = 10
GT_spies = 50
UGA_camps = 5 

# Generate the topography
topo = zeros(Float64, L, L);
topo = topography_gpu(topo, generate_points(seed, L, altPs, max_height), 2.0, 32)

Plots.plot(1:L, 1:L,topo, st=:surface, ratio=1, zlim=[0,L], xlim=[0,L], ylim=[0,L],xlabel="X", ylabel="Y", zlabel="Z", bgcolor="black")

# initialize the state for GT and UGA
# ...


function tick_host()
    # topography is ready, UGA camps are ready, GT spies are ready
    # Create a 2D array of zeros to represent the GT-GT adjacency matrix
    GT = CUDA.zeros(Int, GT_spies, GT_spies)
    # Create a 2D array of zeros to represent the UGA-UGA adjacency matrix
    UGA = CUDA.zeros(Int, UGA_camps, UGA_camps)
    # Create a 2D array of zeros to represent the GT-UGA adjacency matrix
    GT_UGA = CUDA.zeros(Int, GT_spies, UGA_camps)
    # Create a 2D array of zeros to represent the global information list from GT spies on UGA camps
    GT_info = CUDA.zeros(Int8, UGA_camps, 4)
    # Create a 2D array of zeros to represent the global information list from UGA camps on GT spies
    UGA_info = CUDA.zeros(Int8, GT_spies, 4)

    for time in 1:500
        # call the device tick function
        # tick<<<1,1>>>(state, topo, GT, UGA, GT_UGA, GT_info, UGA_info, GT_spies, UGA_camps, L)
        @cuda threads=t blocks=b global(topo, GT, UGA, GT_UGA, GT_info, UGA_info, GT_spies, UGA_camps, L)
        @cuda threads=t blocks=b gt_do(topo, GT, UGA, GT_UGA, GT_info, UGA_info, GT_spies, UGA_camps, L)
        @cuda threads=t blocks=b uga_do(topo, GT, UGA, GT_UGA, GT_info, UGA_info, GT_spies, UGA_camps, L)
    end
end