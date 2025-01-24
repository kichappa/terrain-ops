using CUDA
using Random
using PlotlyJS, Plots
include("headers.jl")
include("terrain.jl")
include("players.jl")
include("tick.jl")

# Constants for setting up the simulation
L = 200 # 1
seed = 758 # 2
altPs = 7 # 3
bush_density = 9 # 4
max_height = 10 # 5
GT_spies = 50 # 6
UGA_camps = 5 # 7
GT_interact_range = 10 # 8
UGA_interact_range = 20 # 9
MAX_ERROR = 0.3 # 10, this is the maximum error in the GT's information acquest on UGA camps
escape_time = 20 # 11, this is the time it takes for a GT spy to escape after being captured
capture_prob_no_bush = 0.5 # 12, this is the minimum probability of capturing a GT spy when it is not in a bush
capture_prob_bush = 0.1 # 13, this is the maximum probability of capturing a GT spy when it is in a bush
sim_time = 1 # 14, this is the number of time steps the simulation will run for

# struct of all constants
sim_constants = simulation_constants(L, seed, altPs, bush_density, max_height, GT_spies, UGA_camps, GT_interact_range, UGA_interact_range, MAX_ERROR, escape_time, capture_prob_no_bush, capture_prob_bush, sim_time)


# Generate the topography
topo, bushes = topography_gpu(zeros(Float64, L, L), generate_points(sim_constants), 3, sim_constants.bush_density, 10)

# Plots.plot(1:L, 1:L, topo, st = :surface, ratio = 1, zlim = [0, L], xlim = [0, L], ylim = [0, L], xlabel = "X", ylabel = "Y", zlabel = "Z", bgcolor = "black")

# initialize the state for GT and UGA
UGA = create_UGA(UGA_camps, topo, L, seed)
GT = create_GT(GT_spies, topo, bushes, L, seed)

println("Spies:")
for spy in eachrow(GT)
	println(spy)
end

println("Camps:")
for camp in eachrow(UGA)
	println(camp)
end

tick_host(GT, UGA, CuArray(topo), CuArray(bushes), sim_constants)