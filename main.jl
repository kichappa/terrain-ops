using CUDA
using Random
using PlutoPlotly, Plots, StaticArrays
include("handle_pluto.jl")
if is_running_in_pluto()
	println("\nThis script is running in a Pluto notebook! Include headers yourself by running the following code:")
	println("\t\tinclude(\"headers.jl\")\n\t\tinclude(\"plots.jl\")")
else
	include("headers.jl")
	include("plots.jl")
end
include("terrain.jl")
include("players.jl")
include("tick.jl")


# Constants for setting up the simulation
sim_time = 10 # 14, this is the number of time steps the simulation will run for
L = 200 # 1
seed = 758 # 2
altPs = 7 # 3
bush_density = 9 # 4
max_height = 10 # 5
GT_spies = 75 # 6
UGA_camps = 5 # 7
GT_interact_range = 10 # 8
UGA_interact_range = 15 # 9
height_range_advantage = 0.2 # 10, this is the advantage in height that a player a has over its enemy b. 
# range changes by e^((a.z - b.z) * height_range_advantage) for visiblity
# error changes by e^((b.z - a.z) * height_range_advantage) for information acquest
MAX_ERROR = 0.3 # 10, this is the maximum error in the GT's information acquest on UGA camps
escape_time = 20 # 11, this is the time it takes for a GT spy to escape after being captured
capture_prob_no_bush = 0.2 # 12, this is the minimum probability of capturing a GT spy when it is not in a bush
capture_prob_bush = 0.05 # 13, this is the maximum probability of capturing a GT spy when it is in a bush
visible_prob = 0.75 # 14, this is the probability of a GT spy being visible to a UGA camp when it is in a bush and not captured
gt_coord_size_threshold = 2 # size threshold to check if the same UGA camps spotted
gt_coord_firepower_threshold = 2 # firepower threshold to check if the same UGA camps spotted
GT_step_size = 2 # step size for GT spies

# struct of all constants
sim_constants = simulation_constants(
	L,
	seed,
	altPs,
	bush_density,
	max_height,
	GT_spies,
	UGA_camps,
	GT_interact_range,
	UGA_interact_range,
	height_range_advantage,
	MAX_ERROR,
	escape_time,
	capture_prob_no_bush,
	capture_prob_bush,
	visible_prob,
	sim_time,
	gt_coord_size_threshold,
	gt_coord_firepower_threshold,
	GT_step_size,
)


# Generate the topography
topo, bushes = topography(zeros(Float64, L, L), generate_points(sim_constants), 3, sim_constants.bush_density, 10)
slopes_x, slopes_y = slope(topo, 20)


# initialize the state for GT and UGA
UGA = create_UGA(UGA_camps, topo, L, seed)
GT = create_GT(GT_spies, topo, bushes, L, seed)

println("Spies:")
for spy in eachindex(GT)
	println(collect(GT)[spy])
end

println("Camps:")
for camp in eachindex(UGA)
	println(collect(UGA)[camp])
end

if is_running_in_pluto()
	println("\nThis script is running in a Pluto notebook! Run the tick_host function yourself by running the following code:\n")
	println("\t\ttick_host(GT, UGA, CuArray(topo), CuArray(bushes), slopes_x, slopes_y, sim_constants)")
else
	tick_host(GT, UGA, CuArray(topo), CuArray(bushes), slopes_x, slopes_y, sim_constants)
end
# Plots.plot(
# 	1:L, 1:L, topo, st = :surface, ratio = 1, zlim = [0, L], xlim = [0, L], ylim = [0, L], xlabel = "X", ylabel = "Y", zlabel = "Z", bgcolor = "black")

# PlutoPlotly.surface(
# 	x = 1:L,
# 	y = 1:L,
# 	z = transpose(topo .+ bushes),
# 	# colorscale = custom_colorscale2,
# 	# surfacecolor = transpose(color_map(topo, bushes, GT, GT_spies, UGA, UGA_camps, sim_constants)),
# 	ratio = 1,
# 	zlim = [0, L],
# 	xlim = [0, L],
# 	ylim = [0, L],
# 	xlabel = "X",
# 	ylabel = "Y",
# 	zlabel = "Z",
# 	showscale = false,
# )
