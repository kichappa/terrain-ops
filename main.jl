using CUDA
using Random
using PlotlyJS, Plots
include("terrain.jl")
include("players.jl")
include("tick.kernels.jl")

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

# GPU array of all constants
sim_constants = CuArray([L, seed, altPs, bush_density, max_height, GT_spies, UGA_camps, GT_interact_range, UGA_interact_range, MAX_ERROR, escape_time, capture_prob_no_bush, capture_prob_bush])

# Generate the topography
topo, bushes = topography_gpu(zeros(Float64, L, L), generate_points(sim_constants), 3)

Plots.plot(1:L, 1:L, topo, st = :surface, ratio = 1, zlim = [0, L], xlim = [0, L], ylim = [0, L], xlabel = "X", ylabel = "Y", zlabel = "Z", bgcolor = "black")

# initialize the state for GT and UGA
UGA = create_UGA(UGA_camps, topo, L, seed)
GT = create_GT(GT_spies, topo, L, seed)


function tick_host()
	# Create a 2D array of zeros to represent the GT-GT adjacency matrix 
	GT_adj = CUDA.zeros(Int, GT_spies, GT_spies) # <--------------------------------------------------------------------Why do we need multiple adjacency matrices?
	# Create a 2D array of zeros to represent the UGA-UGA adjacency matrix 
	UGA_adj = CUDA.zeros(Int, UGA_camps, UGA_camps) # <-----------------------------------------------------------------Why do we need multiple adjacency matrices?
	# Create a 2D array of zeros to represent the GT-UGA adjacency matrix 
	GT_UGA_adj = CUDA.zeros(Int, GT_spies + UGA_camps, GT_spies + UGA_camps) # <----------------------------------------Why do we need multiple adjacency matrices?
	# Create a 2D array of zeros to represent the global information list from GT spies on UGA camps
	GT_hive_info = CUDA.zeros(Int8, UGA_camps, 4)
	# Create a 2D array of zeros to represent the global information list from UGA camps on GT spies
	UGA_hive_info = CUDA.zeros(Int8, GT_spies, 4)
	# --------------------------------------------------------------------------------------------------------
	# In the adjacency matrices, first GT_spies rows and columns are for GT spies. The rest are for UGA camps.
	# --------------------------------------------------------------------------------------------------------

	# Create a 2D array of (t, s, se, f, fe) x 500*3 x GT_spies that stores the information acquired by GT spies on UGA camps in each time step
	GT_knowledge = CUDA.zeros(Int, 5, 500 * 3, GT_spies)
	GT_knowledge_count = CUDA.zeros(Int, GT_spies)
	GT_knowledge_prev_count = CUDA.zeros(Int, GT_spies)
	GT_knowledge_10behind = CUDA.zeros(Int, GT_spies) # not used... do we need this?

	for time in 1:500
		new_randoms = CUDA.rand(Float32, 2, GT_spies + UGA_camps) .* 2 .- 1
		# call the device tick function
		# tick<<<1,1>>>(state, topo, GT_adj, UGA_adGT_adj, GT_UGA_adj, GT_hive_info, UGA_hive_info, GT_spies, UGA_camps, L)
		@cuda threads = t blocks = b global_coherence(topo, bushes, UGA, GT, GT_adj, UGA_adj, GT_UGA_adj, GT_hive_info, UGA_hive_info, sim_constants, time)

		# acquire new information
		@cuda threads = GT_spies + UGA_camps blocks = GT_spies shmem = sizeof(Int) gt_observe(
			GT_knowledge,
			GT_knowledge_count,
			GT_knowledge_prev_count,
			topo,
			UGA,
			GT,
			GT_adj,
			UGA_adj,
			GT_UGA_adj,
			GT_hive_info,
			UGA_hive_info,
			sim_constants,
			new_randoms,
			time,
		)
		@cuda threads = GT_spies + UGA_camps blocks = UGA_camps uga_observe(topo, UGA, GT, GT_adj, UGA_adj, GT_UGA_adj, GT_hive_info, UGA_hive_info, sim_constants, new_randoms, time)

		# move the players
		@cuda threads = 1 blocks = GT_spies gt_move(topo, bushes, UGA, GT, GT_adj, UGA_adj, GT_UGA_adj, GT_hive_info, UGA_hive_info, sim_constants, time)
		@cuda threads = 1 blocks = UGA_camps uga_move(topo, bushes, UGA, GT, GT_adj, UGA_adj, GT_UGA_adj, GT_hive_info, UGA_hive_info, sim_constants, time)
	end
end
