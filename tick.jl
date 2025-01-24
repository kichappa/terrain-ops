include("tick.kernels.jl")
using CSV, DataFrames

function tick_host(GT, UGA, topo, bushes, sim_constants)
	# 	# Create a 2D array of struct to represent the GT-GT adjacency matrix 
	# 	GT_adj = CuArray([gt_gt_struct() for i in 1:GT_spies, j in 1:GT_spies])
	# 	# Create a 2D array of struct to represent the UGA-UGA adjacency matrix 
	# 	UGA_adj = CuArray([uga_uga_struct() for i in 1:UGA_camps, j in 1:UGA_camps])
	# Create a 2D array of struct to represent the GT-UGA adjacency matrix 
	GT_UGA_adj = CuArray([gt_uga_struct() for i in 1:GT_spies+UGA_camps, j in 1:GT_spies+UGA_camps])
	# Create a 2D array of struct to represent the global information list from GT spies on UGA camps
	GT_hive_info = CUDA.zeros(Int8, UGA_camps, 4)
	# Create a 2D array of struct to represent the global information list from UGA camps on GT spies
	UGA_hive_info = CUDA.zeros(Int8, GT_spies, 4)
	# --------------------------------------------------------------------------------------------------------
	# In the adjacency matrices, first GT_spies rows and columns are for GT spies. The rest are for UGA camps.
	# --------------------------------------------------------------------------------------------------------

	# Create a 2D array of (t, s, se, f, fe) x 500*3 x GT_spies that stores the information acquired by GT spies on UGA camps in each time step
	GT_knowledge = CUDA.zeros(Int, 5, 500 * 3, GT_spies)
	GT_knowledge_count = CUDA.zeros(Int, GT_spies)
	GT_knowledge_prev_count = CUDA.zeros(Int, GT_spies)

	# write GT_UGA_adj to tmp/GT_UGA_adj.CSV, save it as "GT_UGA_adj[].visible, GT_UGA_adj[].distance"
	adj_info = map(x -> string(" ", x.visible, ";", x.distance), collect(GT_UGA_adj))
	CSV.write(
		"tmp/GT_UGA_adj.CSV",
		DataFrame(adj_info, :auto),
	)

	for time in 1:sim_constants.sim_time
		# call the device tick function
		# tick<<<1,1>>>(state, topo, GT_adj, UGA_adGT_adj, GT_UGA_adj, GT_hive_info, UGA_hive_info, GT_spies, UGA_camps, L)

		threads = (32, 32)
		blocks = (cld(GT_spies + UGA_camps, threads[1]), cld(GT_spies + UGA_camps, threads[2]))
		@cuda threads = threads blocks = blocks global_coherence(topo, bushes, UGA, GT, sim_constants.GT_spies, sim_constants.UGA_camps, GT_UGA_adj, GT_hive_info, UGA_hive_info, sim_constants, time)

		# write again
		adj_info = map(x -> string(" ", x.visible, ";", x.distance), collect(GT_UGA_adj))
		CSV.write(
			"tmp/GT_UGA_adj_post.CSV",
			DataFrame(adj_info, :auto),
		)

		# acquire new information
		# @cuda threads = GT_spies + UGA_camps blocks = GT_spies shmem = sizeof(Int) gt_observe(
		# 	GT_knowledge,
		# 	GT_knowledge_count,
		# 	GT_knowledge_prev_count,
		# 	topo,
		# 	UGA,
		# 	GT,
		# 	GT_adj,
		# 	UGA_adj,
		# 	GT_UGA_adj,
		# 	GT_hive_info,
		# 	UGA_hive_info,
		# 	sim_constants,
		# 	new_randoms,
		# 	time,
		# )
		# @cuda threads = GT_spies + UGA_camps blocks = UGA_camps uga_observe(topo, UGA, GT, GT_UGA_adj, GT_hive_info, UGA_hive_info, sim_constants, new_randoms, time)

		# # move the players
		# @cuda threads = 1 blocks = GT_spies gt_move(topo, UGA, GT, GT_UGA_adj, GT_hive_info, UGA_hive_info, sim_constants, time)
		# @cuda threads = 1 blocks = UGA_camps uga_move(topo, UGA, GT, GT_UGA_adj, GT_hive_info, UGA_hive_info, sim_constants, time)


	end
end