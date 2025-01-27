include("tick.kernels.jl")
using CSV, DataFrames

function tick_host(GT, UGA, topo, bushes, sim_constants)
	# Create a 2D array of structs to represent the GT-UGA adjacency matrix 
	GT_UGA_adj = CuArray([adjacency() for i in 1:GT_spies+UGA_camps, j in 1:GT_spies+UGA_camps])
	# Create a 2D array of struct to represent the global information list from GT spies on UGA camps
	GT_hive_info = CuArray([spy_hive_knowledge() for _ in 1:2*UGA_camps])
	# Create a 2D array of struct to represent the global information list from UGA camps on GT spies
	UGA_hive_info = CuArray([camp_hive_knowledge() for _ in 1:GT_spies])
	# --------------------------------------------------------------------------------------------------------
	# In the adjacency matrices, first GT_spies rows and columns are for GT spies. The rest are for UGA camps.
	# --------------------------------------------------------------------------------------------------------

	# Create a 2D array of (t, s, se, f, fe) x 500*3 x GT_spies that stores the information acquired by GT spies on UGA camps in each time step
	GT_knowledge = CuArray([spy_knowledge() for _ in 1:sim_constants.sim_time*10, _ in 1:GT_spies])
	GT_knowledge_count = CUDA.ones(Int, GT_spies)
	GT_knowledge_prev_count = CUDA.ones(Int, GT_spies)

	# write GT_UGA_adj to tmp/GT_UGA_adj.CSV, save it as "GT_UGA_adj[].interact, GT_UGA_adj[].distance"
	adj_info = map(x -> string(" ", x.interact, ";", x.distance), collect(GT_UGA_adj))
	CSV.write(
		"tmp/GT_UGA_adj.CSV",
		DataFrame(adj_info, :auto),
	)

	GT_info = map(x -> string(" ", x.time, ";", x.size, ";", x.firepower, ";", x.size_error, ";", x.firepower_error), collect(GT_knowledge))
	CSV.write(
		"tmp/GT_info.CSV",
		DataFrame(permutedims(vcat(collect(GT_knowledge_prev_count)', collect(GT_knowledge_count)', GT_info)), :auto),
	)

	UGA_hive_i = [[x.x, x.y, x.time, x.frozen, x.frozen_cycle][i] for x in collect(UGA_hive_info), i in 1:5]
	CSV.write(
		"tmp/UGA_hive_info.CSV",
		DataFrame(UGA_hive_i, :auto),
	)


	for time in 1:sim_constants.sim_time
		# call the device tick function
		# tick<<<1,1>>>(state, topo, GT_adj, UGA_adGT_adj, GT_UGA_adj, GT_hive_info, UGA_hive_info, GT_spies, UGA_camps, L)
		println("-------------------- Cycle: ", time, " --------------------")
		# ------------------------------------------------------------------------------------ GLOBAL COHERENCE ------------------------------------------------------------------------------------ #
		threads = (32, 9)
		blocks = (cld(GT_spies + UGA_camps, threads[1]), cld(GT_spies + UGA_camps, threads[2]))
		@cuda threads = threads blocks = blocks global_coherence(topo, bushes, UGA, GT, sim_constants.GT_spies, sim_constants.UGA_camps, GT_UGA_adj, GT_hive_info, UGA_hive_info, sim_constants, time)
		# ==================================================================================== ================ ==================================================================================== #

		CUDA.synchronize()
		@cuda threads = GT_spies blocks = GT_spies shmem = sizeof(Int) * 2 gt_exchange(
			GT_knowledge,
			GT_knowledge_count,
			GT_knowledge_prev_count,
			UGA,
			GT_spies,
			GT_UGA_adj,
			GT_hive_info,
			sim_constants,
			time,
		)
		CUDA.synchronize()
		# --------------------------------------------------------------------------------- ACQUIRE NEW INFORMATION -------------------------------------------------------------------------------- #
		@cuda threads = UGA_camps blocks = GT_spies shmem = sizeof(Int) gt_observe(
			GT_knowledge,
			GT_knowledge_count,
			GT_knowledge_prev_count,
			UGA,
			GT_spies,
			GT_UGA_adj,
			GT_hive_info,
			sim_constants,
			time,
		)
		@cuda threads = GT_spies blocks = UGA_camps uga_observe(GT, GT_UGA_adj, UGA_hive_info, time)

		@cuda threads = GT_spies blocks = 1 gt_coordinate(GT_knowledge, GT_knowledge_count, GT_knowledge_prev_count, GT_hive_info)
		# # move the players
		# @cuda threads = 1 blocks = GT_spies gt_move(topo, UGA, GT, GT_UGA_adj, GT_hive_info, UGA_hive_info, sim_constants, time)
		# @cuda threads = 1 blocks = UGA_camps uga_move(topo, UGA, GT, GT_UGA_adj, GT_hive_info, UGA_hive_info, sim_constants, time)


	end
	threads = (32, 9)
	blocks = (cld(GT_spies + UGA_camps, threads[1]), cld(GT_spies + UGA_camps, threads[2]))
	@cuda threads = threads blocks = blocks global_coherence(topo, bushes, UGA, GT, sim_constants.GT_spies, sim_constants.UGA_camps, GT_UGA_adj, GT_hive_info, UGA_hive_info, sim_constants, sim_constants.sim_time)

	adj_info = map(x -> string(" ", x.interact, ";", x.distance), collect(GT_UGA_adj))
	CSV.write(
		"tmp/GT_UGA_adj_post.CSV",
		DataFrame(adj_info, :auto),
	)

	GT_info = map(x -> string(" ", x.time, ";", x.size, ";", x.firepower, ";", x.size_error, ";", x.firepower_error), collect(GT_knowledge))
	CSV.write(
		"tmp/GT_info_post.CSV",
		DataFrame(permutedims(vcat(collect(GT_knowledge_prev_count)', collect(GT_knowledge_count)', GT_info)), :auto),
	)

	UGA_hive_i = [[x.x, x.y, x.time, x.frozen, x.frozen_cycle][i] for x in collect(UGA_hive_info), i in 1:5]
	CSV.write(
		"tmp/UGA_hive_info_post.CSV",
		DataFrame(UGA_hive_i, :auto),
	)
end
