include("tick.kernels.jl")
using CSV, DataFrames

function tick_host(GT, UGA, topo, bushes, slopes_x, slopes_y, sim_constants)
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

	GT_Q_values = CuArray([spy_range_info() for _ in 1:2, _ in 1:GT_spies])

	# learnt parameters
	learnt_params = q_values(5, 5, 1, 2, 1)

	# write GT_UGA_adj to tmp/GT_UGA_adj.CSV, save it as "GT_UGA_adj[].interact, GT_UGA_adj[].distance"
	adj_info = map(x -> string(" ", x.visible, ";", x.interact, ";", x.distance), collect(GT_UGA_adj))
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


	@cuda threads = (sim_constants.GT_interact_range * 2, sim_constants.UGA_interact_range * 2) blocks = sim_constants.GT_spies shmem = sizeof(Float32) * 3 hide_in_bush(bushes, GT, sim_constants)
	CUDA.synchronize()
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
		GT_UGA_adj_host = collect(GT_UGA_adj)
		# spy_uga_adj = [GT_UGA_adj_host[i, j].interact for i in 1:GT_spies, j in GT_spies+1:GT_spies+UGA_camps]
		for j in GT_spies+1:GT_spies+UGA_camps
			for i in 1:GT_spies
				print("$(GT_UGA_adj_host[j, i].interact)")#,$(GT_UGA_adj_host[i, j].distance)")
			end
			println()
		end
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
			GT,
			GT_spies,
			GT_UGA_adj,
			sim_constants,
			time,
		)
		@cuda threads = GT_spies blocks = UGA_camps uga_observe(GT, GT_UGA_adj, UGA_hive_info, time)

		@cuda threads = size(GT_hive_info, 1) blocks = GT_spies gt_coordinate(GT_knowledge, GT_knowledge_count, GT_knowledge_prev_count, GT_hive_info, sim_constants)

		# move the players
		threads = (sim_constants.GT_interact_range, sim_constants.GT_interact_range)
		blocks = GT_spies
		shmem = sizeof(Float32)*3 + sizeof(spy_range_info) * (1 + sim_constants.GT_interact_range * sim_constants.GT_interact_range)
		@cuda threads = threads blocks = blocks shmem = shmem gt_move(
			GT_Q_values,
			learnt_params,
			GT_knowledge,
			GT_knowledge_count,
			GT_knowledge_prev_count,
			topo,
			bushes,
			GT,
			sim_constants,
			time,
		)
		CUDA.synchronize()
		# if time % 10 == 0
		# 	@cuda threads = 1 blocks = UGA_camps uga_move(topo, UGA, GT, GT_UGA_adj, GT_hive_info, UGA_hive_info, sim_constants, time)
		# end
		# provide 
		p = PlotlyJS.plot(
			PlotlyJS.surface(
				x = 1:sim_constants.L,
				y = 1:sim_constants.L,
				z = transpose(collect(topo) .+ collect(bushes)),
				colorscale = colorscale(sim_constants),
				surfacecolor = transpose(color_map(topo, bushes, GT, GT_spies, UGA, UGA_camps, sim_constants, 20)),
				ratio = 1,
				zlim = [0, 10],
				xlim = [0, sim_constants.L],
				ylim = [0, sim_constants.L],
				xlabel = "X",
				ylabel = "Y",
				zlabel = "Z",
				showscale = false,
			),
			layout(sim_constants),
		)
		PlotlyJS.savefig(p, "img/$(time).png")
	end
	threads = (32, 9)
	blocks = (cld(GT_spies + UGA_camps, threads[1]), cld(GT_spies + UGA_camps, threads[2]))
	@cuda threads = threads blocks = blocks global_coherence(topo, bushes, UGA, GT, sim_constants.GT_spies, sim_constants.UGA_camps, GT_UGA_adj, GT_hive_info, UGA_hive_info, sim_constants, sim_constants.sim_time)

	adj_info = map(x -> string(" ", x.visible, ";", x.interact, ";", x.distance), collect(GT_UGA_adj))
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
