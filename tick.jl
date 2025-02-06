include("tick.kernels.jl")
using CSV, DataFrames, FileIO, Glob
using Base.Threads, Base.Iterators

# # Create communication channel
# const write_channel = Channel{Vector{UInt8}}(256)

# # Async writer task
# @async begin
# 	for img_data in write_channel
# 		write(ffmpeg_process, img_data)
# 	end
# 	close(ffmpeg_process)
# end


function tick_host(GT, UGA, topo, bushes, slopes_x, slopes_y, sim_constants, sim_time = nothing)
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
	GT_knowledge = CuArray([spy_knowledge() for _ in 1:sim_constants.sim_time*30, _ in 1:GT_spies])

	GT_knowledge_count = CUDA.ones(Int, GT_spies)
	GT_knowledge_prev_count = CUDA.ones(Int, GT_spies)
	GT_knowledge_before_exchange = CUDA.ones(Int, GT_spies)

	GT_Q_values = CuArray([spy_range_info() for _ in 1:2, _ in 1:GT_spies])

	# learnt parameters
	learnt_params = q_values(5, 5, 1, 2, 1)
	rewards = reinforcement_rewards(3, 2, -10, 0.1)

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

	# ffmpeg_cmd = ```
	# 	./ffmpeg/ffmpeg -y
	# 	-f image2pipe          
	# 	-vcodec png           
	# 	-framerate $fps 
	# 	-i -                  
	# 	-c:v libx264           
	# 	-pix_fmt yuv420p 
	# 	-vf "format=yuv420p"      
	# 	"img/video.mp4"
	# 	```         # Output file

	# # Open a pipe to FFmpeg
	# ffmpeg_process = open(ffmpeg_cmd, "w")

	# Hide the GT spies in the bushes
	@cuda threads = (sim_constants.GT_interact_range * 2, sim_constants.UGA_interact_range * 2) blocks = sim_constants.GT_spies shmem = sizeof(Float32) * 3 hide_in_bush(bushes, GT, sim_constants)
	CUDA.synchronize()

	sim_time = nothing
	if isnothing(sim_time)
		sim_time = sim_constants.sim_time
	end
	for time in 1:sim_time
		# call the device tick function
		# tick<<<1,1>>>(state, topo, GT_adj, UGA_adGT_adj, GT_UGA_adj, GT_hive_info, UGA_hive_info, GT_spies, UGA_camps, L)
		println("-------------------- Cycle: ", time, " --------------------")
		# ------------------------------------------------------------------------------------ GLOBAL COHERENCE ------------------------------------------------------------------------------------ #
		threads = (32, 9)
		blocks = (cld(GT_spies + UGA_camps, threads[1]), cld(GT_spies + UGA_camps, threads[2]))
		@cuda threads = threads blocks = blocks global_coherence(topo, bushes, UGA, GT, sim_constants.GT_spies, sim_constants.UGA_camps, GT_UGA_adj, GT_hive_info, UGA_hive_info, sim_constants, time)
		CUDA.synchronize()
		# ======================================================================================== ======== ======================================================================================== #
		# ---------------------------------------------------------------------------------------- UGA MOVE ---------------------------------------------------------------------------------------- #
		# if time % 10 == 0
		# 	@cuda threads = 1 blocks = UGA_camps uga_move(topo, UGA, GT, GT_UGA_adj, GT_hive_info, UGA_hive_info, sim_constants, time)
		# end 
		# ================================================================================== ==================== ================================================================================== #
		# ---------------------------------------------------------------------------------- EXCHANGE INFORMATION ---------------------------------------------------------------------------------- #
		GT_UGA_adj_host = collect(GT_UGA_adj)
		# spy_uga_adj = [GT_UGA_adj_host[i, j].interact for i in 1:GT_spies, j in GT_spies+1:GT_spies+UGA_camps]
		for j in GT_spies+1:GT_spies+UGA_camps
			for i in 1:GT_spies
				print("$(GT_UGA_adj_host[j, i].interact)")#,$(GT_UGA_adj_host[i, j].distance)")
			end
			println()
		end
		@cuda threads = GT_spies blocks = GT_spies shmem = sizeof(Int) * 5 gt_exchange(
			GT_knowledge,
			GT_knowledge_count,
			GT_knowledge_before_exchange,
			GT_UGA_adj,
			sim_constants,
			time,
		)
		CUDA.synchronize()
		# --------------------------------------------------------------------------------- ACQUIRE NEW INFORMATION -------------------------------------------------------------------------------- #
		@cuda threads = UGA_camps blocks = GT_spies shmem = sizeof(Int) gt_observe(
			GT_knowledge,
			GT_knowledge_count,
			GT_knowledge_prev_count,
			GT_knowledge_before_exchange,
			UGA,
			GT,
			GT_spies,
			GT_UGA_adj,
			sim_constants,
			time,
		)
		@cuda threads = GT_spies blocks = UGA_camps uga_observe(GT, GT_UGA_adj, UGA_hive_info, time)

		@cuda threads = size(GT_hive_info, 1) blocks = GT_spies gt_coordinate(GT_knowledge, GT_knowledge_count, GT_knowledge_prev_count, GT_hive_info, sim_constants)
		# ================================================================================= ====================== ================================================================================= #
		# --------------------------------------------------------------------------------- REINFORCEMENT LEARNING --------------------------------------------------------------------------------- #

		# @cuda threads = GT_spies blocks = 1 reinforcement_learning(
		# 	GT_Q_values,
		# 	learnt_params,
		# 	rewards,
		# 	GT_knowledge,
		# 	GT_knowledge_count,
		# 	GT_knowledge_prev_count,
		# 	GT_hive_info,
		# 	UGA_hive_info,
		# 	sim_constants,
		# 	time,
		# )

		# =============================================================================== =========================== ============================================================================== #
		# ------------------------------------------------------------------------------- MOVE GT WITH LEARNT WEIGHTS ------------------------------------------------------------------------------ #
		# move the players
		threads = (sim_constants.GT_interact_range, sim_constants.GT_interact_range)
		blocks = GT_spies
		shmem = sizeof(Float32) * 3 + sizeof(spy_range_info) * (1 + sim_constants.GT_interact_range * sim_constants.GT_interact_range)
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
		# ========================================================================================= ======== ======================================================================================= #
		# ----------------------------------------------------------------------------------------- PLOTTING --------------------------------------------------------------------------------------- #
		p = PlotlyJS.plot(
			PlutoPlotly.surface(
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
		# PlotlyJS.savefig(io, p; format = "png")  # Write PNG bytes to buffer
		# local_img_data = take!(io)

		# @sync begin
		# 	@async PlotlyJS.savefig(p, "img/$(time).png")
		# 	if time % 10 == 0
		# 		# Write PNG bytes to FFmpeg
		# 		put!(write_channel, local_img_data)
		# 		# write(ffmpeg_process, take!(io))
		# 	end
		# end
	end
	# if length(io.data) > 0
	# 	write(ffmpeg_process, take!(io))
	# end
	# close(ffmpeg_process)
	# isempty(write_channel) || put!(write_channel, take!(io))
	# close(write_channel)
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

	folder = "img"
	fps = 2
	ffmpeg_cmd = `./ffmpeg/ffmpeg -framerate $fps -start_number 1 -i "$folder/*.png" -c:v libx264 -r $fps -pix_fmt yuv420p "img/video.mp4"`
	run(ffmpeg_cmd)
end
