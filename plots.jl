function layout(sim_constants)
	return PlutoPlotly.Layout(
		scene = attr(
			xaxis = attr(range = [0, sim_constants.L], visible = false),
			yaxis = attr(range = [0, sim_constants.L], visible = false),
			zaxis = attr(range = [0, sim_constants.L], visible = false), width = 2560, height = 2560, camera = attr(
				eye = attr(x = 0, y = 0, z = 0.85),  # Set the camera position
				center = attr(x = 0, y = 0, z = 0),  # Set the center point to look at
				up = attr(x = 0, y = 1, z = 0),
			),
		),
		plot_bgcolor = "black",  # Set the background color to black
		paper_bgcolor = "black",
		width = 1000, height = 1000,
	)
end

function colorscale(sim_constants)
	min_v2 = 10 / (sim_constants.max_height + 30)
	max_v2 = (sim_constants.max_height + 10) / (sim_constants.max_height + 30)
	bush_v2 = (sim_constants.max_height + 20) / (sim_constants.max_height + 30)
	return [
		(0, "#3bff00"),  # Green
		(min_v2 - 0.000000001, "#3bff00"),  # Green
		(min_v2, "#222224"),  # Blue
		(min_v2 + 1 * (max_v2 - min_v2) / 5, "#3E2163"),  # Blue
		(min_v2 + 2 * (max_v2 - min_v2) / 5, "#88236A"),# Yellow
		(min_v2 + 3 * (max_v2 - min_v2) / 5, "#D04544"),# Yellow
		(min_v2 + 4 * (max_v2 - min_v2) / 5, "#F78D1E"),# Yellow
		(max_v2 - 0.000000001, "#F1E760"),# Yellow
		(max_v2, "#ffffff"),  # White
		(bush_v2, "#ffffff"),  # White
		(bush_v2 + 0.01, "#000000"),  # Black
		(1, "#000000"),  # Black
	]
end

function color_map(topo, bushes, GT, GT_spies, UGA, UGA_camps, sim_constants, max_threads = nothing)

	threads_x = min(max_threads, sim_constants.L)  # Limit to max_threads threads in the x dimension
	threads_y = min(max_threads, sim_constants.L)  # Limit to max_threads threads in the y dimension
	blocks_x = ceil(Int, sim_constants.L / threads_x)
	blocks_y = ceil(Int, sim_constants.L / threads_y)

	topo_color = CuArray(zeros(Float32, sim_constants.L, sim_constants.L))

	@cuda threads = (threads_x, threads_y) blocks = (blocks_x, blocks_y) color_kernel(topo_color, CuArray(topo), CuArray(bushes), GT, GT_spies, UGA, UGA_camps, sim_constants)
	return collect(topo_color)
end

function color_kernel(topo_color, topo, bushes, GT, GT_spies, UGA, UGA_camps, sim_constants)
	i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
	j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
	if 0 < i <= sim_constants.L && 0 < j <= sim_constants.L

		# check if I am inside a camp
		for camp in 1:UGA_camps
			# if distance(camp, i, j) < camp_size
			if sqrt((UGA[camp].x - i)^2 + (UGA[camp].y - j)^2) <= UGA[camp].size
				topo_color[i, j] = sim_constants.max_height + 20
				return
			end
		end

		# check if a spy is here
		for spy in 1:GT_spies
			if GT[spy].x == i && GT[spy].y == j
				topo_color[i, j] = sim_constants.max_height + 6
				return
			end
		end

		if (bushes[i, j] != 0) # bush
			topo_color[i, j] = -10
			return
		else
			topo_color[i, j] = topo[i, j]
			return
			# flag = 1
			# for k in 1:alt_ps_m
			# 	d = ((alt_p_gpu[k, 2] - i)^2 + (alt_p_gpu[k, 1] - j)^2)^0.5
			# 	if (d > 0 && flag == 1)
			# 		topo_color[i, j] += alt_p_gpu[k, 3] / d^power
			# 		norm += 1 / d^power
			# 	else
			# 		topo_color[i, j] = alt_p_gpu[k, 3]
			# 		flag = 0
			# 	end
			# end
			# if (flag == 1)
			# 	topo_color[i, j] /= norm
			# end
		end
	end
	topo_color[i, j] = 0.0
	return
end

function is_running_in_pluto()
	# Check environment variable
	if get(ENV, "PLUTO_PROJECT", "") != ""
		return true
	end

	# Check call stack
	for stack in stacktrace()
		if occursin("Pluto", string(stack))
			return true
		end
	end

	return false
end
