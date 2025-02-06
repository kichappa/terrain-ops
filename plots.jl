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
		# plot_bgcolor = "black",  # Set the background color to black
		# paper_bgcolor = "black",

		plot_bgcolor = "white",  # Set the background color to white
		paper_bgcolor = "white",
		width = 1000, height = 1000,
	)
end

function colorscale(sim_constants)
	low_offset = 10
	high_offset = 20
	min_v2 = low_offset / (low_offset + sim_constants.max_height + high_offset)
	max_v2 = (low_offset + sim_constants.max_height) / (low_offset + sim_constants.max_height + high_offset)
	bush_v2 = (sim_constants.max_height + high_offset) / (low_offset + sim_constants.max_height + high_offset)
	color_points = [-10, 0, sim_constants.max_height, sim_constants.max_height + 6, sim_constants.max_height + 10, sim_constants.max_height + 20]
	# color_points = [-10, 0, sim_constants.max_height, sim_constants.max_height + 10, sim_constants.max_height + 20]
	colors = ["#3bff00", "#222224", "#F1E760", "#ffffff", "#ff0000", "#000000"]
	# colors = ["#3bff00", "#222224", "#F1E760", "#ffffff", "#000000"]
	topo_start = 0
	topo_colors = ["#222224", "#3E2163", "#88236A", "#D04544", "#F78D1E", "#F1E760"]

	range = color_points[end] - color_points[1]
	# define colorscale as an array of Tuple{Real, String}
	color_scale = Vector{Tuple{Real, String}}([((color_points[1] - color_points[1]) / range, colors[1])])
	# add bush
	color_scale = vcat(color_scale, [((color_points[2] - color_points[1]) / range - 0.000000001, colors[1])])

	# add topo
	# how many topo colors are there
	equal_parts = (color_points[3] - color_points[2]) / (length(topo_colors) - 1)
	for i in 1:length(topo_colors)-1
		color_scale = vcat(color_scale, [((color_points[2] + (i - 1) * equal_parts - color_points[1]) / range, topo_colors[i])])
	end
	color_scale = vcat(color_scale, [((color_points[3] - color_points[1]) / range, topo_colors[length(topo_colors)])])

	for i in 4:length(color_points)
		# add spy - active
		color_scale = vcat(color_scale, [((color_points[i-1] - color_points[1]) / range + 0.000000001, colors[i]), ((color_points[i] - color_points[1]) / range, colors[i])])
	end
	color_scale[begin] = (0, colors[begin])
	color_scale[end] = (1, colors[end])
	# add spy - free
	# color_scale = vcat(color_scale, [((color_points[3] - color_points[1]) / range + 0.000000001,colors[4]), ((color_points[4] - color_points[1]) / range, colors[4])])
	# # add spy - frozen
	# color_scale = vcat(color_scale, [((color_points[4] - color_points[1]) / range + 0.000000001,colors[5]), ((color_points[5] - color_points[1]) / range, colors[5])])
	# # add camp
	# color_scale = vcat(color_scale, [((color_points[5] - color_points[1]) / range + 0.000000001,colors[6]), ((color_points[6] - color_points[1]) / range, colors[6])])

	return color_scale

	# return [
	# 	(0, "#3bff00"),  # Green
	# 	(min_v2 - 0.000000001, "#3bff00"),  # Green
	# 	(min_v2, "#222224"),  # Blue
	# 	(min_v2 + 1 * (max_v2 - min_v2) / 5, "#3E2163"),  # Blue
	# 	(min_v2 + 2 * (max_v2 - min_v2) / 5, "#88236A"),# Yellow
	# 	(min_v2 + 3 * (max_v2 - min_v2) / 5, "#D04544"),# Yellow
	# 	(min_v2 + 4 * (max_v2 - min_v2) / 5, "#F78D1E"),# Yellow
	# 	(max_v2 - 0.000000001, "#F1E760"),# Yellow
	# 	(max_v2, "#ffffff"),  # White
	# 	(bush_v2, "#ffffff"),  # White
	# 	(bush_v2 + 0.01, "#000000"),  # Black
	# 	(1, "#000000"),  # Black
	# ]
end

function colorscale2(sim_constants)
	low_offset = 10
	high_offset = 20
	min_v2 = low_offset / (low_offset + sim_constants.max_height + high_offset)
	max_v2 = (low_offset + sim_constants.max_height) / (low_offset + sim_constants.max_height + high_offset)
	bush_v2 = (sim_constants.max_height + high_offset) / (low_offset + sim_constants.max_height + high_offset)
	color_points = [-10, 0, sim_constants.max_height, sim_constants.max_height + 6, sim_constants.max_height + 10, sim_constants.max_height + 20]
	colors = ["#3bff00", "#222224", "#F1E760", "#ffffff", "#ffff00", "#000000"]
	topo_colors = ["#222224", "#3E2163", "#88236A", "#D04544", "#F78D1E", "#F1E760"]

	color_scale = []

	return [
		(0, "#3bff00"),  # Green
		(min_v2 - 0.000000001, "#3bff00"),  # Green
		(min_v2, "#222224"),  # Blue
		(min_v2 + 1 * (max_v2 - min_v2) / 5, "#3E2163"),# Blue
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
			if GT[spy].x == i && GT[spy].y == j && GT[spy].frozen == 0
				topo_color[i, j] = sim_constants.max_height + 6
				return
			elseif GT[spy].x == i && GT[spy].y == j && GT[spy].frozen == 1
				topo_color[i, j] = sim_constants.max_height + 10
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
