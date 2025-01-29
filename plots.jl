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

function color_map(alt_p, A, enemiesInA, agentsInA, max_height, power)
	m, n = size(A)
	alt_p_gpu = CuArray(alt_p)
	A_gpu = CuArray(A)
	colors_A_gpu = similar(A_gpu)
	enemiesInA_gpu = CuArray(enemiesInA)
	agentsInA_gpu = CuArray(agentsInA)

	threads_x = min(max_threads, m)  # Limit to max_threads threads in the x dimension
	threads_y = min(max_threads, n)  # Limit to max_threads threads in the y dimension
	blocks_x = ceil(Int, m / threads_x)
	blocks_y = ceil(Int, n / threads_y)

	@cuda threads = (threads_x, threads_y) blocks = (blocks_x, blocks_y) color_kernel2(colors_A_gpu, alt_p_gpu, A_gpu, enemiesInA_gpu, agentsInA_gpu, m, n, max_height, power)

	return collect(colors_A_gpu)
end

function color_kernel2(colors_A_gpu, alt_p_gpu, A_gpu, enemiesInA_gpu, agentsInA_gpu, m, n, max_height, power)
	i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
	j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
	if 1 <= i <= m && 1 <= j <= n
		colors_A_gpu[i, j] = 0.0
		alt_ps_m, _ = size(alt_p_gpu)
		norm = 0

		if (enemiesInA_gpu[j, i] != 0) # enemy
			colors_A_gpu[i, j] = max_height + 6
		elseif (agentsInA_gpu[j, i] != 0) # agent
			colors_A_gpu[i, j] = max_height + 20
		elseif (A_gpu[i, j] != 0) # bush
			colors_A_gpu[i, j] = -10
		else
			flag = 1
			for k in 1:alt_ps_m
				d = ((alt_p_gpu[k, 2] - i)^2 + (alt_p_gpu[k, 1] - j)^2)^0.5
				if (d > 0 && flag == 1)
					colors_A_gpu[i, j] += alt_p_gpu[k, 3] / d^power
					norm += 1 / d^power
				else
					colors_A_gpu[i, j] = alt_p_gpu[k, 3]
					flag = 0
				end
			end
			if (flag == 1)
				colors_A_gpu[i, j] /= norm
			end
		end
	end
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
