include("terrain.kernels.jl")
using Random

function generate_points(sim_constants)
	# Random.seed!(seed)
	alt_pos = rand(1:sim_constants.L, (Int64(sim_constants.altPs), 2))
	alt_h = rand(Float64, (Int64(sim_constants.altPs), 1)) * sim_constants.max_height
	hcat(alt_pos, alt_h)
	alt_p = hcat(alt_pos, alt_h)
	return alt_p
end

function topography(A, alt_p, power, b_density, max_threads = nothing)
	m, n = size(A)
	k, _ = size(alt_p)
	A_gpu = CuArray(A)
	B = similar(A_gpu)  # Create a GPU array of the same size and type as A
	if !isnothing(max_threads)
		if !isnothing(m)
			threads_x = min(max_threads, m)
		else
			threads_x = max_threads
		end
	else
		if !isnothing(m)
			threads_x = min(32, m)
		else
			threads_x = 26
		end
	end
	if !isnothing(max_threads)
		if !isnothing(n)
			threads_y = min(max_threads, n)
		else
			threads_y = max_threads
		end
	else
		if !isnothing(n)
			threads_y = min(32, n)
		else
			threads_y = 26
		end
	end
	# threads_x = min(max_threads, m)  # Limit to max_threads threads in the x dimension
	#    threads_y = min(max_threads, n)  # Limit to max_threads threads in the y dimension
	blocks_x = ceil(Int, m / threads_x)
	blocks_y = ceil(Int, n / threads_y)

	@cuda threads = (threads_x, threads_y) blocks = (blocks_x, blocks_y) alt_kernel(B, m, n, CuArray(alt_p), k, power)

	return collect(B), Int32.(collect(CUDA.rand(Float64, L, L) .< (b_density / 100)))
end

function slope(topo, max_threads = nothing)
	m, n = size(topo)
	# println("Sizes of m, n = ", m, " ",n)
	topo_gpu = CuArray(topo)
	outp = fill((0.0, 0.0), n, n)
	output_x = CuArray(fill(0.0, n, n))
	output_y = CuArray(fill(0.0, n, n))
	if !isnothing(max_threads)
		if !isnothing(m)
			threads_x = min(max_threads, m)
		else
			threads_x = max_threads
		end
	else
		if !isnothing(m)
			threads_x = min(32, m)
		else
			threads_x = 26
		end
	end
	if !isnothing(max_threads)
		if !isnothing(n)
			threads_y = min(max_threads, n)
		else
			threads_y = max_threads
		end
	else
		if !isnothing(n)
			threads_y = min(32, n)
		else
			threads_y = 26
		end
	end
	# threads_x = min(max_threads, m)  # Limit to max_threads threads in the x dimension
	#    threads_y = min(max_threads, n)  # Limit to max_threads threads in the y dimension
	blocks_x = ceil(Int, m / threads_x)
	blocks_y = ceil(Int, n / threads_y)

	@cuda threads = (threads_x, threads_y) blocks = (blocks_x, blocks_y) slope_kernel_5(topo_gpu, output_x, output_y, m, n)

	return output_x, output_y
end
