include("terrain.kernels.jl")

function generate_points(seed, n, altPs, max_height)
	Random.seed!(seed)
	alt_pos = rand(1:n, (altPs, 2))
	alt_h = rand(Float64, (altPs, 1)) * max_height
	hcat(alt_pos, alt_h)
	alt_p = hcat(alt_pos, alt_h)
	return alt_p
end

function topography_gpu(A, alt_p, power, max_threads = nothing)
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

	@cuda threads = (threads_x, threads_y) blocks = (blocks_x, blocks_y) alt_kernel(A_gpu, B, m, n, CuArray(alt_p), k, power)

	return collect(B)
end
