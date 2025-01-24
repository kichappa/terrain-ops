# sim_constants are stored like so -------------------------------
#  1 | L
#  2 | seed
#  3 | altPs
#  4 | bush_density
#  5 | max_height
#  6 | GT_spies
#  7 | UGA_camps
#  8 | GT_interact_range
#  9 | UGA_interact_range
# 10 | MAX_ERROR
# 11 | escape_time
# 12 | capture_prob_no_bush
# 13 | capture_prob_bush
# ----------------------------------------------------------------

@inline function distance(x1, y1, x2, y2)
	return sqrt((x1 - x2)^2 + (y1 - y2)^2)
end

@inline function firepower_error(dist, MAX_ERROR, MAX_DIST)
	return MAX_ERROR * (dist / MAX_DIST)^4
end

@inline function size_error(dist, MAX_ERROR, MAX_DIST)
	return MAX_ERROR * ((dist - MAX_DIST) / MAX_DIST)^4
end

@inline function min_capture_probability(dist, MAX_DIST, MIN_PROB)
	return 1 - (1 - MIN_PROB) * (dist / MAX_DIST)^2
end

@inline function gt_gt(this, that, GT, GT_UGA_adj, sim_constants)
	dist = distance(GT[this, 1], GT[this, 2], GT[that, 1], GT[that, 2])
	# Update adjacency matrix
	GT_UGA_adj[that, this] = gt_uga_struct(dist <= sim_constants.GT_interact_range, dist)

	# GT_UGA_adj[that, this].visible = Integer(dist <= sim_constants.GT_interact_range)
	# GT_UGA_adj[that, this].distance = dist
end

@inline function uga_uga(this, that, UGA, GT_UGA_adj, sim_constants)
	dist = distance(UGA[this, 1], UGA[this, 2], UGA[that, 1], UGA[that, 2])
	# Update adjacency matrix
	GT_UGA_adj[that, this] = gt_uga_struct(dist <= sim_constants.UGA_interact_range, dist)
end

@inline function uga_gt(this, that, GT, UGA, GT_UGA_adj, sim_constants, time)
	# how far is it from me?
	dist = distance(UGA[this, 1], UGA[this, 2], GT[that, 1], GT[that, 2])
	# Update adjacency matrix
	if dist <= sim_constants.UGA_interact_range
		# is that in a bush?
		capture = Integer(
			rand(Float)
			<
			GT[that, 5] * sim_constant.capture_prob_bush + (1 - GT[that, 5]) * capture_probability(dist, sim_constants.UGA_interact_range, sim_constants.capture_prob_no_bush),
		)
		if capture == 1
			GT_UGA_adj[that, this] = gt_uga_struct(1, dist)
			GT[i, 3] = 1
			GT[i, 4] = time
		end
	else
		GT_UGA_adj[that, this] = gt_uga_struct(0, dist)
	end
end

@inline function gt_uga(this, that, GT, UGA, GT_UGA_adj, sim_constants)
	# how far is it from me?
	dist = distance(GT[this, 1], GT[this, 2], UGA[that, 1], UGA[that, 2])
	# Update adjacency matrix
	if dist <= sim_constants.GT_interact_range
		GT_UGA_adj[that, this] = gt_uga_struct(1, dist)
	else
		GT_UGA_adj[that, this] = gt_uga_struct(0, dist)
	end
end

function global_coherence(topo, bushes, UGA, GT, GT_spies, UGA_camps, GT_UGA_adj, GT_hive_info, UGA_hive_info, sim_constants, time)
	# Get thread indices
	that = threadIdx().x + (blockIdx().x - 1) * blockDim().x
	this = threadIdx().y + (blockIdx().y - 1) * blockDim().y

	# Check if indices are within bounds
	if that <= GT_spies + UGA_camps && that <= GT_spies + UGA_camps
		if this <= GT_spies && that <= GT_spies
	# 		# we are both GT spies
			gt_gt(this, that, GT, GT_UGA_adj, sim_constants)
	# 	# elseif this <= GT_spies && that > GT_spies
	# 	# 	# I am a GT spy and that is a UGA camp
	# 	# 	gt_uga(this, that, GT, UGA, GT_UGA_adj, sim_constants)
	# 	# elseif this > GT_spies && that <= GT_spies
	# 	# 	# I am a UGA camp and that is a GT spy
	# 	# 	uga_gt(this, that, GT, UGA, GT_UGA_adj, sim_constants, time)
	# 	# else
	# 	# 	# we are both UGA camps
	# 	# 	uga_uga(this, that, UGA, GT_UGA_adj, sim_constants)
		end
	end
	return
end

function uga_observe(topo, UGA, GT, GT_adj, UGA_adj, GT_UGA_adj, GT_hive_info, UGA_hive_info, sim_constants, randoms, time)
	# I am a UGA camp

	me = blockIdx().x
	that = threadIdx().x

	# am I close to that? What adjacency matrix should I check?
	if that <= GT_spies
		# that is a GT spy
		if GT_UGA_adj[me, that] == 1
			# that enemy GT spy could be in my sight (capture probability)
			# is that spy in the bush?
			that_in_bush = GT[that, 5]
			# do I capture it?
			capture = Integer(
				randoms[1, this] + 1 <
				2 * (that_in_bush * sim_constant.capture_prob_bush +
					 (1 - that_in_bush) * capture_probability(distance(UGA[me, 1], UGA[me, 2], GT[that, 1], GT[that, 2]), sim_constants.UGA_interact_range, sim_constants.capture_prob_no_bush)
				),
			)
			if capture == 1
				# I captured that spy
				GT[that, 3] = 1
				GT[that, 4] = time
			end


		end
	end
	return
end


function gt_observe(knowledge, k_count, prev_k_count, topo, UGA, GT, GT_adj, UGA_adj, GT_UGA_adj, GT_info, UGA_info, sim_constants, randoms, time)
	# I am a GT spy

	me = blockIdx().x
	that = threadIdx().x

	# am I captured? If so, is it time to move?
	if GT[me, 3] == 1 && !time - GT[me, 4] < sim_constants.escape_time
		return
	elseif GT[me, 3] == 1 && time - GT[me, 4] >= sim_constants.escape_time
		# time to move
		GT[me, 3] = 0
		GT[me, 4] = -1
	end

	count = CuStaticSharedArray(Int, 1)
	count[1] = k_count[me]

	# update the previous knowledge count so that gt_move can use knowledge from this time step to move
	if me == that
		prev_k_count[me] = k_count[me]
	end

	sync_threads()

	if me == that
		return
	end

	# am I close that? What adjacency matrix should I check?
	if that <= GT_spies
		# that is a GT spy
		if GT_adj[me, that] == 1
			# that teammate is close to me
			# get last 10 lines of knowledge from that teammate
			for i in 1:10
				current_count = CUDA.@atomic count[1] += 1
				knowledge[1, current_count, me] = knowledge[1, k_count[that]-10+i, that]
				knowledge[2, current_count, me] = knowledge[2, k_count[that]-10+i, that]
				knowledge[3, current_count, me] = knowledge[3, k_count[that]-10+i, that]
				knowledge[4, current_count, me] = knowledge[4, k_count[that]-10+i, that]
				knowledge[5, current_count, me] = knowledge[5, k_count[that]-10+i, that]
			end
		end
	else
		# that is a UGA camp
		if GT_UGA_adj[me, that] == 1
			# that enemy UGA camp is in my sight, I can observe it but also should be careful

			# collect knowledge about it from global knowledge list

			# how far is it from me?
			dist = distance(GT[me, 1], GT[me, 2], UGA[that, 1], UGA[that, 2])

			# how much error do I have in my knowledge?
			camp_size_error      = error(dist, sim_constant[10], sim_constants.GT_interact_range)
			camp_firepower_error = firepower_error(dist, sim_constant[10], sim_constants.GT_interact_range)
			# update my knowledge about that camp
			camp_size      = UGA[that, 3] * (1 + camp_size_error * randoms[1, this])
			camp_firepower = UGA[that, 4] * (1 + camp_firepower_error * randoms[2, this])

			# store the knowledge
			current_count = CUDA.@atomic count[1] += 1
			knowledge[1, current_count, me] = time
			knowledge[2, current_count, me] = camp_size
			knowledge[3, current_count, me] = camp_firepower
			knowledge[4, current_count, me] = camp_size_error
			knowledge[5, current_count, me] = camp_firepower_error
		end
	end

	return
end
