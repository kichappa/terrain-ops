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

@inline function gt_gt(this, that, GT, GT_UGA_adj, sim_constants::simulation_constants)
	dist = distance(GT[this].x, GT[this].y, GT[that].x, GT[that].y)
	# Update adjacency matrix
	GT_UGA_adj[that, this] = adjacency(0, dist <= sim_constants.GT_interact_range, dist)

	# GT_UGA_adj[that, this].interact = Integer(dist <= sim_constants.GT_interact_range)
	# GT_UGA_adj[that, this].distance = dist
end

@inline function uga_uga(this, that, GT_spies, UGA, GT_UGA_adj, sim_constants::simulation_constants)
	dist = distance(UGA[this-GT_spies].x, UGA[this-GT_spies].y, UGA[that-GT_spies].x, UGA[that-GT_spies].y) - UGA[this-GT_spies].size / 2 - UGA[that-GT_spies].size / 2
	# Update adjacency matrix
	GT_UGA_adj[that, this] = adjacency(0, dist <= sim_constants.UGA_interact_range, dist)
end

@inline function uga_gt(this, that, GT, GT_spies, UGA, GT_UGA_adj, sim_constants::simulation_constants, time)
	# how far is it from me?
	dist = distance(UGA[this-GT_spies].x, UGA[this-GT_spies].y, GT[that].x, GT[that].y) - UGA[this-GT_spies].size / 2
	# Update adjacency matrix
	if (dist <= sim_constants.UGA_interact_range * exp((UGA[this-GT_spies].z > GT[that].z) * (UGA[this-GT_spies].z - GT[that].z) * sim_constants.height_range_advantage))
		a = 0.0
		for _ in 1:2
			a = rand(Float32)
        	@cuprintln("$(a)")
    	end
		# a = rand(Float32)
		capture =
			a < (
				GT[that].in_bush * sim_constants.capture_prob_bush
				+
				(1 - GT[that].in_bush) * min_capture_probability(dist, sim_constants.UGA_interact_range, sim_constants.capture_prob_no_bush)
			)

		if GT[that].frozen == 0 && capture == 1
			if GT[that].in_bush == 1
				@cuprintln("\tUGA camp $(this-GT_spies) captured GT spy $that. Random was $a. It was in a bush. Capture probability was $(sim_constants.capture_prob_bush).")
			else
				@cuprintln("\tUGA camp $(this-GT_spies) captured GT spy $that. Random was $a. It was not in a bush. Capture probability was $(min_capture_probability(dist, sim_constants.UGA_interact_range, sim_constants.capture_prob_no_bush)).")
			end
			GT_UGA_adj[that, this] = adjacency(0, 1, dist)
			GT[that] = spy(GT[that].x, GT[that].y, GT[that].z, 1, time, GT[that].in_bush)
			# GT[that].frozen_cycle = time
			# GT[that].frozen = 1
		elseif capture == 0 && ((Bool(GT[that].in_bush) && (rand(Float32) < sim_constants.visible_prob) || !Bool(GT[that].in_bush)) || Bool(GT[that].frozen))
			GT_UGA_adj[that, this] = adjacency(1, 0, dist)
		else
			GT_UGA_adj[that, this] = adjacency(0, 0, dist)
		end
	else
		GT_UGA_adj[that, this] = adjacency(0, 0, dist)
	end
end

@inline function gt_uga(this, that, GT, GT_spies, UGA, GT_UGA_adj, sim_constants::simulation_constants)
	# how far is it from me?
	dist = distance(GT[this].x, GT[this].y, UGA[that-GT_spies].x, UGA[that-GT_spies].y) - UGA[that-GT_spies].size / 2
	# Update adjacency matrix
	GT_UGA_adj[that, this] = adjacency(0, dist <= sim_constants.GT_interact_range * exp((GT[this].z > UGA[that-GT_spies].z) * (GT[this].z - UGA[that-GT_spies].z) * sim_constants.height_range_advantage), dist)
end

function global_coherence(topo, bushes, UGA, GT, GT_spies, UGA_camps, GT_UGA_adj, GT_hive_info, UGA_hive_info, sim_constants::simulation_constants, time)
	# Get thread indices
	that = threadIdx().x + (blockIdx().x - 1) * blockDim().x
	this = threadIdx().y + (blockIdx().y - 1) * blockDim().y

	# Check if indices are within bounds
	if this <= GT_spies + UGA_camps && that <= GT_spies + UGA_camps
		if this == that
			if this <= GT_spies
				# I am a GT spy. am I captured? If so, is it time to move?
				if GT[this].frozen == 1 && time - GT[this].frozen_cycle >= sim_constants.escape_time
					# time to move
					GT[this] = spy(GT[this].x, GT[this].y, GT[this].z, 0, -1, GT[this].in_bush)
					# GT[this].frozen = 0
					# GT[this].frozen_cycle = -1
				end
			end
		elseif this <= GT_spies && that <= GT_spies
			# 		# we are both GT spies
			gt_gt(this, that, GT, GT_UGA_adj, sim_constants)
		elseif this <= GT_spies && that > GT_spies
			# I am a GT spy and that is a UGA camp
			gt_uga(this, that, GT, GT_spies, UGA, GT_UGA_adj, sim_constants)
		elseif this > GT_spies && that <= GT_spies
			# I am a UGA camp and that is a GT spy
			uga_gt(this, that, GT, GT_spies, UGA, GT_UGA_adj, sim_constants, time)
		else
			# we are both UGA camps
			uga_uga(this, that, GT_spies, UGA, GT_UGA_adj, sim_constants)
		end
	end
	return
end

function uga_observe(GT, GT_UGA_adj, UGA_hive_info, time)
	# I am a UGA camp

	me = blockIdx().x
	that = threadIdx().x

	# am I close to that? What adjacency matrix should I check?
	# that is a GT spy
	if GT_UGA_adj[me, that].interact == 1 || GT_UGA_adj[me, that].visible == 1
		if GT[that].frozen == 0
			@cuprintln("I am a UGA camp $me and I see a GT spy $that. Let's observe it.")
		else
			@cuprintln("I am a UGA camp $me and I see a captured GT spy $that. Let's observe it.")
		end
		# a new GT spy is close to me and I might have captured it. regardless, I should observe it
		UGA_hive_info[that] = camp_hive_knowledge(GT[that].x, GT[that].y, time, GT[that].frozen, GT[that].frozen_cycle)
	end
	return
end

function gt_exchange(knowledge, k_count, prev_k_count, UGA, GT_spies, GT_UGA_adj, GT_info, sim_constants::simulation_constants, time)
	# I am a GT spy

	me = blockIdx().x
	that = threadIdx().x

	my_count = CuStaticSharedArray(Int, 1)
	my_count[1] = k_count[me]

	its_count = k_count[that]

	adj = GT_UGA_adj[me, that]
	if adj.interact == 1 && me < that
		# that teammate is close to me
		# get last 10 lines of knowledge from that teammate
		my_count_before_exchange = my_count[1]
		k_count_it_gets = 0
		for i in 1:10
			# store the knowledge
			if its_count - i > 0 && knowledge[its_count-i, that].source != me
				current_count = CUDA.@atomic my_count[1] += 1
				knowledge[current_count, me] = knowledge[its_count-i, that]
			end
		end
		for i in 1:10
			# store the knowledge
			if my_count_before_exchange - i > 0 && knowledge[my_count_before_exchange-i, me].source != that
				k_count_it_gets += 1
				knowledge[its_count+i-1, that] = knowledge[my_count_before_exchange-i, me]
			end
		end
		CUDA.@atomic k_count[that] += k_count_it_gets
	end
	sync_threads()

	if me == that
		k_count[me] = my_count[1]
	end
	return
end

function gt_observe(knowledge, k_count, prev_k_count, UGA, GT, GT_spies, GT_UGA_adj, sim_constants::simulation_constants, time)
	# I am a GT spy

	me = blockIdx().x
	that = threadIdx().x

	count = CuStaticSharedArray(Int, 1)
	count[1] = k_count[me]

	# update the previous knowledge count so that gt_move can use knowledge from this time step to move
	if me == that
		prev_k_count[me] = k_count[me]
	end

	sync_threads()

	adj = GT_UGA_adj[me, that+GT_spies]
	if adj.interact == 1
		@cuprintln("\t\tI am a GT spy $me and I see a UGA camp $that. -------------------")
	end
	# am I close that?
	if adj.interact == 1 && GT[me].frozen == 0
		@cuprintln("\tI am a GT spy $me and I see a UGA camp $that. Let's observe it.")
		# that enemy UGA camp is in my sight, I can observe it but also should be careful

		# collect knowledge about it from global knowledge list

		# how far is it from me?
		dist = adj.distance

		# how much error do I have in my knowledge?
		camp_size_error      = size_error(dist, sim_constants.MAX_ERROR * exp((GT[me].z > UGA[that].z) * (UGA[that].z - GT[me].z) * sim_constants.height_range_advantage), sim_constants.UGA_interact_range)
		camp_firepower_error = firepower_error(dist, sim_constants.MAX_ERROR * exp((GT[me].z > UGA[that].z) * (UGA[that].z - GT[me].z) * sim_constants.height_range_advantage), sim_constants.UGA_interact_range)
		# update my knowledge about that camp
		camp_size      = UGA[that].size * (1 + camp_size_error * (rand(Float32) * 2 - 1))
		camp_firepower = UGA[that].firepower * (1 + camp_firepower_error * (rand(Float32) * 2 - 1))

		# store the knowledge
		current_count = CUDA.@atomic count[1] += 1
		knowledge[current_count, me] = spy_knowledge(time, UGA[that].x, UGA[that].y, camp_size, camp_firepower, camp_size_error, camp_firepower_error, me)
	end

	if GT[me].frozen == 1 && that == 1
		# I am captured
		# I can't observe anything
		# I can't move
		# I can't exchange knowledge
		# I can't hide in a bush
		@cuprintln("I am a GT spy $me and I am captured.")
		return
	end

	sync_threads()

	if that == 1
		if count[1] > k_count[me]
			@cuprintln("GT spy $me and I have observed $(count[1] - k_count[me]) UGA camps in this cycle.")
		end
		k_count[me] = count[1]
	end

	return
end

function gt_coordinate(knowledge, k_count, prev_k_count, hive_info, sim_constants::simulation_constants)
	# Adding info collected by spy agents to spy_hive_knowledge
	spy_id = blockIdx().x
	idx = threadIdx().x

	if spy_id > size(knowledge, 2) || idx > size(hive_info, 1)
		return
	end

	# going through the latest entries for a particular spy
	for i in prev_k_count[spy_id]:k_count[spy_id]-1
		spy_knowledge = knowledge[i, spy_id]
		if spy_knowledge.source == spy_id
			new_info = spy_hive_knowledge(
				spy_knowledge.time,
				spy_knowledge.size,
				spy_knowledge.firepower,
				spy_knowledge.size_error,
				spy_knowledge.firepower_error,
			)

			# check if this entry exist in GT_hive_info and add it if it doesn't exist
			current = hive_info[idx]

			hive_size_diff = abs(current.size - new_info.size)
			hive_firepower_diff = abs(current.firepower - new_info.firepower)

			# Check if the entry already exists within thresholds
			if hive_size_diff < sim_constants.gt_coord_size_threshold &&
			   hive_firepower_diff < sim_constants.gt_coord_firepower_threshold

				# Update if the new entry has better accuracy
				if new_info.size_error < current.size_error &&
				   new_info.firepower_error < current.firepower_error
					hive_info[idx] = new_info
					break
				end
			end

			# Replace the first zero-initialized element if not updated yet
			if current.size == 0.0 && current.firepower == 0.0
				hive_info[idx] = new_info
				break
			end
		end
	end
	return
end

function hide_in_bush(bushes, GT, sim_constants::simulation_constants)
	# I am a GT spy

	me = blockIdx().x

	# if threadIdx().x == 1 && threadIdx().y == 1
	# 	@cuprintln("Is GT spy $me captured? $(GT[me].frozen)")
	# end

	found = CuDynamicSharedArray(Int32, 1)
	found[1] = 0

	sync_threads()

	# if I am not in a bush, I should find the nearest bush and hide in it
	if GT[me].in_bush == 0 && found[1] <= 0
		dx = Int32(((threadIdx().y .- 1) .% (blockDim().y / 2) .+ 1) .* (-1) .^ ((threadIdx().y .- 1) .÷ (blockDim().y / 2)))
		dy = Int32(((threadIdx().x .- 1) .% (blockDim().x / 2) .+ 1) .* (-1) .^ ((threadIdx().x .- 1) .÷ (blockDim().x / 2)))

		my_x = GT[me].x
		my_y = GT[me].y

		if 0 < my_x + dx <= sim_constants.L && 0 < my_y + dy <= sim_constants.L && found[1] == 0
			if bushes[Int32(my_x + dx), Int32(my_y + dy)] == 1 && found[1] == 0
				flag = CUDA.@atomic found[1] += 1
				if flag == 0
					@cuprintln("GT spy $me found a bush at $(my_x + dx), $(my_y + dy). Hiding in it.")
					GT[me] = spy(my_x, my_y, GT[me].z, GT[me].frozen, GT[me].frozen_cycle, 1)
				else
					return
				end
			end
		end
	elseif GT[me].in_bush == 1 && found[1] == 0 && threadIdx().x == 1 && threadIdx().y == 1
		@cuprintln("GT spy $me fis already hiding in a bush.")
	end
	return
end

@inline function nearest_bushes(nearest_b, x, y, bushes, L, search_range)
	# find the nearest bushes to x, y
	count = 0
	for i in -search_range:search_range
		for j in -search_range:search_range
			if x + i > 0 && x + i <= L && y + j > 0 && y + j <= L
				if bushes[x+i, y+j] == 1
					count += 1
					nearest_b[count] = (x + i, y + j)
				end
			end
		end
	end
end

@inline function random_jump(amplitude, x, y, L)
	# randomly jump to a new location

	new_x = x + rand(-amplitude:amplitude)
	new_y = y + rand(-amplitude:amplitude)

	new_x = max(1, min(L, new_x))
	new_y = max(1, min(L, new_y))

	return Int32(new_x), Int32(new_y)
end

@inline function q_func(x, y, q, r, x0, y0)
	return q * (exp(3 / 2 - (3 * ((x - x0)^2 + (y - y0)^2)) / (2 * r^2)) * (3 * ((x - x0)^2 + (y - y0)^2) - r^2)) / (2 * r^2)
end


function gt_move(GT_Q_values, q_values, gt_knowledge, k_count, prev_k_count, topo, bushes, GT, sim_constants::simulation_constants)
	# Thread and block indices
	me = blockIdx().x
	x = GT[me].x
	y = GT[me].y

	# am I frozen?
	if GT[me].frozen == 1
		return
	end

	# Shared memory for spy_range_info
	shared_info_struct = CuDynamicSharedArray(spy_range_info, 1)
	shared_info_value = CuDynamicSharedArray(Float32, 1, sizeof(spy_range_info) * 1)
	shared_Q_values = CuDynamicSharedArray(Float32, blockDim().x * blockDim().y, sizeof(spy_range_info) * 1 + sizeof(Float32) * 1)
	if threadIdx().x == 1 && threadIdx().y == 1
		shared_info_struct[1] = spy_range_info(0, 0, -Inf, 0)
		shared_info_value[1] = -Inf
	end
	sync_threads()

	idx = Int32(threadIdx().x + x - sim_constants.GT_interact_range)
	idy = Int32(threadIdx().y + y - sim_constants.GT_interact_range)

	# Check if the indices are within the interaction range
	if 0 < idx <= sim_constants.L && 0 < idy <= sim_constants.L
		value::Float32 = 0.0
		# Compute the value based on knowledge
		is_bush = bushes[idx, idy]
		z = topo[idx, idy]
		for i in prev_k_count[me]:k_count[me]-1
			value += q_func(idx, idy, q_values.q_size, gt_knowledge[i, me].size, gt_knowledge[i, me].x, gt_knowledge[i, me].y) * gt_knowledge[i, me].size_error
			value += q_func(idx, idy, q_values.q_firepower, gt_knowledge[i, me].size + sim_constants.UGA_interact_range, gt_knowledge[i, me].x, gt_knowledge[i, me].y) * gt_knowledge[i, me].firepower_error
			value += q_values.q_bush * is_bush
			value += q_values.q_terrain * z
		end

		shared_Q_values[threadIdx().x, threadIdx().y] = value

		# Check if this value is the highest
		CUDA.@atomic shared_info_value[1] = max(shared_info_value[1], value)
		if value == shared_info_value[1]
			# Update shared memory atomically
			shared_info_struct[1] = spy_range_info(idx, idy, value, is_bush)
		end
	end
	sync_threads()


	# Move the GT spy in the direction of the highest value
	if threadIdx().x == 1 && threadIdx().y == 1 && shared_info_struct[1].value > -Inf
		if shared_info_struct[1].value == 0 || rand(Float32) < 0.3
			# Random jump
			new_x, new_y = random_jump(sim_constants.GT_step_size, x, y, sim_constants.L)
			GT[me] = spy(new_x, new_y, topo[new_x, new_y], GT[me].frozen, GT[me].frozen_cycle, bushes[new_x, new_y])
			@cuprintln("GT spy $me is randomly jumping to $(new_x), $(new_y). Max-Q was $(shared_info_value[1]). Information gained in this cycle: $(k_count[me] - prev_k_count[me]).")
			return
		end
		@cuprintln("GT spy $me has the highest q-value $(shared_info_value[1]) at $(shared_info_struct[1].x), $(shared_info_struct[1].y).")

		# if the best position is a bush, jump into it
		if shared_info_struct[1].in_bush == 1
			GT[me] = spy(shared_info_struct[1].x, shared_info_struct[1].y, topo[Int32(shared_info_struct[1].x), Int32(shared_info_struct[1].y)], GT[me].frozen, GT[me].frozen_cycle, 1)
			@cuprintln("GT spy $me is moving to a bush at $(shared_info_struct[1].x), $(shared_info_struct[1].y).")
			return
		end

		# Calculate the direction vector
		# dir_x = shared_info_struct[1].x - x
		# dir_y = shared_info_struct[1].y - y
		dir = SVector{2, Float32}(shared_info_struct[1].x - x, shared_info_struct[1].y - y)

		# Normalize the direction vector
		# dir_magnitude = sqrt(dir_x^2 + dir_y^2)
		dir_magnitude = sqrt(sum(x -> x^2, dir))
		if dir_magnitude > 0
			dir = dir / dir_magnitude * sim_constants.GT_step_size
			# dir_x /= dir_magnitude
			# dir_y /= dir_magnitude
		end

		# Move with step size GT_step_size and take the floor of the value
		new_xy = SVector{2, Float32}(x, y) + floor.(Int32, dir)
		# new_x = x + floor(Int, dir_x * sim_constants.GT_step_size)
		# new_y = y + floor(Int, dir_y * sim_constants.GT_step_size)

		# Ensure the new position is within the grid boundaries
		new_xy = max.(1, min.(sim_constants.L, new_xy))
		# new_x = max(1, min(sim_constants.L, new_x))
		# new_y = max(1, min(sim_constants.L, new_y))

		# Update the GT spy's position
		GT[me] = spy(new_xy[1], new_xy[2], topo[Int32(new_xy[1]), Int32(new_xy[2])], GT[me].frozen, GT[me].frozen_cycle, bushes[Int32(new_xy[1]), Int32(new_xy[2])])
		@cuprintln("GT spy $me is moving to $(new_xy[1]), $(new_xy[2]). Bush? $(GT[me].in_bush).")

		# Update the shared memory with the new position
		source_xy = new_xy - SVector{2, Float32}(x, y) + SVector{2, Float32}(sim_constants.GT_interact_range, sim_constants.GT_interact_range)
		shared_info_struct[1] = spy_range_info(new_xy[1], new_xy[2], shared_info_value[Int32(source_xy[1]), Int32(source_xy[2])], bushes[Int32(new_xy[1]), Int32(new_xy[2])])
		GT_Q_values[2, me] = shared_info_struct[1]
	end
	update_Q_values!(GT_Q_values, q_values, GT, topo, bushes, reinforcement_rewards, sim_constants)
	return
end

# function to compute q_values from Q TODO: check if this is correct
@inline function compute_q_values(Q_value, x, y, x0, y0, r)
    return (Q_value * (2 * r^2)) / (exp(3 / 2 - (3 * ((x - x0)^2 + (y - y0)^2)) / (2 * r^2)) * (3 * ((x - x0)^2 + (y - y0)^2) - r^2))
end

function compute_reward(spy, reinforcement_rewards)
	# Compute reward based on reinforcement map
	x, y = spy.x, spy.y
	reward = reinforcement_map[x, y]

	if spy.frozen == 1:
		return reinforcement_rewards.frozen_penalty
	
	if spy.in_bush == 1:
		return reinforcement_rewards.bush_reward

	#TODO: add camp reward (if a camp is spotted)

	return reinforcement_rewards.not_frozen_reward # Default reward

	return reward
end

function update_Q_values!(GT_Q_values, q_values, GT, reinforcement_rewards, sim_constants)
    alpha = 0.1  # Learning rate
    gamma = 0.9  # Discount factor

	total_new_Q = 0.0
    for me in 1:length(GT)
        x, y = GT[me].x, GT[me].y

        # Compute reward based on reinforcement map
        reward = compute_reward(GT[me], reinforcement_map)

        # Q-learning update
        old_Q = GT_Q_values[1, me]
        max_next_Q = maximum(GT_Q_values[2, :])  # Best future Q-value
        new_Q = old_Q + alpha * (reward + gamma * max_next_Q)

        # Store updated Q-value
        GT_Q_values[2, me] = new_Q  
		total_new_Q += new_Q
        # # Extract knowledge parameters
        # size = GT[me].size
        # firepower = GT[me].firepower
        # size_error = GT[me].size_error
        # firepower_error = GT[me].firepower_error

    end
	avg_new_Q = total_new_Q / length(GT)
	# Update q_values using inverse q_func
	# TODO: compute q_size, q_firepower, q_bush, q_terrain, we need x, y, x0, y0 for this (what to use?)
	# q_values.q_size = 
	# q_values.q_firepower = 
	# q_values.q_bush = 
	# q_values.q_terrain = 

    # Move learned Q-values forward
    GT_Q_values[1, :] .= GT_Q_values[2, :]
end
