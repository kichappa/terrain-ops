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
	dist = distance(UGA[this-GT_spies].x, UGA[this-GT_spies].y, UGA[that-GT_spies].x, UGA[that-GT_spies].y)
	# Update adjacency matrix
	GT_UGA_adj[that, this] = adjacency(0, dist <= sim_constants.UGA_interact_range, dist)
end

@inline function uga_gt(this, that, GT, GT_spies, UGA, GT_UGA_adj, sim_constants::simulation_constants, time)
	# how far is it from me?
	dist = distance(UGA[this-GT_spies].x, UGA[this-GT_spies].y, GT[that].x, GT[that].y)
	# Update adjacency matrix
	if dist <= sim_constants.UGA_interact_range && GT[that].frozen == 0
		# is that in a bush?
		capture = Integer(
			rand(Float32)
			<
			GT[that].in_bush * sim_constants.capture_prob_bush
			+
			(1 - GT[that].in_bush) * min_capture_probability(dist, sim_constants.UGA_interact_range, sim_constants.capture_prob_no_bush),
		)
		if capture == 1
			GT_UGA_adj[that, this] = adjacency(0, 1, dist)
			GT[that] = spy(GT[that].x, GT[that].y, 1, time, GT[that].in_bush)
		elseif rand(Float32) < sim_constants.visible_prob
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
	dist = distance(GT[this].x, GT[this].y, UGA[that-GT_spies].x, UGA[that-GT_spies].y)
	# Update adjacency matrix
	GT_UGA_adj[that, this] = adjacency(0, dist <= sim_constants.GT_interact_range, dist)
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
					GT[this] = spy(GT[this].x, GT[this].y, 0, -1, GT[this].in_bush)
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
		@cuprintln("I am a UGA camp $me and I see a GT spy $that. Let's observe it.")
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

function gt_observe(knowledge, k_count, prev_k_count, UGA, GT_spies, GT_UGA_adj, GT_info, sim_constants::simulation_constants, time)
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
	# am I close that?
	if adj.interact == 1
		@cuprintln("\tI am a GT spy $me and I see a UGA camp $that. Let's observe it.")
		# that enemy UGA camp is in my sight, I can observe it but also should be careful

		# collect knowledge about it from global knowledge list

		# how far is it from me?
		dist = adj.distance

		# how much error do I have in my knowledge?
		camp_size_error      = size_error(dist, sim_constants.MAX_ERROR, sim_constants.GT_interact_range)
		camp_firepower_error = firepower_error(dist, sim_constants.MAX_ERROR, sim_constants.GT_interact_range)
		# update my knowledge about that camp
		camp_size      = UGA[that].size * (1 + camp_size_error * (rand(Float32) * 2 - 1))
		camp_firepower = UGA[that].firepower * (1 + camp_firepower_error * (rand(Float32) * 2 - 1))

		# store the knowledge
		current_count = CUDA.@atomic count[1] += 1
		knowledge[current_count, me] = spy_knowledge(time, camp_size, camp_firepower, camp_size_error, camp_firepower_error, me)
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

function gt_coordinate(knowledge, k_count, prev_k_count, hive_info)
	# Adding info collected by spy agents to spy_hive_knowledge
	me = threadIdx().x

	n = prev_k_count[me]
	total = k_count[me]

	# going through the latest entries for a particular spy
	for i in total-n+1:total
		spy_knowledge = knowledge[i,me]
		if spy_knowledge.source == me
			spy_hive_knowledge = spy_hive_knowledge(
				spy_knowledge.time,
				spy_knowledge.size,
				spy_knowledge.firepower,
				spy_knowledge.size_error,
				spy_knowledge.firepower_error
			)
		
		# check if this entry exist in GT_hive_info and add it if it doesn't exist
		updated_flag = CuArray([0])
		@cuda threads = size(hive_info, 1) blocks = 1 add_to_gt_hive_info(spy_hive_knowledge, hive_info, updated_flag)
		end
	end

	return
end

function add_to_gt_hive_info(new_info, hive_info, updated_flag)
	# Transfer the CuArray to CPU for processing
    idx = threadIdx().x
	if idx > length(hive_info) || updated_flag[1] == 1
		return
	end

	current = hive_info[idx]

	# Check if the entry already exists within thresholds
	if abs(current.size - new_info.size) <= gt_coord_size_threshold &&
		abs(current.firepower - new_info.firepower) <= gt_coord_firepower_threshold
		
		# Update if the new entry has better accuracy
		if new_info.size_error < current.size_error &&
			new_info.firepower_error < current.firepower_error
			hive_info[idx] = new_info
		end

		updated_flag[1] = 1
		return
	end

	# Replace the first zero-initialized element if not updated yet
	if current.size == 0.0 && current.firepower == 0.0 && updated_flag[1] == 0
		hive_info[idx] = new_info
		updated_flag[1] = 1
	end
	return
end
