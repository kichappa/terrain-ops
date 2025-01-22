# sim_constants are stored like so -------------------------------
#  1 | L
#  2 | seed
#  3 | altPs
#  4 | max_height
#  5 | GT_spies
#  6 | UGA_camps
#  7 | GT_interact_range
#  8 | UGA_interact_range
#  9 | MAX_ERROR
# 10 | escape_time
# ----------------------------------------------------------------

@inline function size_error(x, MAX_ERROR, MAX_DIST)
	return MAX_ERROR * (x / MAX_DIST)^4
end

@inline function firepower_error(x, MAX_ERROR, MAX_DIST)
	return MAX_ERROR * ((x - MAX_DIST) / MAX_DIST)^4
end

function gt_observe(knowledge, k_count, prev_k_count, topo, UGA, GT, GT_adj, UGA_adj, GT_UGA_adj, GT_info, UGA_info, sim_constants, randoms, time)
	# I am a GT spy

	me = blockIdx().x
    that = threadIdx().x

    # am I captured? If so, is it time to move?
    if GT[me, 3] == 1 && !time - GT[me, 4] < sim_constants[10]
        return
    elseif GT[me, 3] == 1 && time - GT[me, 4] >= sim_constants[10]
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

	# am I close that? What adjacency matrix should I check?
	if that <= GT_spies
		# that is a GT spy
		if GT_adj[me, that] == 1
			# that teammate is close to me
			# get last 10 lines of knowledge from that teammate
            for i in 1:10
                current_count = CUDA.@atomic count[1] += 1
                knowledge[1, current_count, me] = knowledge[1, k_count[that] - 10 + i, that]
                knowledge[2, current_count, me] = knowledge[2, k_count[that] - 10 + i, that]
                knowledge[3, current_count, me] = knowledge[3, k_count[that] - 10 + i, that]
                knowledge[4, current_count, me] = knowledge[4, k_count[that] - 10 + i, that]
                knowledge[5, current_count, me] = knowledge[5, k_count[that] - 10 + i, that]
            end
		end
	else
		# that is a UGA camp
		if GT_UGA_adj[me, that] == 1
			# that enemy UGA camp is close to me
			# collect knowledge about it from global knowledge list
			# how far is it from me?
			dist = sqrt((GT[me, 1] - UGA[that, 1])^2 + (GT[me, 2] - UGA[that, 2])^2)
            # how much error do I have in my knowledge?
			camp_size_error      = error(           dist, MAX_ERROR, sim_constants[7])
            camp_firepower_error = firepower_error( dist, MAX_ERROR, sim_constants[7])
            # update my knowledge about that camp
			camp_size            = UGA[that, 3] * camp_size_error       * randoms[1, this]
			camp_firepower       = UGA[that, 4] * camp_firepower_error  * randoms[2, this]

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
