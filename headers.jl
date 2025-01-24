struct simulation_constants
	L::Int32
	seed::Int32
	altPs::Int32
	bush_density::Int32
	max_height::Int32
	GT_spies::Int32
	UGA_camps::Int32
	GT_interact_range::Int32
	UGA_interact_range::Int32
	MAX_ERROR::Float32
	escape_time::Int32
	capture_prob_no_bush::Float32
	capture_prob_bush::Float32
	sim_time::Int32
end

# struct gt_gt_struct
# 	visible::Int32
# end
# gt_gt() = gt_gt(0)


# struct uga_uga_struct
# 	visible::Int32
# end
# uga_uga() = uga_uga(0)

struct gt_uga_struct
	visible::Int32
	distance::Float32
end
gt_uga_struct() = gt_uga_struct(0, 0.0)
