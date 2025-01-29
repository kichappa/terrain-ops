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
	height_range_advantage::Float32
	MAX_ERROR::Float32
	escape_time::Int32
	capture_prob_no_bush::Float32
	capture_prob_bush::Float32
	visible_prob::Float32
	sim_time::Int32
	gt_coord_size_threshold::Int32
	gt_coord_firepower_threshold::Int32
end

struct camp
	x::Int32
	y::Int32
	z::Float32
	size::Int32
	firepower::Int32
end

struct spy
	x::Int32
	y::Int32
	z::Float32
	frozen::Int32
	frozen_cycle::Int32
	in_bush::Int32
end

struct spy_knowledge
	time::Int32
	x::Int32
	y::Int32
	size::Float32
	firepower::Float32
	size_error::Float32
	firepower_error::Float32
	source::Int32
end
spy_knowledge() = spy_knowledge(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0)

struct spy_hive_knowledge
	time::Int32
	size::Float32
	firepower::Float32
	size_error::Float32
	firepower_error::Float32
end
spy_hive_knowledge() = spy_hive_knowledge(0, 0.0, 0.0, 0.0, 0.0)

struct camp_hive_knowledge
	x::Int32
	y::Int32
	time::Int16
	frozen::Int16
	frozen_cycle::Int32
end
camp_hive_knowledge() = camp_hive_knowledge(0, 0, 0, 0, 0)

struct adjacency
	visible::Int16 # the spy is visible to the camp
	interact::Int16# spy-spy = exchange of information, spy-camp = capture
	distance::Float32
end
adjacency() = adjacency(0, 0.0)
adjacency(i, d) = adjacency(0, i, d)

struct spy_range_info
	x::Int16
	y::Int16
	value::Float64
	ifbush::Int32
end
spy_range_info() = spy_range_info(0, 0, 0.0, 0)

struct q_values
	q_size::Float32
	q_firepower::Float32
	q_bush::Float32
	q_terrain::Float32
end
q_values() = q_values(0.0, 0.0, 0.0, 0.0)


function Base.show(io::IO, obj::camp)
	print(io, "$(obj.x), $(obj.y), $(obj.size), $(obj.firepower)")
end

function Base.show(io::IO, obj::spy)
	print(io, "$(obj.x), $(obj.y), $(obj.frozen), $(obj.frozen_cycle), $(obj.in_bush)")
end