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
	visible_prob::Float32
	sim_time::Int32
end

struct camp
	x::Int32
	y::Int32
	size::Int32
	firepower::Int32
end

struct spy
	x::Int32
	y::Int32
	frozen::Int32
	frozen_cycle::Int32
	in_bush::Int32
end

struct spy_knowledge
	time::Int32
	size::Float32
	firepower::Float32
	size_error::Float32
	firepower_error::Float32
	source::Int32
end
spy_knowledge() = spy_knowledge(0, 0.0, 0.0, 0.0, 0.0, 0)

struct spy_hive_knowledge
	time::Int32
	size::Float32
	firepower::Float32
	size_error::Float32
	firepower_error::Float32
end
spy_hive_knowledge = spy_hive_knowledge(0, 0.0, 0.0, 0.0, 0.0)

struct camp_hive_knowledge
	x::Int32
	y::Int32
	time::Int16
	frozen::Int16
	frozen_cycle::Int32
end
camp_hive_knowledge() = camp_hive_knowledge(0, 0, 0, 0, 0)

struct adjacency
	visible::Int16
	interact::Int16
	distance::Float32
end
adjacency() = adjacency(0, 0.0)
adjacency(i, d) = adjacency(0, i, d)

function Base.show(io::IO, obj::camp)
    print(io, "$(obj.x), $(obj.y), $(obj.size), $(obj.firepower)")
end

function Base.show(io::IO, obj::spy)
	print(io, "$(obj.x), $(obj.y), $(obj.frozen), $(obj.frozen_cycle), $(obj.in_bush)")
end