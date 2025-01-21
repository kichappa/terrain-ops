using Random

function create_UGA(n_enem, topo, L, seed=0)
    # if !isnothing(seed)
    #     Random.seed!(seed)
    # end
    Random.seed!(seed)
	enem_pos = rand(1:L, (n_enem,2))
	enem_z = [topo[row[1], row[2]] for row in eachrow(enem_pos)]
	enem_r = rand(1:3, (n_enem,1))
	enemies = hcat(enem_pos, enem_r)
    return CuArray(enemies)
end

function create_GT(n_spies, topo, L, seed=0)
    # if !isnothing(seed)
    #     Random.seed!(seed)
    # end
    Random.seed!(seed)
    spy_pos = rand(1:L, (n_spies,2))
    spy_z = [topo[row[1], row[2]] for row in eachrow(spy_pos)]
    # spies = hcat(spy_pos, spy_r)
    return CuArray(spy_pos)
end