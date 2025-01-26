using Random

function create_UGA(n_enem, topo, L, seed=0)
    # if !isnothing(seed)
    #     Random.seed!(seed)
    # end
    # Random.seed!(seed)
	enem_pos = rand(1:L, (n_enem,2))
	# enem_z = [topo[row[1], row[2]] for row in eachrow(enem_pos)]
	enem_r = rand(1:3, (n_enem,1))
    enem_firepower = rand(1:100, (n_enem,1))
    return CuArray([camp(enem_pos[i,1], enem_pos[i,2], enem_r[i], enem_firepower[i]) for i in 1:n_enem])
end

function create_GT(n_spies, topo, bushes, L, seed=0)
    # if !isnothing(seed)
    #     Random.seed!(seed)
    # end
    # Random.seed!(seed)
    println("L: ", L)
    spy_pos = rand(1:L, (n_spies,2))
    # spy_z = [topo[row[1], row[2]] for row in eachrow(spy_pos)]
    spy_frozen = rand(0:1, (n_spies,1))
    spy_frozen_cycle = fill(-1, (n_spies,1))
    spy_in_bush = [bushes[row[1], row[2]] for row in eachrow(spy_pos)]
    # spies = hcat(spy_pos, spy_r)
    #                    1, 2  , 3         , 4               , 5
    # return CuArray(hcat(spy_pos, spy_frozen, spy_frozen_cycle, spy_in_bush))
    return CuArray([spy(spy_pos[i,1], spy_pos[i,2], spy_frozen[i], spy_frozen_cycle[i], spy_in_bush[i]) for i in 1:n_spies])
end