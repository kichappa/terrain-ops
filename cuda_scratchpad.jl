using CUDA, Random, StaticArrays

# Constants for setting up the simulation
L = 200 # 1
seed = 758 # 2
altPs = 7 # 3
max_height = 10 # 4
GT_spies = 50 # 5
UGA_camps = 5 # 6
UGA_interact_range = 20 # 7
GT_interact_range = 10 # 8

# GPU array of all constants
sim_constants = CuArray([L, seed, altPs, max_height, GT_spies, UGA_camps, UGA_interact_range, GT_interact_range])

struct foo
    bar1::Int32
    bar2::Float32
end

function return_array()
    return [i for i in 1:8]
end

function return_foos()
    return foo(1, 2), foo(3, 4)
end

function test_kernel(f, arr)
    # create a static array of size 2, with elements from f
    new_vector = SVector{2, Float32}(1.2141, -2.2134)
    @cuprintln("$(new_vector[1]), $(new_vector[2])")
    norm = sqrt(sum(x -> x^2, new_vector))
    new_vector = floor.(new_vector / norm*5) 
    @cuprintln("$(new_vector[1]), $(new_vector[2])")
    
    new_vector = max.(-1, min.(new_vector, 3))

    @cuprintln("$(new_vector[1]), $(new_vector[2])")
    @cuprintln("sizeof foo: $(sizeof(foo))")

    arr[1] = f[1]

    for _ in 1:10
        @cuprintln("$(rand(Float32))")
    end

    a = SVector{5, Float32}(1.2141, -2.2134, 0.0, 0.0, 0.0)
    a +=  SVector{5, Float32}(1.2141, 0.0, 0.0, 0.0, 0.0)
    a = exp.(a)

    return
end

f = foo(1, 2)

# println(f[1], f[2])

array = [foo(i, j) for i in 1:8, j in 1:2]


println("Size of array: ", size(array))

# for i in 1:8
#     for j in 1:2
#         print("$(array[i, j].bar1) $(array[i, j].bar2) |")
#     end
#     println()
# end
b = CuArray([foo(3, 4)])
@cuda threads = 1 blocks = 1 shmem = sizeof(Int) test_kernel(CuArray([f]), b)

b = collect(b)
println("b: $(b[1].bar1), $(b[1].bar2)")