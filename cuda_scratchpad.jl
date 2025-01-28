using CUDA, Random

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

    a = CuArray([1, 2, 3, 4, 5, 6, 7, 8])
    @cuprintln("e^$(threadIdx().x) = ", exp(threadIdx().x))
    @cuprintln("Here's a random number between 1 and 10: $(rand(1:10)) from thread ", threadIdx().x)
    
    f1, f2 = return_foos()
    @cuprintln("f1.bar1 = $(f1.bar1), f1.bar2 = $(f1.bar2)")
    @cuprintln("f2.bar1 = $(f2.bar1), f2.bar2 = $(f2.bar2)")
    # for i in 1:2
    #     @cuprintln("Hello from thread ", threadIdx().x, " i = $i")
    #     if rand(Float32) > 0.5
    #         i-=1
    #         @cuprintln("Here's an extra random number: ", rand(Float32), " from thread ", threadIdx().x, " i = $i")
    #         continue
    #     else 
    #         @cuprintln("Here's a random number: ", rand(Float32), " from thread ", threadIdx().x, " i = $i")

    #     end
    # end

    return
end

f = foo(1, 2)

# println(f[1], f[2])

array = [foo(i, j) for i in 1:8, j in 1:2]


println("Size of array: ", size(array))

for i in 1:8
    for j in 1:2
        print("$(array[i, j].bar1) $(array[i, j].bar2) |")
    end
    println()
end

@cuda threads = 1 blocks = 1 shmem = sizeof(Int) test_kernel(f, CUDA.zeros(Int32, 8))