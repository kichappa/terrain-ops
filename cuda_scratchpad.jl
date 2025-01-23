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

function test_kernel(f::foo)
    threadNum = threadIdx().x
    @cuprintln("Foo f: $(f.bar1) $(f.bar2) Thread: $threadNum")
    return
end

f = foo(1, 2)

println(f[1], f[2])

@cuda threads = 8 blocks = 1 shmem = sizeof(Int) test_kernel(f)