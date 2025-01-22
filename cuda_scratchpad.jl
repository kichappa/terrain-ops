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

function test_kernel(sim_constants)
    threadNum = threadIdx().x
    a = CuStaticSharedArray(Int, 1)
    a[1] = 0
    b=0
    b = CUDA.@atomic a[1] += 1
    @cuprintln("Thread $threadNum: a = $(a[1]), b = $b")
    sync_threads()
    b = a[1]
    @cuprintln("After sync threads $threadNum: a = $(a[1]), b = $b")
    return
end

@cuda threads = 8 blocks = 1 shmem = sizeof(Int) test_kernel(sim_constants)