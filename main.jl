using CUDA
using Random
using PlotlyJS, Plots
include("terrain.jl")

L = 200
seed = 758
altPs = 7
max_height = 10

topo = zeros(Float64, L, L);
topo = topography_gpu(topo, generate_points(seed, L, altPs, max_height), 2.0, 32)

# plotly()
Plots.plot(1:L, 1:L,topo, st=:surface, ratio=1, zlim=[0,L], xlim=[0,L], ylim=[0,L],xlabel="X", ylabel="Y", zlabel="Z", bgcolor="black")