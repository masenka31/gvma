using Pkg
Pkg.activate(".")

using IterTools
using Dictionaries
using Flux, Flux.Zygote
using ClusterLosses
using JSON

using DataFrames, CSV
using Random, StatsBase

# include the utility functions
include("src.jl")

# change to the parent directory
cd("../..")
if pwd()[end-3:end] != "gvma"
    error("Change to the right directory!")
else
    @info "You are in the right directory, good luck ;)"
end

# check ability to parallelize
if Threads.nthreads() == 1
    error("You are using only 1 threads. Calculation needs to use more CPUs.")
else
    @info "You are using $(Threads.nthreads()) CPU cores."
end

# get parameters
seed = parse(Int64, ARGS[1])
ratio = parse(Float64, ARGS[2])

# load data, prefixes
fps, prefix2int, labels = load_gvma_jaccard();

using BSON
# load weights
w = BSON.load("data/weighted_jaccard/weights_ratio=$(ratio)_seed=$seed.bson")[:w]
# calculate the jaccard distance matrix
@info "Everything loaded, calculating the Jaccard distance matrix."
using LinearAlgebra
@time L = _jpairwise(fps, prefix2int, σ.(w) .+ 1f-6)
# save it
safesave("data/weighted_jaccard/weighted_matrix_ratio=$(ratio)_seed=$seed.bson", Dict(:L => L))