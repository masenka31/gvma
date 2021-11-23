"""
This scripts needs Flux@12. Therefore, it is necessary to load
a different than GVMA directory and Manifest using DrWatson.

Do
`] activate`
to activate this environment.
"""

using IterTools
using Dictionaries
using Flux, Flux.Zygote
using ClusterLosses
using JSON

using DataFrames, CSV
using Random, StatsBase

# include the utility functions
include("src.jl")

cd("../..")
if pwd()[end-3:end] != "gvma"
    error("Change to the right directory!")
else
    @info "You are in the right directory, good luck ;)"
end

#################
### GVMA data ###
#################

seed = parse(Int64, ARGS[1])
ratio = parse(Float64, ARGS[1])

# load data and split
fps, prefix2int, labels = load_gvma_jaccard();
(fps_train, l_tr), (fps_test, l_test) = train_test_split(fps, labels; ratio=0.2, seed=1);

# initialize parameters
w = ones(Float32, length(prefix2int))

# batches
batchsize = 128
function prepare_minibatch()
    ix = rand(1:length(l_tr), batchsize)
    fps_train[ix], l_tr[ix]
end
mb_provider = IterTools.repeatedly(prepare_minibatch, 10)
batch = prepare_minibatch();

# loss function
lossf(x, y) = loss(Triplet(), jpairwise(x, prefix2int, σ.(w) .+ 1f-6), y)

# Flux parameters, optimizer
ps = Flux.params([w])
opt = ADAM()

# try training - does the loss decrease?
lossf(batch...)
Flux.train!(ps, mb_provider, opt) do x, y
    loss(Triplet(), jpairwise(x, prefix2int, σ.(w) .+ 1f-6), y)
end
lossf(batch...)

using Flux: @epochs

# train for given number of epochs
@epochs 100 begin
    Flux.train!(ps, mb_provider, opt) do x, y
        loss(Triplet(), jpairwise(x, prefix2int, σ.(w) .+ 1f-6), y)
    end
    @show lossf(batch...)
end

# save trained weights
safesave("data/weighted_jaccard/weights_ratio=$(ratio)_seed=$seed.bson", Dict(:w => w))

# calculate the resulting distance matrix
# how many threads are available?
@info "no of threads: " Threads.nthreads()
# can I calculate the distance matrix outside of training the weights?
# I should probably test whether the weights end up similar for different seeds...
w = BSON.load("data/weighted_jaccard/weights.bson")[:w]
@time L_full_weighted = _jpairwise(fps, prefix2int, σ.(w) .+ 1f-6)
safesave("data/weighted_jaccard/weigted_matrix.bson", Dict(:L => L_full_weighted))