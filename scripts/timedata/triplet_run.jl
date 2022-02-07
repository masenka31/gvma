using DrWatson
@quickactivate

# load data and necessary packages
include(srcdir("init_timedata.jl"))

using gvma: InfoDataset, TimeDataset, Info
using UMAP
using Distances
using ClusterLosses
using Flux
using Mill
using JsonGrinder
using IterTools

using JSON, HDF5

include(scriptsdir("timedata", "run_loop.jl"))

# sample parameters
function sample_params()
    mdim = sample(2 .^ [4,5,6,7])
    agg = sample(["meanmax_aggregation", "meanmax_aggregation", "max_aggregation"])
    activation = "relu"
    nlayers = sample([1,2,3])
    odim = sample([3,5,10,20,50])
    margin = sample([1f0, 10f0, 0.1f0])
    batchsize = sample([128, 256, 512, 1024])
    init_seed = rand(10000:100000)
    return (mdim=mdim, agg=agg, activation=activation, nlayers=nlayers, odim=odim, margin=margin, batchsize=batchsize, init_seed=init_seed)
end

# load infodataset
infodataset = InfoDataset()
experimental_loop(infodataset, sample_params, 3, max_train_time=60*60*2)

# calculate results
# Xtr, ytr, Xv, yv, Xts, yts = sample_results_data(infodataset)
# model_file = "/home/maskomic/projects/gvma/data/timedata_results/triplet/models/seed=1/model_activation=relu_agg=meanmax_aggregation_batchsize=256_init_seed=60038_margin=1.0_mdim=32_nlayers=1_odim=3.bson"
# calculate_results(Xtr, ytr, Xv, yv, Xts, yts, model_file)