using DrWatson
@quickactivate

# load data and necessary packages
include(srcdir("init_timedata.jl"))
using UMAP
using Distances
using ClusterLosses
using Flux
using Mill
using JsonGrinder
using IterTools

using Plots
ENV["GKSwstype"] = "100"
gr(ms=2, markerstrokewidth=0, color=:jet);

# get external parameters
seed = parse(Int64, ARGS[1])        # which seed to use for classes split

@time dataset = TDataset()
tr_x, val_x, ts_x = time_split(dataset; seed=seed)
# train_minibatch = minibatch(dataset, tr_x)

# create loss function
triplet_loss(x, y) = ClusterLosses.loss(Triplet(margin), SqEuclidean(), model(x), y)

# create minibatch function
function minibatch_train()
    x, y, t = minibatch(dataset, tr_x, batchsize)
    return x, y
end
batch = minibatch_train()

function minibatch_val()
    x, y, t = minibatch(dataset, val_x, 500)
    return x, y
end
valbatch = minibatch_val()

# minibatch iterator
mb_provider = IterTools.repeatedly(minibatch_train, 10)

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

# parameters and optimiser
vec = sample_params()
activation = eval(Symbol(vec.activation))
agg = eval(Symbol(vec.agg))
margin, batchsize = vec.margin, vec.batchsize
model = triplet_mill_constructor(batch[1], vec.mdim, activation, agg, vec.nlayers; odim = vec.odim, seed = vec.init_seed)
ps = Flux.params(model)
opt = ADAM()

# training params
max_train_time = 60*60*2 # two hours maximum
best_loss = Inf
best_model = deepcopy(model)
patience = 100
_patience = 0

# train
start_time = time()
while time() - start_time < max_train_time
    Flux.train!(ps, mb_provider, opt) do x, y
        triplet_loss(x, y)
    end

    l = mean(triplet_loss(minibatch_val()...) for _ in 1:10)

    if l < best_loss
        @show l
        global best_loss = l
        global best_model = deepcopy(model)
        global _patience = 0
    else
        global _patience += 1
    end
    if _patience == patience
        @info "Loss did not improve for $patience number of epochs, stopped training."
        break
    end
    if l < 5e-6
        @info "Training loss descreased enough, stopped training."
        break
    end
end

# clustering - train data
@time Xtr, ytr, ttr = minibatch(dataset, tr_x, 3000)

enc_train = best_model(Xtr)
M_train = pairwise(Euclidean(), enc_train)
cluster_data1(M_train, ytr, type="train")

scatter2(enc_train, zcolor=gvma.encode(ytr))
savefig("train2d.svg")
scatter3(enc_train, zcolor=gvma.encode(ytr))
savefig("train3d.svg")

emb_train = umap(enc_train, 2, n_neighbors=15)
E_train = pairwise(Euclidean(), emb_train)
res_train = cluster_data(M_train, E_train, ytr, type="train")

# clustering - test data
@time Xts, yts, _ = minibatch(dataset, ts_x, 3000)

enc_test = best_model(Xts)
M_test = pairwise(Euclidean(), enc_test)
cluster_data1(M_test, yts, type="test")

scatter2(enc_test, zcolor=gvma.encode(yts))
savefig("test2d.svg")
scatter3(enc_test, zcolor=gvma.encode(yts))
savefig("test3d.svg")

emb_test = umap(enc_test, 2, n_neighbors=15)
E_test = pairwise(Euclidean(), emb_test)
res_test = cluster_data(M_test, E_test, yts, type="test")

# kNN clustering
Xv, yv, tv = minibatch(dataset, val_x, 30000)
enc_val = best_model(Xv)
M = pairwise(Euclidean(), enc_val, enc_train)
foreach(k -> dist_knn(k, M, ytr, yv), 1:10)

M = pairwise(Euclidean(), enc_test, enc_train)
foreach(k -> dist_knn(k, M, ytr, yts)[2], 1:10)

# full results
results_dict = merge(res_known, res_unknown, res_full)
params_dict = Dict(keys(vec) .=> values(vec))
args_dict = Dict(:seed => seed, :margin => margin, :batchsize => batchsize)

results = merge(results_dict, params_dict, args_dict)

name = savename(vec, "bson")
safesave(datadir("triplet_embedding", "full=$full", "clean=$clean","seed=$seed",name), results)

function filter(d::TDataset, class::String)
    b = d.strain .== class
    ix = collect(1:length(b))[b]

    return d[ix]
end
function encode(x::Vector{DateTime})
    m = minimum(x)
    dif = x .- m
    dif_val = map(x -> x.value, dif)
    dif_val ./ maximum(dif_val)
end

class = "Capsfin"
xc, yc, tc = filter(dataset, class)
enc = best_model(xc)
M = pairwise(Euclidean(), enc)

scatter2(enc, zcolor=encode(tc))
savefig("class_2d.svg")
scatter3(enc, zcolor=encode(tc))
savefig("class_3d.svg")

experimental_loop(dataset, sample_params, 1, max_train_time=60)