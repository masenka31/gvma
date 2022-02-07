using DrWatson
@quickactivate

# load data and necessary packages
include(srcdir("init_strain.jl"))
using UMAP
using Distances
using ClusterLosses
using gvma: split_in_two

# get external parameters
seed = parse(Int64, ARGS[1])        # which seed to use for classes split
k_known = parse(Int64, ARGS[2])     # how many known classes
full = parse(Bool, ARGS[3])         # if to use all classes (bit+small) or only 10 big classes
clean = parse(Bool, ARGS[4])        # include clean or not
margin = parse(Float32, ARGS[5])    # the margin in triplet loss
batchsize = parse(Int64, ARGS[6])   # batchsize


# split data to known and unknown classes
Xf, yf = cat(X, Xs), vcat(y, ys)
if full
    (Xk, yk), (Xu, yu) = split_in_two(Xf, yf; k_known=k_known, clean=clean, seed=seed)
else
    (Xk, yk), (Xu, yu) = split_in_two(X, y; k_known=k_known, clean=clean, seed=seed)
end

# create loss function
triplet_loss(x, y) = ClusterLosses.loss(Triplet(margin), SqEuclidean(), model(x), y)

# create minibatch function
function minibatch()
    ix = sample(1:nobs(Xk), batchsize)
    return (Xk[ix], yk[ix])
end
batch = minibatch()

function val_batch()
    ix = sample(1:nobs(Xk), 1024)
    return (Xk[ix], yk[ix])
end
valbatch = val_batch()

# minibatch iterator
using IterTools
mb_provider = IterTools.repeatedly(minibatch, 10)

# sample parameters
function sample_params()
    mdim = sample(2 .^ [4,5,6,7])
    agg = sample(["meanmax_aggregation", "meanmax_aggregation", "max_aggregation"])
    activation = "relu"
    nlayers = sample([1,2,3])
    odim = sample([3,5,10,20])
    init_seed = rand(10000:100000)
    return (mdim=mdim, agg=agg, activation=activation, nlayers=nlayers, odim=odim, init_seed=init_seed)
end

# parameters and optimiser
vec = sample_params()
activation = eval(Symbol(vec.activation))
agg = eval(Symbol(vec.agg))
model = triplet_mill_constructor(Xk, vec.mdim, activation, agg, vec.nlayers; odim = vec.odim, seed = vec.init_seed)
ps = Flux.params(model)
opt = ADAM()

# training params
max_train_time=  60*60*2 # two hours maximum
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

    l = mean(triplet_loss(minibatch()...) for _ in 1:10)

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
enc_train = best_model(Xk)
M_train = pairwise(Euclidean(), enc_train)
emb_train = umap(enc_train, 2, n_neighbors=15)
E_train = pairwise(Euclidean(), emb_train)

res_known = cluster_data(M_train, E_train, yk, type="known")

# clustering - test data
enc_test = best_model(Xu)
M_test = pairwise(Euclidean(), enc_test)
emb_test = umap(enc_test, 2, n_neighbors=15)
E_test = pairwise(Euclidean(), emb_test)

res_unknown = cluster_data(M_test, E_test, yu, type="unknown")

# clustering - full data
enc_full = model(Xf)
M_full = pairwise(Euclidean(), enc_full)
emb_full = umap(enc_full, 2, n_neighbors=15)
E_full = pairwise(Euclidean(), emb_full)

res_full = cluster_data(M_full, E_full, yf, type="full")

results_dict = merge(res_known, res_unknown, res_full)
params_dict = Dict(keys(vec) .=> values(vec))
args_dict = Dict(:seed => seed, :k_known => k_known, :full => full, :clean => clean, :margin => margin, :batchsize => batchsize)

results = merge(results_dict, params_dict, args_dict)

name = savename(vec, "bson")
safesave(datadir("triplet_embedding", "full=$full", "clean=$clean","seed=$seed",name), results)