using DrWatson
@quickactivate

using gvma
using ClusterLosses
using Distances
using Flux
using Flux: @epochs
using UMAP

# plotting on headless environment
using Plots
ENV["GKSwstype"] = "100"

include(srcdir("init_strain.jl"))
include(scriptsdir("triplet_loss", "cluster_fun.jl"))

# get args
seed = parse(Int64, ARGS[1])
all_classes = parse(Bool, ARGS[2])
distance = ARGS[3]

# divide data to train/test
yf = vcat(y, ys)
Xf = cat(X, Xs)

if all_classes
    (Xtrain, ytrain), (Xtest, ytest) = train_test_split(Xf, yf; ratio = 0.5, seed = seed)
else
    (Xtrain, ytrain), (Xtest, ytest) = train_test_split(X, y; ratio = 0.5, seed = seed)
end

# create minibatch function
batchsize = 128
function minibatch()
    ix = sample(1:nobs(Xtrain), batchsize)
    return (Xtrain[ix], ytrain[ix])
end
batch = minibatch()

# minibatch iterator
using IterTools
mb_provider = IterTools.repeatedly(minibatch, 100)

# initialize optimizer, model, parameters
opt = ADAM()
model = triplet_mill_constructor(Xtrain, 32, relu, meanmax_aggregation, 2; odim = 10)
ps = Flux.params(model)

# try the loss function
margin = 3f0
dist = eval(Symbol(distance))
lossf(x, y) = ClusterLosses.loss(Triplet(margin), dist(), model(x), y)
lossf(batch...)

# train for some max train time
max_train_time = 60*30 # half-hour
start_time = time()
while time() - start_time < max_train_time
    Flux.train!(ps, mb_provider, opt) do x, y
        lossf(x, y)
    end
    @show lossf(batch...)
end

##################################
### Clustering in latent space ###
##################################

using Clustering, Distances

# create train/test encoding
enc_train = model(Xtrain)
enc_test = model(Xtest)

# calculate simple pairwise distance matrix
M_train = pairwise(dist(), enc_train)
M_test = pairwise(dist(), enc_test)

# calculate 2D umap embedding
emb_train = umap(enc_train, 2)
emb_test = umap(enc_test, 2)

# calculate distance matrix on umap embedding
E_train = pairwise(Euclidean(), emb_train)
E_test = pairwise(Euclidean(), emb_test)

# cluster with k-medoids and hierarchical clustering
train_results = cluster_data(M_train, E_train, ytrain)
test_results = cluster_data(M_test, E_test, ytest, type="test")

###########
### kNN ###
###########

# calculate full distance matrix
M = pairwise(dist(), model(cat(Xtrain, Xtest)))
trx = 1:nobs(Xtrain)
tstx = length(trx)+1:nobs(cat(Xtrain, Xtest))
dm = M[tstx, trx]

knn_acc, knn_k = findmax(map(k -> dist_knn(k,dm, ytrain, ytest)[2], 1:10))
knn_results = Dict(
    :knn_acc => knn_acc,
    :knn_k => knn_k
)

###############
### Results ###
###############

full_results = merge(
    train_results,
    test_results,
    knn_results,
    Dict(
        :seed => seed,
        :all_classes => all_classes,
        :opt => opt,
        :model => model,
        :distance => distance
    )
)

name_dict = Dict(
    :seed => seed,
    :all_classes => all_classes,
    :knn_acc => knn_acc,
    :distance => distance
)
safesave(datadir("triplet_clustering", savename(name_dict, "bson")), full_results)