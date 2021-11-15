using DrWatson
@quickactivate

# load data and necessary packages
include(srcdir("init_strain.jl"))

# load the Jaccard matrix
using BSON
using LinearAlgebra
L = BSON.load(datadir("jaccard_matrix_strain.bson"))[:L]
# create full matrix
L_full = Symmetric(L)

# divide to train/test data with the Jaccard matrix properly transformed, too
ratio = 0.2
(Xtrain, ytrain), (Xtest, ytest), (dm_train, dm_test) = train_test_split_reg(X, y, L_full; ratio=ratio, seed=1)
y_oh = Flux.onehotbatch(ytrain, labelnames)     # onehot train labels
y_oh_t = Flux.onehotbatch(ytest, labelnames)    # onehot test labels

##########################################
### Parameters, model, loss & accuracy ###
##########################################

function sample_params()
    mdim = sample([8,16,32,64])
    activation = sample([sigmoid, tanh, relu, swish])
    aggregation = sample([mean_aggregation, max_aggregation, meanmax_aggregation])
    nlayers = sample(1:3)
    return mdim, activation, aggregation, nlayers
end

# Parameters
mdim, activation, aggregation, nlayers = 32, relu, meanmax_aggregation, 3
# mdim, activation, aggregation, nlayers = sample_params()

full_model = classifier_constructor(Xtrain, mdim, activation, aggregation, nlayers, seed = 13)
mill_model = full_model[1]

################################################
### Loss and Jaccard distance regulatization ###
################################################

"""
Preprocess: Get the indexes of nearest neighbors before training and save the
indexes, such that it can be called inside the loss function as a variable.
"""

Xnearest = ProductNode[]
n = nobs(Xtrain)
for i in 1:n
    x = Xtrain[i]
    yoh = y_oh[:, i]
    
    label = Flux.onecold(yoh, labelnames)
    nonlabel_ix = findall(x -> x != label, ytrain)

    neighbors_ix = partialsortperm(dm_train[1,:], 1:n)
    nonzero_ix = findall(x -> x > 0, dm_train[1,neighbors_ix])

    sd = setdiff(nonzero_ix, nonlabel_ix)
    first_ix = sd[1]

    x_nearest = Xtrain[first_ix]
    push!(Xnearest, x_nearest)
end

Xnearest = cat(Xnearest...)

# for each training point we need to find the closest neighbor of the same class
# and minimize the Euclidean distance between the point and its closest neighbor

function loss(x, yoh, x_nearest; α=1)
    # cross entropy loss
    ce = Flux.logitcrossentropy(full_model(x), y_oh)
    # Jaccard regulatization
    reg = Flux.mse(mill_model(x).data, mill_model(x_nearest).data)

    return ce + α*reg
end

accuracy(x, y) = mean(labelnames[Flux.onecold(full_model(x))] .== y)

opt = ADAM()

# train the model
@epochs 30 begin
    Flux.train!(loss, Flux.params(full_model), repeated((Xtrain, y_oh, Xnearest), 10), opt)
    println("train: ", loss(Xtrain, y_oh, Xnearest))
    train_acc = accuracy(Xtrain, ytrain)
    println("accuracy train: ", train_acc)
    println("accuracy test: ", accuracy(Xtest, ytest))
    if train_acc == 1
        @info "Train accuracy reached 100%, stopped training."
        break
    end
end

################################################
### UMAP latent space of train vs. test data ###
################################################

enc_tr = mill_model(Xtrain).data
enc_ts = mill_model(Xtest).data

for nn in 1:10
    emb_tr = umap(enc_tr, 2, n_neighbors = nn)
    p = scatter2(emb_tr; zcolor=gvma.encode(ytrain, labelnames),
                color=:tab10, markerstrokewidth=0, label="k = $nn");
    wsave(plotsdir("classifier + reg", "train", "$(nn)_umap.png"), p)

    emb_ts = umap(enc_ts, 2, n_neighbors = nn)
    p = scatter2(emb_ts; zcolor=gvma.encode(ytest, labelnames),
                color=:tab10, markerstrokewidth=0, label="k = $nn");
    wsave(plotsdir("classifier + reg", "test", "$(nn)_umap.png"), p)
end

######################################################
### kNN - using Euclidean distance on latent space ###
######################################################

# use Euclidean distance to get the distance matrix on latent space
using Distances
enc = mill_model(X).data
M = pairwise(Euclidean(), enc)
# get the same distance matrix train/test split as the initial data split
_, _, distance_matrix = train_test_split(X, y, M, ratio=ratio, seed=1)

# knn
for k in 1:10
    pred, acc = dist_knn(k, distance_matrix, ytrain, ytest)
end

########################################################
### k-medoids clustering on latent space (Euclidean) ###
########################################################

using Clustering

M_test = pairwise(Euclidean(), enc_ts)

k = 10
c = kmedoids(M_test, k)
clabels = assignments(c)
yenc = gvma.encode(ytest, labelnames)

ri = randindex(clabels, yenc)
c = counts(clabels, yenc)
vi = Clustering.varinfo(clabels, yenc)
vm = vmeasure(clabels, yenc)
mi = mutualinfo(clabels, yenc)