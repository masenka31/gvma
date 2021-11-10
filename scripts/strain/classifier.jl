using DrWatson
@quickactivate

# load data and necessary packages
include(srcdir("init_strain.jl"))

# divide to train/test data
ratio = 0.2
(Xtrain, ytrain), (Xtest, ytest) = train_test_split(X, y, ratio=ratio, seed=1)
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

opt = ADAM()

# loss and accuracy
loss(X, y_oh) = Flux.logitcrossentropy(full_model(X), y_oh)
accuracy(x, y) = mean(labelnames[Flux.onecold(full_model(x))] .== y)

@info "Model created, starting training..."

# train the model
@epochs 40 begin
    Flux.train!(loss, Flux.params(full_model), repeated((Xtrain, y_oh), 10), opt)
    train_acc = accuracy(Xtrain, ytrain)
    println("accuracy train: ", train_acc)
    println("accuracy test: ", accuracy(Xtest, ytest))
    if train_acc == 1
        @info "Train accuracy reached 100%, stopped training."
        break
    end
end

#############################
### Model and data saving ###
#############################

save_data = Dict(
    :model => full_model,
    :indices => (tix = tix, vix = vix),
    :mdim => mdim,
    :activation => activation,
    :aggregation => aggregation,
    :nlayers => nlayers,
    :train_accuracy => accuracy(X_train, y_train),
    :val_accuracy => accuracy(X_val, y_val),
)

safesave(datadir("strain classifier", savename(save_data, "bson")), save_data)

################################################
### UMAP latent space of train vs. test data ###
################################################

enc_tr = mill_model(Xtrain).data
enc_ts = mill_model(Xtest).data

for nn in 1:10
    emb_tr = umap(enc_tr, 2, n_neighbors = nn)
    p = scatter2(emb_tr; zcolor=gvma.encode(ytrain, labelnames),
                color=:tab10, markerstrokewidth=0, label="k = $nn");
    wsave(plotsdir("classifier", "train", "$(nn)_umap.png"), p)

    emb_ts = umap(enc_ts, 2, n_neighbors = nn)
    p = scatter2(emb_ts; zcolor=gvma.encode(ytest, labelnames),
                color=:tab10, markerstrokewidth=0, label="k = $nn");
    wsave(plotsdir("classifier", "test", "$(nn)_umap.png"), p)
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