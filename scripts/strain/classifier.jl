using DrWatson
@quickactivate

# load data and necessary packages
include(srcdir("init_strain.jl"))

# divide to train/test data
ratio = 0.01
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

# model
m = reflectinmodel(
    Xtrain[1],
    k -> Dense(k, mdim, activation),
    d -> aggregation(d)
)

# create the net after Mill model
if nlayers == 1
    net = Dense(mdim, 10)
elseif nlayers == 2
    net = Chain(Dense(mdim, mdim, activation), Dense(mdim, 10))
else
    net = Chain(Dense(mdim, mdim, activation), Dense(mdim, mdim, activation), Dense(mdim, 10))
end

# connect the full model
full_model = Chain(m, Mill.data, net, softmax)

# try that the model works
try
    full_model(X[1])
catch e
    error("Model wrong, error: $e")
end

opt = ADAM()

# loss and accuracy
loss(X, y_oh) = Flux.logitcrossentropy(full_model(X), y_oh)
accuracy(x, y) = mean(labelnames[Flux.onecold(full_model(x))] .== y)

@info "Model created, starting training..."

# train the model
@epochs 40 begin
    Flux.train!(loss, Flux.params(full_model), repeated((Xtrain, y_oh), 10), opt)
    println("train: ", loss(Xtrain, y_oh))
    println("val: ", loss(Xtest, y_oh_t))
    println("accuracy train: ", accuracy(Xtrain, ytrain))
    println("accuracy validation: ", accuracy(Xtest, ytest))
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

#############################
### Evaluation, UMAP etc. ###
#############################

# encode labels to numbers
function encode(labels, labelnames)
    num_labels = ones(Int, length(labels))
    for i in 1:length(labels)
        v = findall(x -> x == labels[i], labelnames)
        num_labels[i] = v[1]
    end
    return num_labels
end

# test data UMAP for different number of neighbors
enc_tst = m(Xtest).data

for nn in [1,2,3,4,5,10,15,20]
    emb_tst = umap(enc_tst, 2, n_neighbors=nn)
    p = scatter2(emb_tst; zcolor=encode(ytest, labelnames), color=:tab10, markerstrokewidth=0);

    name = savename("nn=$nn", Dict(:mdim => mdim, :act => "$activation", :agg => "$aggregation", :nl => nlayers, :ratio => ratio), "png")
    safesave(plotsdir("strain classifier", name), p)
end
@info "Validation UMAP encoding saved."

######################################################
### kNN - using Euclidean distance on latent space ###
######################################################

# use Euclidean distance to get the distance matrix on latent space
using Distances
enc = m(X).data
M = pairwise(Euclidean(), enc)
# get the same distance matrix train/test split as the initial data split
_, _, distance_matrix = train_test_split(X, y, M, ratio=0.01, seed=1)

# knn
for k in 1:10
    pred, acc = dist_knn(k, distance_matrix, ytrain, ytest)
end