################################################
### Activate environmet, packages, load data ###
################################################

using DrWatson
@quickactivate

include("utilities.jl")
using UMAP, Plots
include("plotting.jl")
using Statistics
using Flux
using Flux: throttle, @epochs
using StatsBase
using Base.Iterators: repeated

ENV["GKSwstype"] = "100"

# load dataset
dataset = Dataset("./data/samples.csv", "./data/schema.bson")

# create data, labels
labelnames = unique(dataset.targets)
@time X, y = dataset[:]

# divide to train/test data
ix = sample(1:8000, 8000, replace=false)
tix = ix[1:6000]
vix = ix[6001:end]

X_train, y_train = X[tix], y[tix]
X_val, y_val = X[vix], y[vix]
y_oh = Flux.onehotbatch(y_train, labelnames)
y_oh_val = Flux.onehotbatch(y_val, labelnames)

@info "Data loaded and prepared."

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
mdim, activation, aggregation, nlayers = sample_params()

# model
m = reflectinmodel(
    X,
    k -> Dense(k, mdim, activation),
    d -> aggregation(d)
)

if nlayers == 1
    net = Dense(mdim, 10)
elseif nlayers == 2
    net = Chain(Dense(mdim, mdim, activation), Dense(mdim, 10))
else
    net = Chain(Dense(mdim, mdim, activation), Dense(mdim, mdim, activation), Dense(mdim, 10))
end

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
@epochs 50 begin
    Flux.train!(loss, Flux.params(full_model), repeated((X_train, y_oh), 10), opt)
    println("train: ", loss(X_train, y_oh))
    println("val: ", loss(X_val, y_oh_val))
    println("accuracy train: ", accuracy(X_train, y_train))
    println("accuracy validation: ", accuracy(X_val, y_val))
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

safesave(datadir("classifier", savename(save_data, "bson")), save_data)

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

# train data
enc_t = m(X_train).data
emb_t = umap(enc_t, 2)
p = scatter2(emb_t; zcolor=encode(y_train, labelnames), color=:tab10, markerstrokewidth=0);

name = savename("train", Dict(:mdim => mdim, :act => "$activation", :agg => "$aggregation", :nl => nlayers), "png")
safesave(plotsdir("classifier", name), p)
@info "Train UMAP encoding saved."

# val data
enc_v = m(X_val).data
emb_v = umap(enc_v, 2)
p = scatter2(emb_v; zcolor=encode(y_val, labelnames), color=:tab10, markerstrokewidth=0);

name = savename("validation", Dict(:mdim => mdim, :act => "$activation", :agg => "$aggregation", :nl => nlayers), "png")
safesave(plotsdir("classifier", name), p)
@info "Validation UMAP encoding saved."

# full data
enc = m(X).data
emb = umap(enc, 2)
p = scatter2(emb; zcolor=encode(y, labelnames), color=:tab10, markerstrokewidth=0);

name = savename("full", Dict(:mdim => mdim, :act => "$activation", :agg => "$aggregation", :nl => nlayers), "png")
safesave(plotsdir("classifier", name), p)
@info "Full data UMAP encoding saved."

