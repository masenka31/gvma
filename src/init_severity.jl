using DrWatson
@quickactivate

using gvma

using JsonGrinder
using Statistics
using Flux
using Flux: throttle, @epochs
using StatsBase
using Base.Iterators: repeated

using UMAP, Plots
ENV["GKSwstype"] = "100"

# load dataset
# dataset = Dataset("./data/samples.csv", "./data/schema.bson") - without strain
dataset = Dataset("./data/samples_strain.csv", "./data/schema.bson")
X,type,strain = dataset[:]

# create data, labels
labelnames = unique(dataset.targets)
@time X, y = dataset[:]

labels = zeros(Int, 8000)
for i in 1:8000
    idx = findall(x -> x == 1, y[i] .== labelnames)
    labels[i] = idx[1]
end

# divide to train/test data
ix = sample(1:8000, 8000, replace=false)
tix = ix[1:6000]
vix = ix[6001:end]

X_train, y_train = X[tix], y[tix]
X_val, y_val = X[vix], y[vix]
y_oh = Flux.onehotbatch(y_train, labelnames)
y_oh_val = Flux.onehotbatch(y_val, labelnames)

@info "Data loaded and prepared."

function train_val_split(X, y)
    # divide to train/test data
    ix = sample(1:8000, 8000, replace=false)
    tix = ix[1:6000]
    vix = ix[6001:end]

    X_train, y_train = X[tix], y[tix]
    X_val, y_val = X[vix], y[vix]
    return (X_train, y_train), (X_val, y_val)
end