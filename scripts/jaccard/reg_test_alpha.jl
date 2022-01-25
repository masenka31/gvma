using DrWatson
@quickactivate

seed = parse(Int64, ARGS[1])
ratio = parse(Float64, ARGS[2])
α = parse(Float32, ARGS[3])

# load data and necessary packages
include(srcdir("init_strain.jl"))
using Distances, Clustering

# load the Jaccard matrix
using BSON
using LinearAlgebra
L = BSON.load(datadir("jaccard_matrix_strain.bson"))[:L]
# create full matrix
L_full = Symmetric(L)

# divide to train/test data with the Jaccard matrix properly transformed, too
(Xtrain, ytrain), (Xtest, ytest), (dm_train, dm_test) = train_test_split_reg(X, y, L_full; ratio=ratio, seed=seed)
y_oh = Flux.onehotbatch(ytrain, labelnames)     # onehot train labels
y_oh_t = Flux.onehotbatch(ytest, labelnames)    # onehot test labels

################################################
### Loss and Jaccard distance regulatization ###
################################################

# loss with regulatization
function loss_reg(x, yoh, x_nearest, α)
    # cross entropy loss
    ce = Flux.logitcrossentropy(full_model(x), yoh)
    # Jaccard regulatization
    reg = Flux.mse(mill_model(x).data, mill_model(x_nearest).data)

    return ce + α*reg
end
if α == 0
    loss_reg(x, yoh, x_nearest) = Flux.logitcrossentropy(full_model(x), yoh)
else 
    loss_reg(x, yoh, x_nearest) = loss_reg(x, yoh, x_nearest, α)
end

accuracy(x, y) = mean(labelnames[Flux.onecold(full_model(x))] .== y)

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

##########################################
### Train and save with regularization ###
##########################################

# Parameters are predefined
mdim, activation, aggregation, nlayers = 32, relu, meanmax_aggregation, 3
opt = ADAM()

# construct model
full_model = classifier_constructor(Xtrain, mdim, activation, aggregation, nlayers, seed = seed)
mill_model = full_model[1]

# train the model
@epochs 50 begin
    Flux.train!(loss_reg, Flux.params(full_model), repeated((Xtrain, y_oh, Xnearest), 10), opt)
    println("train loss: ", loss_reg(Xtrain, y_oh, Xnearest))
    train_acc = accuracy(Xtrain, ytrain)
    println("accuracy train: ", train_acc)
    println("accuracy test: ", accuracy(Xtest, ytest))
    if train_acc == 1
        @info "Train accuracy reached 100%, stopped training."
        break
    end
end

# results
train_acc = accuracy(Xtrain, ytrain)
test_acc = accuracy(Xtest, ytest)

# Clustering
enc_ts = mill_model(Xtest).data
M_test = pairwise(Euclidean(), enc_ts)

k = 10
c = kmedoids(M_test, k)
clabels = assignments(c)
yenc = gvma.encode(ytest, labelnames)

ri = randindex(clabels, yenc)
cs = counts(clabels, yenc)
vi = Clustering.varinfo(clabels, yenc)
vm = vmeasure(clabels, yenc)
mi = mutualinfo(clabels, yenc)

# save results
d1 = Dict(
    :train_acc => train_acc,
    :test_acc => test_acc,
    ((:ajd_randindex, :randindex, :mirkin, :hubert) .=> ri)...,
    :counts => cs,
    :varinfo => vi,
    :vmeasure => vm,
    :mutualinfo => mi,
    :seed => seed,
    :ratio => ratio,
    :α => α
)
safesave(datadir("regularization", "α=$(α)_seed=$(seed)_ratio=$(ratio).bson"), d1)