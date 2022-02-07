using DrWatson
@quickactivate

seed = parse(Int64, ARGS[1])
ratio = parse(Float64, ARGS[2])
α = parse(Float32, ARGS[3])
missing_class = ARGS[4]

# load data and necessary packages
include(srcdir("init_strain.jl"))
using Distances, Clustering
using ClusterLosses
using DataFrames: Not

# divide to train/test data with the Jaccard matrix properly transformed, too
(Xtrain, ytrain), (Xtest, ytest) = train_test_split(X, y, missing_class; ratio=ratio, seed=seed)

ix_missing = findall(x -> x == missing_class, labelnames)[1]
y_oh = Flux.onehotbatch(ytrain, labelnames[Not(ix_missing)])                            # onehot train labels
y_oh_t = Flux.onehotbatch(ytest[ytest .!= missing_class], labelnames[Not(ix_missing)])  # onehot test labels

Xt = Xtest[ytest .!= missing_class]
yt = ytest[ytest .!= missing_class]

# numerical train labels
labels = gvma.encode(ytrain, labelnames[Not(ix_missing)])

################################################
### Loss and Jaccard distance regulatization ###
################################################

# loss with regulatization
function loss_reg(x, yoh, labels, α)
    # cross entropy loss
    ce = Flux.logitcrossentropy(full_model(x), yoh)
    # Jaccard regulatization
    enc = mill_model(x).data
    trl = loss(Triplet(), SqEuclidean(), enc, labels)

    return ce + α*trl
end

if α == 0
    loss_reg(x, yoh, labels) = Flux.logitcrossentropy(full_model(x), yoh)
else 
    loss_reg(x, yoh, labels) = loss_reg(x, yoh, labels, α)
end

accuracy(x, y) = mean(labelnames[Flux.onecold(full_model(x))] .== y)

##########################################
### Train and save with regularization ###
##########################################

# Parameters are predefined
mdim, activation, aggregation, nlayers = 32, relu, meanmax_aggregation, 3
opt = ADAM()

# construct model
full_model = classifier_constructor(Xtrain, mdim, activation, aggregation, nlayers, odim = 9, seed = seed)
mill_model = full_model[1]

# train the model
start_time = time()
@epochs 300 begin
    Flux.train!(loss_reg, Flux.params(full_model), repeated((Xtrain, y_oh, labels), 10), opt)
    println("train loss: ", loss_reg(Xtrain, y_oh, labels))
    train_acc = accuracy(Xtrain, ytrain)
    println("accuracy train: ", train_acc)
    println("accuracy test: ", accuracy(Xt, yt))
    if train_acc == 1
        @info "Train accuracy reached 100%, stopped training."
        break
    end
    if time() - start_time > 60*60*3
        @info "Training time exceeded, stopped training."
        break
    end
end

train_acc = accuracy(Xtrain, ytrain)
test_acc = accuracy(Xt, yt)


d = Dict(Symbol.([:ratio, :seed, :α, :missing_class, :train_acc, :test_acc]) .=> [ratio, seed, α, missing_class, train_acc, test_acc])
dm = Dict(Symbol.([:ratio, :seed, :α, :missing_class, :train_acc, :test_acc, :full_model]) .=> [ratio, seed, α, missing_class, train_acc, test_acc, full_model])
safesave(datadir("models", "triplet_missing", savename("model", d, "bson")), dm)