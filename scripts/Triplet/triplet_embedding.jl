using DrWatson
@quickactivate

# load data and necessary packages
include(srcdir("init_strain.jl"))
using UMAP
using Distances

seed = 1
ratio = 0.5

newX, newY = X[y .!= "clean"], y[y .!= "clean"]
labelnames = unique(newY)
(Xtrain, ytrain), (Xtest, ytest) = train_test_split(newX, newY; ratio=ratio, seed=seed)

yoh_train = Flux.onehotbatch(ytrain, labelnames)
yoh_test = Flux.onehotbatch(ytest, labelnames)

init_seed = 2053
model = triplet_mill_constructor(Xtrain, 32, relu, meanmax_aggregation, 2; odim = 9, seed = init_seed)
opt = ADAM()

discriminative_loss(x, yoh) = Flux.logitcrossentropy(model(x), yoh)
accuracy(x, y) = mean(labelnames[Flux.onecold(model(x))] .== y)

# train the model
max_train_time = 60*3
start_time = time()
while time() - start_time < max_train_time
    Flux.train!(discriminative_loss, Flux.params(model), repeated((Xtrain, yoh_train), 10), opt)
    println("train loss: ", discriminative_loss(Xtrain, yoh_train))
    train_acc = accuracy(Xtrain, ytrain)
    println("accuracy train: ", train_acc)
    println("accuracy test: ", accuracy(Xtest, ytest))
end

# clustering
enc_test = model(Xtest)
M = pairwise(Euclidean(), enc_test)
emb = umap(enc_test, 2, n_neighbors=5)
E = pairwise(Euclidean(), emb)

cluster_data(M, E, ytest, type="test")

scatter2(enc_test, 3, 4, zcolor=gvma.encode(ytest, labelnames), color=:Paired_9)
savefig("enc.png")

scatter2(emb, zcolor=gvma.encode(ytest, labelnames), color=:Paired_9)
savefig("emb.png")

### triplet loss
using ClusterLosses
using gvma: split_in_two
# split data to known and unknown classes
Xf, yf = cat(X, Xs), vcat(y, ys)
(Xk, yk), (Xu, yu) = split_in_two(X, y; k_known=5)
(Xk, yk), (Xu, yu) = split_in_two(Xf, yf; k_known=16)
triplet_loss(x, y) = ClusterLosses.loss(Triplet(gap), SqEuclidean(), model(x), y)

# create minibatch function
batchsize = 128
function minibatch()
    ix = sample(1:nobs(Xk), batchsize)
    return (Xk[ix], yk[ix])
end
batch = minibatch()

# minibatch iterator
using IterTools
mb_provider = IterTools.repeatedly(minibatch, 10)

# parameters and optimiser
gap = 3f0
init_seed = 2053
model = triplet_mill_constructor(Xk, 32, relu, meanmax_aggregation, 2; odim = 5, seed = init_seed)
ps = Flux.params(model)
opt = ADAM(0.005)

# training
max_train_time=  60*1
start_time = time()
while time() - start_time < max_train_time
    Flux.train!(ps, mb_provider, opt) do x, y
        triplet_loss(x, y)
    end
    l = triplet_loss(batch...) # does this take too long?
    @show l
end

# clustering - train data
enc_train = model(Xk)
M_train = pairwise(Euclidean(), enc_train)
emb_train = umap(enc_train, 2, n_neighbors=15)
E_train = pairwise(Euclidean(), emb_train)

res_train = cluster_data(M_train, E_train, yk, type="train")

scatter2(enc_train, 5, 2, zcolor=gvma.encode(yk), color=:jet, ms=2, markerstrokewidth=0)
savefig("enc_train.svg")

scatter2(emb_train, zcolor=gvma.encode(yk), color=:jet, ms=2, markerstrokewidth=0)
savefig("emb_train.svg")

# clustering - test data
enc_test = model(Xu)
M_test = pairwise(Euclidean(), enc_test)
emb_test = umap(enc_test, 2, n_neighbors=15)
E_test = pairwise(Euclidean(), emb_test)

res_test = cluster_data(M_test, E_test, yu, type="test")

scatter2(enc_test, 1, 2, zcolor=gvma.encode(yu), color=:jet, ms=2, markerstrokewidth=0)
savefig("enc_test.svg")

scatter2(emb_test, zcolor=gvma.encode(yu), color=:jet, ms=2, markerstrokewidth=0)
savefig("emb_test.svg")

# clustering - full data
enc_full = model(Xf)
M_full = pairwise(Euclidean(), enc_full)
emb_full = umap(enc_full, 2, n_neighbors=15)
E_full = pairwise(Euclidean(), emb_full)

res_full = cluster_data(M_full, E_full, yf, type="test")

scatter2(enc_full, 1, 2, zcolor=gvma.encode(yf), color=:jet, ms=2, markerstrokewidth=0)
savefig("enc_full.svg")

scatter2(emb_full, zcolor=gvma.encode(yf), color=:jet, ms=2, markerstrokewidth=0)
savefig("emb_full.svg")



#########################################################################################
# classifier whether pair or not

function predict(x, y, z)
    x_hat = first_layer(x)
    y_hat = first_layer(y)
    sum(second_layer(vcat(x_hat, y_hat)))
end


function paired_loss(x, y, label)
    x_hat = first_layer(x)
    y_hat = first_layer(y)
    prediction = sum(second_layer(vcat(x_hat, y_hat)))
    Flux.binarycrossentropy(prediction, label)
end

# paired data
function sample_same(X, y)
    chosen = sample(y)
    vec = collect(1:length(y))
    ix = sample(vec[y .== chosen],2)
    return X[ix[1]], X[ix[2]], 1f0
end

function sample_diff(X, y)
    chosen = sample(y, 2, replace=false)
    vec = collect(1:length(y))
    ix1 = sample(vec[y .== chosen[1]])
    ix2 = sample(vec[y .== chosen[2]])
    return X[ix1], X[ix2], 0f0
end

paired = vcat(
    [sample_same(Xu, yu) for _ in 1:1000],
    [sample_diff(Xu, yu) for _ in 1:1000]
)

using Random
paired_data = shuffle(paired)
labels = [x[3] for x in paired_data]

first_layer = triplet_mill_constructor(Xk, 16, relu, meanmax_aggregation, 1, odim=5)
second_layer = Chain(Dense(10,16,relu), Dense(16,1,σ))

ps = Flux.params(first_layer, second_layer)
opt = ADAM()

@epochs 20 begin
    Flux.train!(paired_loss, ps, paired_data, opt)
    mean_loss = mean(map(x -> paired_loss(x...), paired_data))
    @show mean_loss
end

predictions = map(x -> predict(x...), paired_data)
# mean train loss
mean(map(x -> paired_loss(x...), paired_data))
# accuracy
sum(1 .- abs.(round.(predictions) .- labels)) / length(labels)

paired_test = vcat(
    [sample_same(Xk, yk) for _ in 1:1000],
    [sample_diff(Xk, yk) for _ in 1:1000]
)
labels_test = [x[3] for x in paired_test]

predictions = map(x -> predict(x...), paired_test)
mean(map(x -> paired_loss(x...), paired_test))
sum(1 .- abs.(round.(predictions) .- labels_test)) / length(labels_test)