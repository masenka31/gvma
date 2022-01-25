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

# divide data to train/test
(Xtrain, ytrain), (Xtest, ytest) = train_test_split(X, y; ratio = 0.2, seed = 1)

# create minibatch function
batchsize = 256
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
model = triplet_mill_constructor(Xtrain, 32, relu, meanmax_aggregation, 3; odim = 10)
ps = Flux.params(model)

# try the loss function
lossf(x, y) = loss(Triplet(), SqEuclidean(), model(x), y)
lossf(batch...)

# train the model
@epochs 100 Flux.train!(ps, mb_provider, opt) do x, y
    loss(Triplet(), SqEuclidean(), model(x), y)
end

max_train_time = 120
start_time = time()
while time() - start_time < max_train_time
    Flux.train!(ps, mb_provider, opt) do x, y
        loss(Triplet(), SqEuclidean(), model(x), y)
    end
    @show lossf(batch...)
end

# create train/test encoding and plot it
enc = model(Xtrain)
scatter2(enc, color=gvma.encode(ytrain, labelnames))
savefig("encoding1.png")

enc = model(Xtest)
scatter2(enc, color=gvma.encode(ytest, labelnames))
savefig("encoding2.png")

# create UMAP encoding from distance matrix and plot it
Md = pairwise(SqEuclidean(), model(Xtrain))
emb = umap(Md, 2, metric=:precomputed)
scatter2(emb, color=color=gvma.encode(ytrain, labelnames))
savefig("embedding1.png")

Md2 = pairwise(SqEuclidean(), model(Xtest))
emb2 = umap(Md2, 2, metric=:precomputed)
scatter2(emb2, color=color=gvma.encode(ytest, labelnames))
savefig("embedding2.png")

# for small classes
enc = model(cat(Xtest, Xs))
scatter2(enc, zcolor=gvma.encode(vcat(ytest, ys), unique(vcat(ytest, ys))), color=:jet)
savefig("encoding_Xs.png")

Md3 = pairwise(SqEuclidean(), model(cat(Xtest,Xs)))
emb = umap(Md3, 2; metric=:precomputed)
scatter2(emb, color=gvma.encode(vcat(ytest, ys), unique(vcat(ytest, ys))))
savefig("embedding_Xs.png")

###########
### kNN ###
###########

# get full distance matrix for all data
M = pairwise(SqEuclidean(), model(cat(Xtrain, Xtest, Xs)))

# only the main classes
trx = 1:nobs(Xtrain)
tstx = length(trx)+1:nobs(X)
dm = M[tstx, trx]

for k in 1:10
    dist_knn(k,dm, ytrain, ytest)
end

# for the small classes
yf = vcat(ytrain, ytest, ys)
Xf = cat(Xtrain, Xtest, Xs)
trx = vcat(trx, sample(length(trx)+1:size(M,2),100))
ytr = yf[trx]
# sample long enough to have all classes in "train" data
while length(unique(ytr)) < 27
    trx = vcat(trx, sample(length(trx)+1:size(M,2),100))
    ytr = yf[trx]
end

tstx = setdiff(1:size(M,2), trx)
yts = yf[tstx]
dm = M[tstx, trx]

for k in 1:10
    dist_knn(k,dm, ytr, yts)
end