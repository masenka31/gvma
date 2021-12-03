using DrWatson
@quickactivate

using gvma
using ClusterLosses
using Distances
using Flux

# plotting on headless environment
using Plots
ENV["GKSwstype"] = "100"

include(srcdir("init_strain.jl"))

(Xtrain, ytrain), (Xtest, ytest) = train_test_split(X, y; ratio = 0.2, seed = 1)

batchsize = 128
function minibatch()
    ix = sample(1:nobs(Xtrain), batchsize)
    return (Xtrain[ix], ytrain[ix])
end
batch = minibatch()

using IterTools
mb_provider = IterTools.repeatedly(minibatch, 100)

lossf(x, y) = loss(Triplet(), SqEuclidean(), model(x), y)

opt = ADAM()
model = triplet_mill_constructor(Xtrain, 32, relu, meanmax_aggregation, 2; odim = 10)
ps = Flux.params(model)
lossf(batch...)

using Flux: @epochs

@epochs 10 Flux.train!(ps, mb_provider, opt) do x, y
    loss(Triplet(), SqEuclidean(), model(x), y)
end

enc = model(Xtrain)
scatter2(enc, color=gvma.encode(ytrain, labelnames))
savefig("encoding1.png")

enc = model(Xtest)
scatter2(enc, color=gvma.encode(ytest, labelnames))
savefig("encoding2.png")

Md = pairwise(SqEuclidean(), model(Xtrain))
emb = umap(Md, 2, metric=:precomputed)
scatter2(emb, color=color=gvma.encode(ytrain, labelnames))
savefig("embedding1.png")

Md2 = pairwise(SqEuclidean(), model(Xtest))
emb2 = umap(Md2, 2, metric=:precomputed)
scatter2(emb2, color=color=gvma.encode(ytest, labelnames))
savefig("embedding2.png")


M = pairwise(SqEuclidean(), model(cat(Xtrain, Xtest, Xs)))
trx = 1:nobs(Xtrain)
tstx = length(trx)+1:nobs(X)
dm = M[tstx, trx]

for k in 1:10
    dist_knn(k,dm, ytrain, ytest)
end

yf = vcat(ytrain, ytest, ys)
Xf = cat(Xtrain, Xtest, Xs)
trx = vcat(trx, sample(length(trx)+1:size(M,2),100))
ytr = yf[trx]
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


enc = model(cat(Xtest, Xs))
scatter2(enc, zcolor=gvma.encode(vcat(ytest, ys), unique(vcat(ytest, ys))), color=:jet)
savefig("encoding_Xs.png")

using UMAP
emb = umap(M, 2; metric=:precomputed)
scatter2(emb, color=gvma.encode(vcat(ytest, ys), unique(vcat(ytest, ys))))