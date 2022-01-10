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
yf = vcat(ytrain, ytest, ys)
Xf = cat(Xtrain, Xtest, Xs)
(Xtrain, ytrain), (Xtest, ytest) = train_test_split(X, y; ratio = 0.5, seed = 1)
(Xtrain, ytrain), (Xtest, ytest) = train_test_split(Xf, yf; ratio = 0.5, seed = 1)

# create minibatch function
batchsize = 128
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
model = triplet_mill_constructor(Xtrain, 32, relu, meanmax_aggregation, 2; odim = 10)
ps = Flux.params(model)

# try the loss function
lossf(x, y) = loss(Triplet(3f0), SqEuclidean(), model(x), y)
lossf(batch...)

# train for some max train time
max_train_time = 60*10
start_time = time()
while time() - start_time < max_train_time
    Flux.train!(ps, mb_provider, opt) do x, y
        lossf(x, y)
    end
    @show lossf(batch...)
end

# create train/test encoding and plot it
enc = model(Xtrain)
scatter2(enc, zcolor=gvma.encode(ytrain, labelnames), color=:tab10)
savefig("encoding1.png")

enc = model(Xtest)
scatter2(enc, color=gvma.encode(ytest, labelnames))
savefig("encoding2.png")

# create UMAP encoding from distance matrix and plot it
Md = pairwise(SqEuclidean(), model(Xtrain))
emb = umap(Md, 2, metric=:precomputed, n_neighbors=5)
scatter2(emb, color=color=gvma.encode(ytrain, labelnames))
savefig("embedding1.png")

Md2 = pairwise(SqEuclidean(), model(Xtest))
emb2 = umap(Md2, 2, metric=:precomputed, n_neighbors=5)
scatter2(emb2, color=color=gvma.encode(ytest, labelnames))
savefig("embedding2.png")

# for small classes
enc = model(cat(Xtest, Xs))
scatter2(enc, zcolor=gvma.encode(vcat(ytest, ys), unique(vcat(ytest, ys))), color=:jet, opacity=0.6)
savefig("encoding_Xs.png")

Md3 = pairwise(SqEuclidean(), model(cat(Xtest,Xs)))
emb = umap(Md3, 2; metric=:precomputed)
scatter2(emb, color=gvma.encode(vcat(ytest, ys), unique(vcat(ytest, ys))))
savefig("embedding_Xs.png")

###########
### kNN ###
###########

M = pairwise(SqEuclidean(), model(cat(Xtrain, Xtest)))
trx = 1:nobs(Xtrain)
tstx = length(trx)+1:nobs(Xf)
dm = M[tstx, trx]

foreach(k -> dist_knn(k,dm, ytrain, ytest), 1:20)


# get full distance matrix for all data
M = pairwise(SqEuclidean(), model(cat(Xtrain, Xtest, Xs)))

# only the main classes
trx = 1:nobs(Xtrain)
tstx = length(trx)+1:nobs(X)
dm = M[tstx, trx]

foreach(k -> dist_knn(k,dm, ytrain, ytest), 1:10)

# for the small classes
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

foreach(k -> dist_knn(k,dm, ytr, yts), 1:10)

##################################
### Clustering in latent space ###
##################################

using Clustering

enc = model(Xtrain)
c = kmedoids(pairwise(Euclidean(), enc), 10)
randindex(c, gvma.encode(ytrain, labelnames))

enc = model(Xtest)
c = kmedoids(pairwise(Euclidean(), enc), 10)
randindex(c, gvma.encode(ytest, labelnames))


scatter2(enc, zcolor=assignments(c))
savefig("cluster.png")

c2 = cutree(hclust(pairwise(Euclidean(), enc), linkage=:average), k=10)
randindex(c2, gvma.encode(ytest, labelnames))

scatter2(enc, zcolor=c2)
savefig("cluster.png")

enc = model(cat(Xtest, Xs))
M = pairwise(Euclidean(), enc)
c = kmedoids(M, 27)
randindex(c, gvma.encode(vcat(ytest, ys), vcat(labelnames, labelnames_s)))
scatter2(enc, 6, 10, color=assignments(c))
savefig("emb.png")

c = cutree(hclust(M, linkage=:ward), k=27)
randindex(c, gvma.encode(vcat(ytest, ys), vcat(labelnames, labelnames_s)))
scatter2(enc, 7, 4, color=c)
savefig("emb.png")

emb = umap(enc, 2)
scatter2(emb, color=gvma.encode(vcat(ytest, ys), vcat(labelnames, labelnames_s)))
savefig("emb.png")

M = pairwise(Euclidean(), emb)
c = cutree(hclust(M, linkage=:average), k=27)
randindex(c, gvma.encode(vcat(ytest, ys), vcat(labelnames, labelnames_s)))
