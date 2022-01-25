"""
Toy problem

Onehot na integers 1-100
V jedné třídě je vždy unikátní klíč
Vyzkoušet, jak se to bude chovat, jak pro novou třídu
Dokážu clusterovat známé třídy?
Učit se váhy listů, tabulku?
"""

using DrWatson
@quickactivate

using gvma
using gvma: jaccard_distance, jpairwise

using Distributions
using StatsBase
using Mill
using Random
using Flux
using Flux: @epochs
using Flux.Zygote
using Base.Iterators: repeated

using IterTools
using Clustering, Distances
using ClusterLosses

using Plots, UMAP


"""
    get_obs(x)

Returns bag indices from an of iterable collection.
For a vector of lengths `l = [4,5,2]` returns `obs = [1:4, 5:9, 10:11]`.
"""
function get_obs(x)
    l = nobs.(x)
    n = length(l)
    lv = vcat(0,l)
    mp = map(i -> sum(lv[1:i+1]), 1:n)
    mpv = vcat(0,mp)
    obs = map(i -> mpv[i]+1:mpv[i+1], 1:n)
end

function generate_mill_noise(class_code, λ)
    space = [1:100,1:100]
    s = map(i -> sample.(space), 1:rand(Poisson(λ)))
    s = setdiff(s, class_code)
    oh = map(x -> Flux.onehotbatch(x,1:100), s)
    an = cat(ArrayNode.(oh)...)
    bn = BagNode(an, map(i -> 2i-1:2i, 1:length(s)))
end

function generate_mill_data(n_classes, n_bags; λ = 20, seed = nothing)
    # class code can be fixed with a seed, noise cannot
    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    space = [1:100,1:100]
    class_code = [sample.(space) for _ in 1:n_classes]

    # reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    data = BagNode[]
    labels = Int[]

    for i in 1:n_bags
        ns = generate_mill_noise(class_code, λ)
        ix = sample(1:n_classes)
        oh_code = Flux.onehotbatch(class_code[ix], 1:100)
        bn_code = BagNode(ArrayNode(oh_code), [1:2])
        push!(data, cat(ns, bn_code))
        
        push!(labels, collect(1:n_classes)[ix])
    end
    return BagNode(cat(data...), get_obs(data)), labels, class_code
end

"""
    unpack(X)

Unpacks a BagNode{BagNode{ArrayNode}} structure to vector of vectors...
"""
function unpack(X)
    x1 = map(i -> X[i].data, 1:nobs(X))
    x2 = map(x -> x.data.data, x1)
    x3 = map(bag -> map(i -> bag[:, 2i-1:2i], 1:size(bag,2)÷2), x2)
end

seed = 2
n_classes = 5
n_bags = 1000

data, labels, code = generate_mill_data(n_classes, n_bags; λ=60, seed = seed)

(Xtrain, ytrain), (Xtest, ytest) = train_test_split(data, labels, 5; seed = seed)

model = reflectinmodel(Xtrain[1])
full_model = Chain(model, Mill.data, Dense(10,n_classes-1), softmax)

loss(x, y) = Flux.logitcrossentropy(full_model(x), y)
accuracy(x, y) = sum(x .== y) / length(x)
ps = Flux.params(full_model)

yoh = Flux.onehotbatch(ytrain, unique(ytrain))
opt = ADAM()

@epochs 100 begin
    Flux.train!(loss, ps, repeated((Xtrain, yoh), 10), opt)
    @show loss(Xtrain, yoh)
    model_labels = Flux.onecold(full_model(Xtrain), unique(ytrain))
    @show accuracy(ytrain, model_labels)
    model_test_labels = Flux.onecold(full_model(Xtest)[:, ytest .!= 5], unique(ytrain))
    @show accuracy(ytest[ytest .!= 5], model_test_labels)
end

# test model's accuracy
model_labels = Flux.onecold(full_model(Xtrain), unique(ytrain))
accuracy(ytrain, model_labels)
accuracy(ytest[ytest .!= 5], Flux.onecold(full_model(Xtest)[:, ytest .!= 5], unique(ytrain)))

# test clustering ability and the
enc_tr = model(Xtrain).data
scatter2(enc_tr, 3, 6, color=ytrain)

emb_tr = umap(pairwise(Euclidean(), enc_tr), 2, metric=:precomputed, n_neighbors=5)
scatter2(emb_tr, zcolor=ytrain, color=:jet)

enc_ts = model(Xtest).data[:, ytest .!= 5]
scatter2(enc_ts, 3, 6, color=ytest[ytest .!= 5])

emb_ts = umap(pairwise(Euclidean(), enc_ts), 2, metric=:precomputed, n_neighbors=5)
scatter2(emb_ts, zcolor=ytest[ytest .!= 5], color=:jet)

enc_tsx = model(Xtest).data
scatter2(enc_tsx, 3, 6, color=ytest)

emb_tsx = umap(pairwise(Euclidean(), enc_tsx), 2, metric=:precomputed, n_neighbors=5)
scatter2(emb_tsx, zcolor=ytest, color=:jet, label="", markerstrokewidth=0, markersize=3)




M = pairwise(Euclidean(), enc)
c = kmedoids(M, 4)
randindex(c, ytest)

emb = umap(enc, 2; n_neighbors=10)
emb = umap(M, 2; metric=:precomputed)
scatter2(emb, zcolor=ytest, color=:jet)

using gvma: dist_matrix
M = pairwise(Euclidean(), model(cat(Xtrain, Xtest)).data)
M2 = dist_matrix(M, Xtrain) 
foreach(k -> dist_knn(k, M2, ytrain, ytest), 1:10)


# using Jaccard Distance

JD = jpairwise(unpack(Xtrain))
emb = umap(JD, 2; metric=:precomputed)
scatter2(emb, color=ytrain)

JD = jpairwise(unpack(cat(Xtrain, Xtest)))
Mjd = dist_matrix(JD, Xtrain)

emb = umap(JD, 2; metric=:precomputed)
scatter2(emb, zcolor=vcat(ytrain, ytest), markerstrokewidth=0, markersize=3, color=:jet, label="")

foreach(k -> dist_knn(k, Mjd, ytrain, ytest), 1:10)


### Dictionary of terms
seed = 2
n_classes = 7
n_bags = 700
data, labels, code = generate_mill_data(n_classes, n_bags; λ = 60, seed = seed)

# split data
(Xtrain, ytrain), (Xtest, ytest) = train_test_split(data, labels, [5,6,7]; seed = seed)
# unpack the Mill data
xun = unpack(Xtrain);
xun_ts = unpack(Xtest);

# functions
function jaccard_weighted(j1, j2)
    int = Zygote.@ignore intersect(j1,j2)
    un  = Zygote.@ignore union(j1,j2)

    isempty(int) && return(one(eltype(weight_model(un[1]))))
    one(eltype(weight_model(un[1]))) - sum(weight_model.(int)) / sum(weight_model.(un))
end

function jpairwise_weighted(data)
    d = Zygote.Buffer(zeros(Float32, length(data), length(data)))
    for i in 2:length(data)
        for j in 1:i-1
            v = jaccard_weighted(data[i], data[j])
            d[i,j] = v
            d[j,i] = v
        end
    end
    copy(d)
end

# prepare minibatch
batchsize = 64
function prepare_minibatch()
    ix = rand(1:length(ytrain), batchsize)
    xun[ix], ytrain[ix]
end
mb_provider = IterTools.repeatedly(prepare_minibatch, 10)
batch = prepare_minibatch();

# create weight model
weight = Dense(100,1,σ)
weight_model = Chain(weight, mean)
ps = Flux.params(weight_model)
opt = ADAM(0.005)

# create loss and compute it on a batch
lossf(x, y) = ClusterLosses.loss(Triplet(), jpairwise_weighted(x), y)
lossf(batch...)

# train using triplet loss
@epochs 30 begin
    Flux.train!(ps, mb_provider, opt) do x, y
        lossf(x, y)
    end
    @show lossf(batch...)
end

# find key with maximum number of values in the data
# and see what weight it has
c = countmap(vcat(xun...))
max_key = findmax(c)[2];
weight_model(max_key)

# same for minimum
min_key = findmin(c)[2];
weight_model(min_key)

# weights for the unique code
code_onehot = map(x -> Flux.onehotbatch(x, 1:100), code)
weight_model.(code_onehot)


# compare Jaccard distance and weighted jaccard distance with leaves
# train data
JDtr = jpairwise(xun)
emb_jd_tr = umap(JDtr, 2; metric=:precomputed)
scatter2(emb_jd_tr, zcolor=ytrain, color=:Set1_4, label="", markerstrokewidth=0)

WJDtr = jpairwise_weighted(xun)
emb_wjd_tr = umap(WJDtr, 2; metric=:precomputed)
scatter2(emb_wjd_tr, zcolor=ytrain, color=:Set1_4, label="", markerstrokewidth=0)

# test data
JD = jpairwise(xun_ts)
emb_jd = umap(JD, 2; metric=:precomputed)
scatter2(emb_jd, zcolor=ytest, color=:Set1_7, label="", markerstrokewidth=0)

WJD = jpairwise_weighted(xun_ts)
emb_wjd = umap(WJD, 2; metric=:precomputed)
scatter2(emb_wjd, zcolor=ytest, color=:Set1_7, label="", markerstrokewidth=0)



# kNN on distance matrices
JD_d = jpairwise(vcat(xun, xun_ts))
WJD_d = jpairwise_weighted(vcat(xun, xun_ts))

function add_unknown_class(M, ytrain, ytest, n_classes; n = 20)
    trx = 1:length(ytrain)
    yf = vcat(ytrain, ytest)

    trx = vcat(trx, sample(length(trx)+1:size(M,2),n))
    ytr = yf[trx]
    while length(unique(ytr)) < n_classes
        trx = vcat(trx, sample(length(trx)+1:size(M,2),n))
        ytr = yf[trx]
    end

    tstx = setdiff(1:size(M,2), trx)
    yts = yf[tstx]
    dm = M[tstx, trx]

    return (ytr, yts, dm)
end

ytr1, yts1, dm1 = add_unknown_class(JD_d, ytrain, ytest, 5; n = 30)
ytr2, yts2, dm2 = add_unknown_class(WJD_d, ytrain, ytest, 5; n = 30)

max1 = findmax(map(k -> dist_knn(k, dm1, ytr1, yts1)[2], 1:20))
max2 = findmax(map(k -> dist_knn(k, dm2, ytr2, yts2)[2], 1:20))
max1, max2

# using clustering on the distance matrices
JD = jpairwise(xun_ts)
emb_jd = umap(JD, 2; metric=:precomputed)
scatter2(emb_jd, zcolor=ytest, color=:Set1_7, markerstrokewidth=0)

WJD = jpairwise_weighted(xun_ts)
emb_wjd = umap(WJD, 2; metric=:precomputed)
scatter2(emb_wjd, zcolor=ytest, color=:Set1_7, markerstrokewidth=0)

c1 = kmedoids(JD, 7)
c2 = kmedoids(WJD, 7)

randindex(c1, ytest)
randindex(c2, ytest)

scatter2(emb_jd, zcolor=assignments(c1), color=:Set1_7, markershape=marker_shape(ytest), markerstrokewidth=0)
scatter2(emb_wjd, zcolor=assignments(c2), color=:Set1_7, markershape=marker_shape(ytest), markerstrokewidth=0)

c1 = kmedoids(pairwise(Euclidean(), emb_jd), 7)
c2 = kmedoids(pairwise(Euclidean(), emb_wjd), 7)

randindex(c1, ytest)
randindex(c2, ytest)

function marker_shape(labels)
    shapes = [:circle, :square, :x, :utriangle, :dtriangle, :hline, :vline]
    s = Symbol[]
    for i in labels
        push!(s, shapes[i])
    end
    s
end

scatter2(emb_jd, zcolor=assignments(c1), color=:Set1_7, markershape=marker_shape(ytest), markerstrokewidth=0, colorbar_title="Cluster class", label="True class")
scatter2(emb_wjd, zcolor=assignments(c2), color=:Set1_7, markershape=marker_shape(ytest), markerstrokewidth=0, colorbar_title="Cluster class", label="True class")
savefig("plot.png")

# what about hierarchical clustering?

cl1 = cutree(hclust(JD; linkage=:average), k=7)
cl2 = cutree(hclust(WJD; linkage=:average), k=7)

randindex(cl1, ytest)
randindex(cl2, ytest)

cl1 = cutree(hclust(pairwise(Euclidean(), emb_jd); linkage=:average), k=7)
cl2 = cutree(hclust(pairwise(Euclidean(), emb_wjd); linkage=:average), k=7)

randindex(cl1, ytest)
randindex(cl2, ytest)

scatter2(emb_jd, zcolor=cl1, color=:Set1_5, markershape=marker_shape(ytest), markerstrokewidth=0, colorbar_title="Cluster class", label="True class")
scatter2(emb_wjd, zcolor=cl2, color=:Set1_5, markershape=marker_shape(ytest), markerstrokewidth=0, colorbar_title="Cluster class", label="True class")

################################################################
# implementing SetJaccard and WeightedJaccard in Distances.jl
using Distance: UnionMetric
import Distances: result_type

struct SetJaccard <: UnionMetric end
struct WeightedJaccard <: UnionMetric
    weight_model
end

function (dist::SetJaccard)(x, y)
    int = length(intersect(x,y))
    un = length(union(x,y))
    return (un - int)/un
end
result_type(dist::SetJaccard, x, y) = Float32

function (dist::WeightedJaccard)(j1, j2)
    int = Zygote.@ignore intersect(j1,j2)
    un  = Zygote.@ignore union(j1,j2)

    isempty(int) && return(one(eltype(dist.weight_model(un[1]))))
    one(eltype(dist.weight_model(un[1]))) - sum(dist.weight_model.(int)) / sum(dist.weight_model.(un))
end
result_type(dist::WeightedJaccard, x, y) = Float32

pairwise(SetJaccard(), xun) == jpairwise(xun)
pairwise(WeightedJaccard(weight_bag), xun) == jpairwise(xun)


# HEUREKA, THIS WORKS!

using NearestNeighborDescent

g = nndescent(xun_ts, 15, SetJaccard())
