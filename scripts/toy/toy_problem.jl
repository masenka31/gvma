using DrWatson
@quickactivate

include(scriptsdir("toy", "src.jl"))

### Dictionary of terms
seed = 1
n_classes = 7
n_bags = 700
data, labels, code = generate_mill_data(n_classes, n_bags; λ = 60, seed = seed)

# split data
(Xtrain, ytrain), (Xtest, ytest) = train_test_split(data, labels, [5,6,7]; seed = seed)
# unpack the Mill data
xun = unpack(Xtrain);
xun_ts = unpack(Xtest);

instance_dict = unique(vcat(xun...))
instance2int = Dict(reverse.(enumerate(instance_dict)))

# prepare minibatch
batchsize = 32
function prepare_minibatch()
    ix = rand(1:length(ytrain), batchsize)
    xun[ix], ytrain[ix]
end
mb_provider = IterTools.repeatedly(prepare_minibatch, 10)
batch = prepare_minibatch();

# create weight model
W = ones(Float32, length(instance2int))
ps = Flux.params(W)
opt = ADAM(0.005)

# create loss and compute it on a batch
lossf(x, y) = ClusterLosses.loss(Triplet(), jwpairwise(x, instance2int, W), y)
lossf(batch...)

@epochs 50 begin
    Flux.train!(ps, mb_provider, opt) do x, y
        lossf(x, y)
    end
    @show lossf(batch...)
end

c = countmap(vcat(xun...))
max_key = findmax(c)[2]
min_key = findmin(c)[2]
weight_model(max_key, W, instance2int)
weight_model(min_key, W, instance2int)

code_onehot = map(x -> Flux.onehotbatch(x, 1:100), code)
map(c -> weight_model(c, W, instance2int), code_onehot)

######################################
### Scatter on UMAP representation ###
######################################

# look at the weighted jaccard distance
JDtr = jwpairwise(xun, instance2int, W)
emb_jd_tr = umap(JDtr, 2, metric=:precomputed, n_neighbors=30)
scatter2(emb_jd_tr, zcolor=ytrain, color=:Set1_4)

JD = jwpairwise(xun_ts, instance2int, W)
emb_jd = umap(JD, 2, metric=:precomputed, n_neighbors=50)
scatter2(emb_jd, zcolor=ytest, color=:Set1_7)

# look at the simple Jaccard distance
jd = pairwise(SetJaccard(), xun)
emb_tr = umap(jd, 2; metric=:precomputed, n_neighbors=50)
scatter2(emb_tr, zcolor=ytrain, color=:Set1_4, markerstrokewidth=0)

jd2 = pairwise(SetJaccard(), xun_ts)
emb_ts = umap(jd2, 2; metric=:precomputed, n_neighbors=50)
scatter2(emb_ts, zcolor=ytest, color=:Set1_7, markerstrokewidth=0)

##################
### Clustering ###
##################

# train data, different approaches
c1 = kmedoids(JDtr, 4)
randindex(c1,ytrain)
ptr1 = scatter2(emb_jd_tr, zcolor=assignments(c1), color=:Set1_4, markerstrokewidth=0, markershape=marker_shape(ytrain))
wsave(plotsdir("train_type=matrix_4-medoids.png"), ptr1)

c2 = kmedoids(pairwise(Euclidean(), emb_jd_tr), 4)
randindex(c2,ytrain)
ptr2 = scatter2(emb_jd_tr, zcolor=assignments(c2), color=:Set1_4, markerstrokewidth=0, markershape=marker_shape(ytrain))
wsave(plotsdir("train_type=umap_4-medoids.png"), ptr2)

c3 = cutree(hclust(JDtr, linkage=:average), k=4)
randindex(c3,ytrain)
ptr3 = scatter2(emb_jd_tr, zcolor=c3, color=:Set1_4, markerstrokewidth=0, markershape=marker_shape(ytrain))
wsave(plotsdir("train_type=matrix_4-hclust-average.png"), ptr3)

c4 = cutree(hclust(pairwise(Euclidean(), emb_jd_tr), linkage=:average), k=4)
randindex(c4,ytrain)
ptr4 = scatter2(emb_jd_tr, zcolor=c4, color=:Set1_4, markerstrokewidth=0, markershape=marker_shape(ytrain))
wsave(plotsdir("train_type=umap_4-hclust-average.png"), ptr4)

# test data, different approaches
c1 = kmedoids(JD, 7)
randindex(c1,ytest)
pts1 = scatter2(emb_jd, zcolor=assignments(c1), color=:Set1_7, markerstrokewidth=0, markershape=marker_shape(ytest))
wsave(plotsdir("test_type=matrix_7-medoids.png"), pts1)

c2 = kmedoids(pairwise(Euclidean(), emb_jd), 7)
randindex(c2,ytest)
pts2 = scatter2(emb_jd, zcolor=assignments(c2), color=:Set1_7, markerstrokewidth=0, markershape=marker_shape(ytest))
wsave(plotsdir("test_type=umap_7-medoids.png"), pts2)

c3 = cutree(hclust(JD, linkage=:average), k=7)
randindex(c3,ytest)
pts3 = scatter2(emb_jd, zcolor=c3, color=:Set1_7, markerstrokewidth=0, markershape=marker_shape(ytest))
wsave(plotsdir("test_type=matrix_7-hclust-average.png"), pts3)

c4 = cutree(hclust(pairwise(Euclidean(), emb_jd), linkage=:average), k=7)
randindex(c4,ytest)
pts4 = scatter2(emb_jd, zcolor=c4, color=:Set1_7, markerstrokewidth=0, markershape=marker_shape(ytest))
wsave(plotsdir("test_type=umap_7-hclust-average.png"), pts4)

####################################
### Linear regression on weights ###
####################################

uniques = unique(vcat(xun...))

c = countmap(vcat(xun...))
k = collect(keys(c))
v = collect(values(c))
kw = map(x -> weight_model(x, W, instance2int), k)
p1 = scatter(v, kw, ylims=(0,1.05), label="", xlabel="# of occurences", ylabel="weight")
wsave(plotsdir("weights_by_occurence.png"), p1)

c = countmap(vcat(xun_ts...))
k = collect(keys(c))
v = collect(values(c))
kw = map(x -> weight_model(x, W, instance2int), k)
is_in = map(x -> in(x, uniques), k)
p2 = scatter(v, kw, color=Int.(is_in), ylims=(0,1.05), label="", xlabel="# of occurences", ylabel="weight")
wsave(plotsdir("weights_by_occurence_test.png"), p2)


##################################
### Create Jaccard as a metric ###
##################################

using Distances: UnionMetric
import Distances: result_type

struct SetJaccard <: UnionMetric end

function (dist::SetJaccard)(x, y)
    int = length(intersect(x,y))
    un = length(union(x,y))
    return (un - int)/un
end
result_type(dist::SetJaccard, x, y) = Float32

pairwise(SetJaccard(), xun[1:20])

struct WeightedJaccard <: UnionMetric
    instance2int::Dict
    W::Vector
end
WeightedJaccard(instance2int::Dict) = WeightedJaccard(instance2int, ones(Float32, length(instance2int)))
function WeightedJaccard(data::Vector)
    instance_dict = unique(data)
    instance2int = Dict(reverse.(enumerate(instance_dict)))
    WeightedJaccard(instance2int)
end
import Base: length
length(dist::WeightedJaccard) = length(dist.W)

function weight_model(x, w, instance2int)
    if x in keys(instance2int)
        return σ(w[instance2int[x]])
    else
        return 1f0
    end
end

function (dist::WeightedJaccard)(x, y)
    int = Zygote.@ignore intersect(x, y)
    un  = Zygote.@ignore union(x, y)

    int_w = map(i -> weight_model(i, dist.W, dist.instance2int), int)
    un_w = map(i -> weight_model(i, dist.W, dist.instance2int), un)

    isempty(int) && return(one(eltype(W)))
    one(eltype(int_w)) - sum(int_w) / sum(un_w)
end
result_type(dist::WeightedJaccard, x, y) = Float32

@time pairwise(WeightedJaccard(instance2int, W), xun)
@time jwpairwise(xun, instance2int, W)

Flux.trainable(dist::WeightedJaccard) = (W = dist.W, )


# create weight model
model = WeightedJaccard(vcat(xun...))
ps = Flux.params(model)
opt = ADAM(0.005)

# create loss and compute it on a batch
# lossf(x, y) = ClusterLosses.loss(Triplet(), jwpairwise(x, instance2int, W), y)
lossf(x, y) = ClusterLosses.loss(Triplet(), pairwise(model, x), y)
lossf(batch...)

@epochs 5 begin
    Flux.train!(ps, mb_provider, opt) do x, y
        lossf(x, y)
    end
    @show lossf(batch...)
end


