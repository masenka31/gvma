using DrWatson
@quickactivate

include(scriptsdir("toy", "src.jl"))
include(scriptsdir("toy", "cluster_fun.jl"))

### Dictionary of terms
seed = 1
n_classes = 10
n_normal = 5
n_bags = n_classes * 50
λ = 60
max_val = 1000
data, labels, code = generate_mill_data(n_classes, n_bags; λ = λ, seed = seed, max_val = max_val)
data, labels, code = generate_mill_unique(n_classes, n_bags; λ = λ, seed = seed, max_val = max_val)

# split data
(Xtrain, ytrain), (Xtest, ytest) = train_test_split(data, labels, collect(n_normal+1:n_classes); seed = seed, ratio = 0.5)
# unpack the Mill data
xun = unpack(Xtrain);
xun_ts = unpack(Xtest);
xun = unpack2int(Xtrain, max_val);
xun_ts = unpack2int(Xtest, max_val);
length(xun)
length(xun_ts)

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

@epochs 10 begin
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

code_onehot = map(x -> Flux.onehotbatch(x, 1:max_val), code)
map(c -> weight_model(c, W, instance2int), code)
map(c -> weight_model(c, W, instance2int), code_onehot)

######################################
### Scatter on UMAP representation ###
######################################

# look at the weighted jaccard distance
JDtr = jwpairwise(xun, instance2int, W)
emb_jd_tr = umap(JDtr, 2, metric=:precomputed)
scatter2(emb_jd_tr, zcolor=ytrain, color=:jet)

JD = jwpairwise(xun_ts, instance2int, W)
emb_jd = umap(JD, 2, metric=:precomputed)
yt = Int.(map(x -> any(x .== 41:50), ytest))
scplt = scatter2(emb_jd, color=yt)
# wsave("fig.svg", scplt)

# look at the simple Jaccard distance
jd = pairwise(SetJaccard(), xun)
emb_tr = umap(jd, 2; metric=:precomputed, n_neighbors=50)
scatter2(emb_tr, zcolor=ytrain, color=:Set1_4, markerstrokewidth=0)

jd2 = pairwise(SetJaccard(), xun_ts)
emb_ts = umap(jd2, 2; metric=:precomputed, n_neighbors=50)
scatter2(emb_ts, zcolor=ytest, color=:Set1_7, markerstrokewidth=0)

##################
### Evaluation ###
##################

# Jaccard distance as a metric
include(scriptsdir("toy", "jaccard_metric.jl"))
wm = WeightedJaccard(instance2int, W)
jwpairwise(xun) = pairwise(wm, xun)

# distance matrix for train and test data
@time M_train = jwpairwise(xun)
@time M_test = jwpairwise(xun_ts)

# umap embeddings for train and test data
emb_train = umap(M_train, 2, n_neighbors=10)
emb_test = umap(M_test, 2, n_neighbors=10)

# distance matrix for umap embeddings
E_train = pairwise(Euclidean(), emb_train)
E_test = pairwise(Euclidean(), emb_test)

# calculate clustering results
df_train = cluster_data(M_train, E_train, ytrain; type="train")
df_test = cluster_data(M_test, E_test, ytest; type="test")

####################################
### Linear regression on weights ###
####################################

uniques = unique(vcat(xun...))

c = countmap(vcat(xun...))
k = collect(keys(c))
v = collect(values(c))
kw = map(x -> weight_model(x, W, instance2int), k)
p1 = scatter(v, kw, ylims=(-0.05,1.05), label="", xlabel="# of occurences", ylabel="weight")
#wsave(plotsdir("weights_by_occurence.png"), p1)

c = countmap(vcat(xun_ts...))
k = collect(keys(c))
v = collect(values(c))
kw = map(x -> weight_model(x, W, instance2int), k)
is_in = map(x -> in(x, uniques), k)
p2 = scatter(v, kw, color=Int.(is_in), ylims=(0,1.05), label="", xlabel="# of occurences", ylabel="weight")
#wsave(plotsdir("weights_by_occurence_test.png"), p2)
