using DrWatson
@quickactivate

include(scriptsdir("toy", "src.jl"))
ENV["GKSwstype"] = "100"

############
### Data ###
############

# create data
seed = parse(Int64, ARGS[1])
n_classes = parse(Int64, ARGS[2])
n_normal = parse(Int64, ARGS[3])
λ = parse(Int64, ARGS[4])
n_bags = 1000
data, labels, code = generate_mill_data(n_classes, n_bags; λ = λ, seed = seed)

# split data
# 4 classes known, 3 unknown
(Xtrain, ytrain), (Xtest, ytest) = train_test_split(data, labels, collect(n_normal+1:n_classes); seed = seed, ratio=0.5)

# unpack the Mill data
xun = unpack(Xtrain);
xun_ts = unpack(Xtest);

# create the dictionary
instance_dict = unique(vcat(xun...))
instance2int = Dict(reverse.(enumerate(instance_dict)))

# prepare minibatches
batchsize = 32
function prepare_minibatch()
    ix = rand(1:length(ytrain), batchsize)
    xun[ix], ytrain[ix]
end
mb_provider = IterTools.repeatedly(prepare_minibatch, 10)
batch = prepare_minibatch();

########################
### Model & Training ###
########################

# create weight model
W = ones(Float32, length(instance2int))
ps = Flux.params(W)
opt = ADAM(0.005)

# create loss and compute it on a batch
lossf(x, y) = ClusterLosses.loss(Triplet(), jwpairwise(x, instance2int, W), y)
lossf(batch...)

# training
max_train_time = 60*10 # 30 minutes
start_time = time()

while time() - start_time < max_train_time
    Flux.train!(ps, mb_provider, opt) do x, y
        lossf(x, y)
    end
    @show lossf(batch...)
end

##################
### Evaluation ###
##################

# distance matrix for train and test data
jwpairwise(x) = jwpairwise(x, instance2int, W)
M_train = jwpairwise(xun)
M_test = jwpairwise(xun_ts)

# umap embeddings for train and test data
emb_train = umap(M_train, 2, metric=:precomputed)
emb_test = umap(M_test, 2, metric=:precomputed)

# distance matrix for umap embeddings
E_train = pairwise(Euclidean(), emb_train)
E_test = pairwise(Euclidean(), emb_test)

### Clustering
# try k-medoids 10 times and choose the best one
# hclust is deterministic, compute only once

# training data

k_train = length(unique(ytrain))

cm_train = kmedoids(M_train, k_train)
vm = randindex(cm_train, ytrain)[1]
for i in 1:9
    _cm_train = kmedoids(M_train, k_train)
    _vm = randindex(_cm_train, ytrain)[1]
    if _vm > vm
        global vm = _vm
        global cm_train = _cm_train
    end
end

ce_train = kmedoids(E_train, k_train)
ve = randindex(ce_train, ytrain)[1]
for i in 1:9
    _ce_train = kmedoids(M_train, k_train)
    _ve = randindex(_ce_train, ytrain)[1]
    if _ve > ve
        global ve = _ve
        global ce_train = _ce_train
    end
end

hm_train = cutree(hclust(M_train, linkage=:average), k=k_train)
hmr = randindex(hm_train, ytrain)[1]
he_train = cutree(hclust(E_train, linkage=:average), k=k_train)
her = randindex(he_train, ytrain)[1]

train_results = Dict(
    :ri_medoids_train_dm => vm,
    :ri_medoids_train_emb => ve,
    :ri_hclust_train_dm => hmr,
    :ri_hclust_train_emb => her
)

# test data

k_test = length(unique(ytest))

cm_test = kmedoids(M_test, k_test)
vm = randindex(cm_test, ytest)[1]
for i in 1:9
    _cm_test = kmedoids(M_test, k_test)
    _vm = randindex(_cm_test, ytest)[1]
    if _vm > vm
        global vm = _vm
        global cm_test = _cm_test
    end
end

ce_test = kmedoids(E_test, k_test)
ve = randindex(ce_test, ytest)[1]
for i in 1:9
    _ce_test = kmedoids(M_test, k_test)
    _ve = randindex(_ce_test, ytest)[1]
    if _ve > ve
        global  ve = _ve
        global ce_test = _ce_test
    end
end

hm_test = cutree(hclust(M_test, linkage=:average), k=k_test)
hmr = randindex(hm_test, ytest)[1]
he_test = cutree(hclust(E_test, linkage=:average), k=k_test)
her = randindex(he_test, ytest)[1]

test_results = Dict(
    :ri_medoids_test_dm => vm,
    :ri_medoids_test_emb => ve,
    :ri_hclust_test_dm => hmr,
    :ri_hclust_test_emb => her
)

full_results = merge(
    train_results,
    test_results,
    Dict(
        :seed => seed,
        :n_classes => n_classes,
        :n_normal => n_normal,
        :code => [code],
        :model => (instance2int, W),
        :opt => opt
    )
)

using DataFrames
df = DataFrame(full_results)

# save results
sname = "seed=$seed.bson"
wsave(datadir("toy_problem", sname), full_results)

scatter2(emb_train, zcolor=ytrain, color=:Set1_3)
savefig("3class_embedding.png")

scatter2(emb_test, zcolor=ytest, color=:Set1_8)
savefig("8class_embedding.png")

code_onehot = map(x -> Flux.onehotbatch(x, 1:100), code)
c = countmap(unique(vcat(xun...)))
minimum(map(x -> weight_model(x, W, instance2int), collect(keys(c))))
maximum(map(x -> weight_model(x, W, instance2int), collect(keys(c))))
map(x -> weight_model(x, W, instance2int), code_onehot)

