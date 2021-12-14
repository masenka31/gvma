using DrWatson
@quickactivate

include(scriptsdir("toy", "src.jl"))

############
### Data ###
############

# create data
seed = parse(Int64, ARGS[1])
n_classes = 7
n_bags = 700
data, labels, code = generate_mill_data(n_classes, n_bags; λ = 60, seed = seed)

# split data
# 4 classes known, 3 unknown
(Xtrain, ytrain), (Xtest, ytest) = train_test_split(data, labels, [5,6,7]; seed = seed)

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
max_train_time = 60*120 # 2 hours
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
emb_train = umap(M_train, 2)
emb_test = umap(M_test, 2)

# distance matrix for umap embeddings
E_train = pairwise(Euclidean(), emb_train)
E_test = pairwise(Euclidean(), emb_test)

### Clustering
# try k-medoids 10 times and choose the best one
# hclust is deterministic, compute only once

# training data

cm_train = kmedoids(M_train, 4)
vm = randindex(cm_train, ytrain)[1]
for i in 1:9
    _cm_train = kmedoids(M_train, 4)
    _vm = randindex(_cm_train, ytrain)[1]
    if _vm > vm
        vm = _vm
        cm_train = _cm_train
    end
end

ce_train = kmedoids(E_train, 4)
ve = randindex(ce_train, ytrain)[1]
for i in 1:9
    _ce_train = kmedoids(M_train, 4)
    _ve = randindex(_ce_train, ytrain)[1]
    if _ve > ve
        ve = _ve
        ce_train = _ce_train
    end
end

hm_train = cutree(hclust(M_train, linkage=:average), k=4)
hmr = randindex(hm_train, ytrain)[1]
he_train = cutree(hclust(E_train, linkage=:average), k=4)
her = randindex(he_train, ytrain)[1]

train_results = Dict(
    :ri_medoids_train_dm => vm,
    :ri_medoids_train_emb => ve,
    :ri_hclust_train_dm => hmr,
    :ri_hclust_train_emb => her
)

# test data

cm_test = kmedoids(M_test, 7)
vm = randindex(cm_test, ytest)[1]
for i in 1:9
    _cm_test = kmedoids(M_test, 7)
    _vm = randindex(_cm_test, ytest)[1]
    if _vm > vm
        vm = _vm
        cm_test = _cm_test
    end
end

ce_test = kmedoids(E_test, 7)
ve = randindex(ce_test, ytest)[1]
for i in 1:9
    _ce_test = kmedoids(M_test, 7)
    _ve = randindex(_ce_test, ytest)[1]
    if _ve > ve
        ve = _ve
        ce_test = _ce_test
    end
end

hm_test = cutree(hclust(M_test, linkage=:average), k=7)
hmr = randindex(hm_test, ytest)[1]
he_test = cutree(hclust(E_test, linkage=:average), k=7)
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
        :code => [code],
        :model => (instance2int, W),
        :opt => opt
    )
)

using DataFrames
df = DataFrame(full_results)

# save results
sname = "seed=$seed.bson"
wsave(datadir("toy_problem", sname))