using DrWatson
@quickactivate

include(scriptsdir("toy", "src.jl"))
include(scriptsdir("toy", "cluster_fun.jl"))
include(scriptsdir("toy", "jaccard_metric.jl"))

############
### Data ###
############

# create data
seed = parse(Int64, ARGS[1])
unq = parse(Bool, ARGS[2])
n_classes = parse(Int64, ARGS[3])
λ = parse(Int64, ARGS[4])

n_normal = n_classes - 10
n_bags = n_classes * 50
max_val = 1000

# whether to use all unique code or not
if unq
    data, labels, code = generate_mill_unique(n_classes, n_bags; λ = λ, seed = seed, max_val = max_val)
else
    data, labels, code = generate_mill_data(n_classes, n_bags; λ = λ, seed = seed, max_val = max_val)
end

# split data
(Xtrain, ytrain), (Xtest, ytest) = train_test_split(data, labels, collect(n_normal+1:n_classes); seed = seed, ratio = 0.5)

# unpack the Mill data
xun = unpack2int(Xtrain, max_val);
xun_ts = unpack2int(Xtest, max_val);


##################
### Evaluation ###
##################

# distance matrix for train and test data
@time M_train = pairwise(SetJaccard(), xun)
@time M_test = pairwise(SetJaccard(), xun_ts)

# umap embeddings for train and test data
emb_train = umap(M_train, 2)
emb_test = umap(M_test, 2)

# distance matrix for umap embeddings
E_train = pairwise(Euclidean(), emb_train)
E_test = pairwise(Euclidean(), emb_test)

##################
### Clustering ###
##################

# k-medoids and hierarchical clustering with linkage=:average
# training data
df_train = cluster_data(M_train, E_train, ytrain; type="train")
# test data
df_test = cluster_data(M_test, E_test, ytest; type="test")
@info "Results calculated."

full_results = merge(
    df_train,
    df_test,
    Dict(
        :seed => seed,
        :code => [code],
        :unq => unq,
        :n_classes => n_classes,
        :n_normal => n_normal,
        :λ => λ
    )
)

using DataFrames
df = DataFrame(full_results)

# save results
sname = savename(
    "jaccard",
    Dict(
        :seed => seed,
        :unique => unq,
        :n_classes => n_classes,
        :n_normal => n_normal,
        :λ => λ
    ),
    "bson"
)
safesave(datadir("toy_max=$max_val", "jaccard", sname), full_results)