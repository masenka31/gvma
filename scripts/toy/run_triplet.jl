using DrWatson
@quickactivate

include(scriptsdir("toy", "src.jl"))
include(scriptsdir("toy", "cluster_fun.jl"))

############
### Data ###
############

# create data
seed = parse(Int64, ARGS[1])
unq = parse(Bool, ARGS[2])
n_classes = parse(Int64, ARGS[3])
λ = parse(Int64, ARGS[4])
activation = ARGS[5]

n_normal = n_classes - 10
n_bags = n_classes * 50
max_val = 10000

# whether to use all unique code or not
if unq
    data, labels, code = generate_mill_unique(n_classes, n_bags; λ = λ, seed = seed, max_val = max_val)
else
    data, labels, code = generate_mill_data(n_classes, n_bags; λ = λ, seed = seed, max_val = max_val)
end

# split data
(Xtrain, ytrain), (Xtest, ytest) = train_test_split(data, labels, collect(n_normal+1:n_classes); seed = seed, ratio = 0.5)

########################
### Model & Training ###
########################

# create model
act_fun = eval(Symbol(activation))
model = reflectinmodel(
    Xtrain[1],
    d -> Dense(d, 10, act_fun),
    d -> meanmax_aggregation(d)
)
full_model = Chain(model, Mill.data, Dense(10,10))

# loss function
lossf(x, y) = ClusterLosses.loss(Triplet(), SqEuclidean(), full_model(x), y)

# create minibatch function
batchsize = 64
function minibatch()
    ix = sample(1:nobs(Xtrain), batchsize)
    return (Xtrain[ix], ytrain[ix])
end
batch = minibatch()

# minibatch iterator
using IterTools
mb_provider = IterTools.repeatedly(minibatch, 10)

# parameters and optimiser
ps = Flux.params(full_model)
opt = ADAM()

# train for some max train time
max_train_time = 60*60*n_classes/10 # based on number of classes
tr_loss = Inf
best_model = deepcopy(full_model)
patience = 100
_patience = 0

start_time = time()
while time() - start_time < max_train_time
    Flux.train!(ps, mb_provider, opt) do x, y
        lossf(x, y)
    end
    l = lossf(Xtrain, ytrain) # does this take too long?
    if l < tr_loss
        global _patience = 0
        global tr_loss = l
        @show tr_loss
        global best_model = deepcopy(full_model)
    else
        global _patience += 1
    end
    if _patience > patience
        @info "Model has not improved for $_patience epochs."
        break
    end
end
@info "Training finished."
full_model = deepcopy(best_model)

##################
### Evaluation ###
##################

# Mill embedding
enc_train = full_model(Xtrain)
enc_test = full_model(Xtest)
M_train = pairwise(SqEuclidean(), enc_train)
M_test = pairwise(SqEuclidean(), enc_test)

# umap embeddings for train and test data
emb_train = umap(enc_train, 2)
emb_test = umap(enc_test, 2)

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
        :model => full_model,
        :opt => opt,
        :unq => unq,
        :n_classes => n_classes,
        :n_normal => n_normal,
        :λ => λ,
        :activation => activation
    )
)

using DataFrames
df = DataFrame(full_results)

# save results
sname = savename(
    "triplet",
    Dict(
        :seed => seed,
        :unique => unq,
        :n_classes => n_classes,
        :n_normal => n_normal,
        :λ => λ,
        :activation => activation
    ),
    "bson"
)
safesave(datadir("toy_max=$max_val", "triplet", sname), full_results)