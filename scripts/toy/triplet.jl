using DrWatson
@quickactivate

include(scriptsdir("toy", "src.jl"))

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

############################
### Discriminative model ###
############################

# model and training
model = reflectinmodel(Xtrain[1])
full_model = Chain(model, Mill.data, Dense(10,length(unique(ytrain))), softmax)

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
end

# train data
# full output
enc = model(Xtrain).data
scatter2(enc, color=ytrain)
emb = umap(enc, 2)
scatter2(emb, color=ytrain)

M = pairwise(Euclidean(), enc)
E = pairwise(Euclidean(), emb)
cluster_data(M, E, ytrain)


# test data
enc = model(Xtest).data
scatter2(enc, zcolor=ytest)

emb = umap(enc, 2)
scatter2(emb, zcolor=ytest, color=:jet)

M = pairwise(Euclidean(), enc)
E = pairwise(Euclidean(), emb)
cluster_data(M, E, ytest, type="test")

##############################
### Triplet loss embedding ###
##############################

# model and training
model = reflectinmodel(Xtrain[1])
full_model = Chain(model, Mill.data, Dense(10,10))

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

lossf(x, y) = ClusterLosses.loss(Triplet(3f0), SqEuclidean(), full_model(x), y)
ps = Flux.params(full_model)
opt = ADAM()
lossf(batch...)

# train for some max train time
max_train_time = 60*3
start_time = time()
while time() - start_time < max_train_time
    Flux.train!(ps, mb_provider, opt) do x, y
        lossf(x, y)
    end
    @show lossf(batch...)
end

# train data
enc = model(Xtrain).data
scatter2(enc, zcolor=ytrain, color=:Set1_3, title="Train")
savefig("train.png")

emb = umap(enc, 2)
scatter2(emb, color=ytrain)

M = pairwise(Euclidean(), enc)
E = pairwise(Euclidean(), emb)
cluster_data(M, E, ytrain)

# test data
enc = model(Xtest).data
scatter2(enc, 2, 1, zcolor=ytest, color=:jet, title="Test")
emb = umap(enc, 2)
scatter2(emb, zcolor=ytest, color=:jet)

M = pairwise(Euclidean(), enc)
E = pairwise(Euclidean(), emb)
cluster_data(M, E, ytest, type="test")

enc = full_model(Xtest)
scatter2(enc, 5, 4, zcolor=ytest, color=:jet, title="Test")
emb = umap(pairwise(Euclidean(), enc), 2, n_neighbors=15)
scatter2(emb, zcolor=ytest, color=:jet)


M = pairwise(Euclidean(), enc)
E = pairwise(Euclidean(), emb)
cluster_data(M, E, ytest, type="test")
