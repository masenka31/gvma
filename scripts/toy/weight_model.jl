using DrWatson
@quickactivate

include(scriptsdir("toy", "src.jl"))

### Dictionary of terms
seed = 1
n_classes = 5
n_normal = 3
n_bags = 300
λ = 50
data, labels, code = generate_mill_data(n_classes, n_bags; λ = λ, seed = seed)

# split data
(Xtrain, ytrain), (Xtest, ytest) = train_test_split(data, labels, collect(n_normal+1:n_classes); seed = seed, ratio=0.5)

m = Chain(reflectinmodel(Xtrain[1].data), Mill.data, Dense(10, 1, σ))
m(Xtrain[1].data)