using DrWatson
@quickactivate

using gvma
using Flux, DataFrames, BSON
using PrettyTables
using Statistics
using Mill
using Distances

# load dataframe and models
df = collect_results(datadir("regularization"))
models = df[:, :full_model]
mill_models = map(m -> m[1], models)
seeds = df.seed
ratios = df.ratio
alphas = df.α

# load data
include(srcdir("init_strain.jl"))

# split data to train/test
# get latent encoding from mill model on test data
# calculate distance matrix on latent encoding
# run knn on the matrix

acc = Float64[]
for (seed, ratio, model) in map(i -> (seeds[i], ratios[i], mill_models[i]), 1:length(seeds))
    enc = model(X).data
    M = pairwise(Euclidean(), enc)

    (_, ytrain), (Xtest, ytest), dm = train_test_split(X, y, M; ratio = ratio, seed = seed)
    
    pred = dist_knn(1, dm, ytrain, ytest)[2]
    push!(acc, pred)
end

df.knn_accuracy = acc

_df = df[:, Not([:full_model, :path])]
g = groupby(_df, [:α, :ratio])
cdf = combine(
    g,
    [:train_acc, :test_acc, :knn_accuracy, :ri, :adj_ri, :silhouettes] .=> mean,
    renamecols=false
)

cdf_rounded = round.(cdf, digits=3)
pretty_table(cdf_rounded, tf = tf_markdown, nosubheader=true, crop=:none)


### kNN for small classes
acc_small = Float64[]
Xf, yf = cat(X, Xs), vcat(y, ys)
for (seed, ratio, model) in map(i -> (seeds[i], ratios[i], mill_models[i]), 1:length(seeds))
    enc = model(Xf).data
    M = pairwise(Euclidean(), enc)

    (ytrain, ytest), dm = train_test_split(yf, M; ratio = ratio, seed = seed)
    
    pred = dist_knn(1, dm, ytrain, ytest)[2]
    push!(acc_small, pred)
end

df.knn_accuracy_small = acc_small
d = df[:, Not([:full_model, :path])]


g = groupby(d, [:α, :ratio])
cdf = combine(
    g,
    [:train_acc, :test_acc, :knn_accuracy_small, :ri, :adj_ri, :silhouettes] .=> mean,
    renamecols=false
)

cdf_rounded = round.(cdf, digits=3)
pretty_table(cdf_rounded, tf = tf_markdown, nosubheader=true, crop=:none)