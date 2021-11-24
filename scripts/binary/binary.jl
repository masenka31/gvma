using DrWatson
@quickactivate

# load data and necessary packages
include(srcdir("init_strain.jl"))
include(srcdir("confusion_matrix.jl"))
include(scriptsdir("binary", "evaluate_model.jl"))
using Distances, Clustering
using gvma: encode

# load the Jaccard matrix
using BSON
L = BSON.load(datadir("jaccard_matrix_new_flatten.bson"))[:L]

# load parameters, model, and split data to the original split
d = BSON.load(datadir("models/missing/model_missing_class=HarHar_ratio=0.2_seed=1_α=0.bson"))
@unpack ratio, seed, α, missing_class, model = d
(Xtrain, ytrain), (Xtest, ytest) = train_test_split(X, y, missing_class; ratio=ratio, seed=seed)
full_model = model
mill_model = model[1]

# get the latent encoding
enc_ts = mill_model(Xtest).data
# calculate pairwise Euclidean distance
M_test = pairwise(Euclidean(), enc_ts)

# perform k-medoids clustering
k = 15
c = kmedoids(M_test, k)
clabels = assignments(c)
yenc_binary = encode(ytest, missing_class)

# get the counts for clusters and labels
cs = counts(clabels, yenc_binary)

# get "purity" percentages of clusters with missing class
s = sum(cs, dims=2)
miss_cluster_ix = findall(x -> x != 0, cs[:, 2])
percentages = cs[miss_cluster_ix, 2] ./ s[miss_cluster_ix]

# get confusion matrix for binary classification
new_bit = cs[:,1] .< cs[:,2]
TN, FN = Tuple(sum(cs[.!new_bit, :], dims=1)) # gives TN, FN
FP, TP = Tuple(sum(cs[new_bit, :], dims=1))   # gives FP, TP
CM = ConfusionMatrix(TP, FP, FN, TN)

# create a report
report(CM)

###########################################
### For multiple models, datasets, etc. ###
###########################################

names = readdir(datadir("models", "missing"))

filtered_names = filter(name -> occursin("HarHar", name), names)

dfs = DataFrame[]
for n in filtered_names
    df = evaluate_model(X, y, L, n)
    push!(dfs, df)
end
results = vcat(dfs...)
# results = results[1:end-1, :]

g = groupby(results[:, Not(:percentages)], [:ratio, :α])
c = DataFrames.names(g[1])[1:end-4]
cdf = combine(g, c .=> mean, renamecols=false)
sort!(cdf, [:ratio, :α])

# parallelize
using DataFrames
results = repeat([DataFrame()], 10)

using Base.Threads: @threads
@show Threads.nthreads();

files = readdir(datadir("models", "missing"))

@threads for i in 1:10
    filtered_names = filter(name -> occursin(labelnames[i], name), files)
    dfs = DataFrame[]
    for n in filtered_names
        df = evaluate_model(X, y, L, n)
        push!(dfs, df)
    end
    res = vcat(dfs...)
    results[i] = res
end

function nan_mean(x)
    b = isnan.(x)
    return mean(x[.!b])
end

# having results df, let's look at them more thoroughly
i = 1
function comb(results, i)
    df = results[i]
    g = groupby(df[:, Not(:percentages)], [:ratio, :α])
    nm = names(g[1])[1:end-4]
    cdf = combine(g, nm .=> nan_mean, renamecols=false)
    cdf = round.(cdf, digits=3)
    return hcat(cdf, DataFrame(:class => repeat([df[1,:missing_class]], size(cdf,2))))
end

combined_results = map(i -> comb(results, i), 1:10)
comb_mater = vcat(combined_results...)

f = filter(:α => a -> a == 0.1f0, filter(:ratio => r -> r == 0.2, comb_mater))
pretty_table(f, tf = tf_markdown, nosubheader=true)

f = filter(:α => a -> a == 0.0f0, filter(:ratio => r -> r == 0.2, comb_mater))
pretty_table(f, tf = tf_markdown, nosubheader=true)

f = filter(:α => a -> a == 1.0f0, filter(:ratio => r -> r == 0.2, comb_mater))
pretty_table(f, tf = tf_markdown, nosubheader=true)

map(i -> pretty_table(comb(results, i), crop=:none, nosubheader=true), 1:10)
map(i -> pretty_table(comb(results, i), tf = tf_markdown, crop=:none, nosubheader=true), 1:10)