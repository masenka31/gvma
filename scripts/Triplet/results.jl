using DrWatson
@quickactivate

include(scriptsdir("toy", "src.jl"))
include(scriptsdir("toy", "cluster_fun.jl"))
include(scriptsdir("toy", "jaccard_metric.jl"))

using DataFrames
using DataFrames: groupby
using PrettyTables

using Plots, StatsPlots
ENV["GKSwstype"] = "100"

# choose which folder to use
full = true
clean = true

# collect results over 5 seeds
df = collect_results(datadir("triplet_embedding/full=$full/clean=$clean"), subfolders=true)
foreach(x -> println(x), names(df))

# get some dataframe names
dat_types = ["known", "unknown", "full"]    # choose which data
type_types = ["emb", "dm"]                  # choose if to compare embedding or distance matrix results

dat = dat_types[2]
type = type_types[2]

metrics = ["ri_hclust_", "ri_medoids_", "slt_hclust_", "slt_medoids_"] .* dat .* "_" .* type
params = ["odim", "nlayers", "batchsize", "agg", "mdim", "k_known", "margin", "seed"] # not activation because it is only relu

df2 = df[:, vcat(params, metrics)]
par = :seed
gdf = groupby(df2, par)
cdf = sort(combine(gdf, metrics .=> mean, renamecols=false), par)

sort!(df2, metrics[1], rev=true)
