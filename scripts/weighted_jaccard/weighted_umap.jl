using DrWatson
@quickactivate

using gvma
using BSON, UMAP
using Plots

include(srcdir("init_strain.jl"))
labels = gvma.encode(y, labelnames)

L = BSON.load(scriptsdir("weighted_matrix.bson"))[:L]

emb = umap(L, 2; metric=:precomputed, n_neighbors=3)
scatter2(emb, zcolor=labels, color=:tab10)

###########
### kNN ###
###########

