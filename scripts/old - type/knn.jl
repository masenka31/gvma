using DrWatson
@quickactivate
include(srcdir("init.jl"))
include(srcdir("flattening.jl"))
using LinearAlgebra
using UMAP
using BSON

# if you want to compute the distance matrix,
# reffer to script setdiff.jl

L = BSON.load(datadir("jaccard_matrix.bson"))[:L]
L_full = Symmetric(L)

