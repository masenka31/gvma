using DrWatson
@quickactivate

include(srcdir("init_strain.jl"))
include(srcdir("flattening.jl"))
using LinearAlgebra
using UMAP
using BSON
using Random

# load the Jaccard distance matrix
L = BSON.load(datadir("jaccard_matrix_strain.bson"))[:L]
L_full = Symmetric(L)

# clustering power
using Clustering

# use k-medoids to cluster the data
k = 10
c = kmedoids(L_full, k)
clabels = assignments(c)

yenc = gvma.encode(y, labelnames)

randindex(clabels, yenc)

# UMAP encoding with labels given clustering
using UMAP

for nn in 1:10
    emb = umap(L_full, n_neighbors=nn, metric=:precomputed)

    p1 = scatter2(emb; zcolor=clabels,
                        color=:tab10, markerstrokewidth=0, label="",
                        title="clustering labels");
    wsave(plotsdir("clustering", "n=$(nn)_kmedoids.png"), p1)

    p2 = scatter2(emb; zcolor=yenc,
                        color=:tab10, markerstrokewidth=0, label="",
                        title="true labels");
    wsave(plotsdir("clustering", "n=$(nn)_true.png"), p2)
end

# how to best match true labels with cluster labels?
matches = Tuple[]
counts = zeros(Int, 10,10)
for j in 1:10
    cbit = clabels .== j
    mx = 0
    chosen_class = 0
    mv = map(i -> sum(cbit .== (yenc .== i) .== 1), 1:10)
    counts[j,:] .= mv
    @show j, findmax(mv)
    push!(matches, (j, findmax(mv)[2]))
end

df = DataFrame(counts, map(i -> "$i", 1:10))

matched_labels = map(x -> x[2], matches)
function match_c(i, matches)
    ix = findall(x -> x[1] == i, matches)[1]
    matches[ix][2]
end

clb = similar(clabels)
for i in 1:8000
    prev = clabels[i]
    new = match_c(prev, matches)
    clb[i] = new
end


pix = map(i -> partialsortperm(counts[i,:], 1:10, rev=true), 1:10)
pcounts = map(i -> counts[i, pix[i]], 1:10)
counts

matches2 = vcat(
    (1,4),
    (2,1),
    matches[3:end]
)

clb_matched = similar(clabels)
for i in 1:8000
    prev = clabels[i]
    new = match_c(prev, matches2)
    clb_matched[i] = new
end

sum(clb_matched .== gvma.encode(y, labelnames)) / 8000

for nn in 1:10
    emb = umap(L_full, n_neighbors=nn, metric=:precomputed)
    
    p1 = scatter2(emb; zcolor=clabels,
                        color=:tab10, markerstrokewidth=0, label="",
                        title="clustering labels");
    wsave(plotsdir("clustering", "n=$(nn)_kmedoids.png"), p1)

    p2 = scatter2(emb; zcolor=yenc,
                        color=:tab10, markerstrokewidth=0, label="",
                        title="true labels");
    wsave(plotsdir("clustering", "n=$(nn)_true.png"), p2)

    p3 = scatter2(emb; zcolor=clb_matched,
                        color=:tab10, markerstrokewidth=0, label="",
                        title="clustering labels");
    wsave(plotsdir("clustering", "n=$(nn)_kmedoids_matched.png"), p3)
end