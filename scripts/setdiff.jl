using DrWatson
@quickactivate
include(srcdir("init.jl"))
include(srcdir("flattening.jl"))
using LinearAlgebra
using UMAP
using BSON

files = datadir.("samples", dataset.samples .* ".json");
jsons = read_json.(files);
jsons_flatten = flatten_json.(jsons);

"""
    jaccard_distance(d1::Vector, d2::Vector)

Calculate the Jaccard distance between two vectors
as number of point in intersetion divided by number
of points in the union: d = # intersection / # union.
"""
function jaccard_distance(d1::Vector, d2::Vector)
    #dif = length(setdiff(d1,d2))
    int = length(intersect(d1,d2))
    un = length(union(d1,d2))
    return (un - int)/un
end
function jaccard_distance2(d1::Vector, d2::Vector)
    dif = length(setdiff(d1,d2))
    un = length(union(d1,d2))
    return dif/un
end

n = 8000
L = zeros(n,n) |> UpperTriangular
@time @threads for i in 1:n
    for j in i+1:n
        Ji = jsons_flatten[i]
        Jj = jsons_flatten[j]
        L[i,j] = jaccard_distance(Ji, Jj)
    end
end

L = BSON.load(datadir("jaccard_matrix.bson"))[:L]
L_full = Symmetric(L)

for nn in [1,2,3,5,10,20,50]
    embedding = umap(L_full, 2; metric = :precomputed, n_neighbors=nn)
    p = scatter2(embedding, zcolor=labels, color=:tab10, markerstrokewidth=0, opacity=0.7)
    wsave(plotsdir("jaccard_embedding_n=$nn.png"),p)
end

n = 8000
Ldiff = zeros(n,n) |> UpperTriangular
@time @threads for i in 1:n
    for j in i+1:n
        Ji = jsons_flatten[i]
        Jj = jsons_flatten[j]
        Ldiff[i,j] = jaccard_distance2(Ji, Jj)
    end
end

L_full_diff = Symmetric(Ldiff)

for nn in [1,2,3,5,10,20,50]
    embedding = umap(L_full_diff, 2; metric = :precomputed, n_neighbors=nn)
    p = scatter2(embedding, zcolor=labels, color=:tab10, markerstrokewidth=0, opacity=0.7)
    wsave(plotsdir("jaccard", "diff_jaccard_embedding_n=$nn.png"),p)
end