using DrWatson
@quickactivate

include(srcdir("init_strain.jl"))
include(srcdir("flattening.jl"))
using LinearAlgebra
using UMAP
using BSON
using Random

files = datadir.("samples_strain", dataset.samples .* ".json");
jsons = gvma.read_json.(files);
jsons_flatten = flatten_json.(jsons);

"""
    jaccard_distance(d1::Vector, d2::Vector)

Calculate the Jaccard distance between two vectors
as number of points in intersetion divided by number
of points in the union: d = # intersection / # union.
"""
function jaccard_distance(d1::Vector, d2::Vector)
    #dif = length(setdiff(d1,d2))
    int = length(intersect(d1,d2))
    un = length(union(d1,d2))
    return (un - int)/un
end

# calculate the Jaccard distance for each pair
# the matrix is symmetric and with zeros on the diagonal
using ThreadTools: @threads
n = 8000
L = zeros(n,n) |> UpperTriangular
@time @threads for i in 1:8000-1
    v = map(x -> jaccard_distance(x, jsons_flatten[i]), jsons_flatten[i+1:end])
    L[i,i+1:end] .= v
end

# save the matrix, if calculated for the first time
safesave(datadir("jaccard_matrix_strain.bson"), Dict(:L => L))
# load the matrix, if already calculated (only UpperDiagonal matrix)
L = BSON.load(datadir("jaccard_matrix_strain.bson"))[:L]
# create full matrix
L_full = Symmetric(L)

# calculate the UMAP embedding and save the plots
for nn in [1,2,3,5,10,20,50]
    embedding = umap(L_full, 2; metric = :precomputed, n_neighbors=nn)
    p = scatter2(embedding, zcolor=labels, color=:tab10, markerstrokewidth=0, opacity=0.7)
    wsave(plotsdir("strain", "jaccard_embedding_n=$nn.png"),p)
end

#############################################
### kNN using precomputed distance matrix ###
#############################################

for seed in 1:5
    plt = plot()
    for ratio in [0.01, 0.05, 0.1, 0.2, 0.5]
        (Xtrain, ytrain), (Xtest, ytest), distance_matrix = train_test_split(X, y; ratio=ratio, seed=seed)
        acc = Float16[]
        for kn in 1:20
            pred = dist_knn(kn, distance_matrix, ytrain, ytest);
            push!(acc, pred[2])
        end

        plt = plot!(1:20, acc, marker=:circle,  ylims=(0,1), legend=:bottomleft,
                    xlabel="k neighbors", ylabel="accuracy", label="# train = $(8000*ratio)")
    end
    wsave(plotsdir("strain_knn", "k-accuracy_seed=$seed.png"), plt)
end

for ratio in [0.01, 0.05, 0.1, 0.2, 0.5]
    plt = plot()
    for seed in 1:5
        (Xtrain, ytrain), (Xtest, ytest), distance_matrix = train_test_split(X, y; ratio=ratio, seed=seed)
        acc = Float16[]
        for kn in 1:20
            pred = dist_knn(kn, distance_matrix, ytrain, ytest);
            push!(acc, pred[2])
        end

        plt = plot!(1:20, acc, marker=:circle, ylims=(0,1), legend=:bottomleft,
                        xlabel="k neighbors", ylabel="accuracy", label="seed = $seed")
    end
    wsave(plotsdir("strain_knn", "k-accuracy_ntr=$(8000*ratio)_01.png"), plt)
end