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

ratio = 0.2
seed = 1

acc = Float16[]
for seed in 1:20
    (Xtrain, ytrain), (Xtest, ytest), distance_matrix = train_test_split(X, y, L; ratio=ratio, seed=seed)
    pred = dist_knn(1, distance_matrix, ytrain, ytest);
    push!(acc, pred[2])
end

println("Weighted Jaccard\nMean test accuracy of kNN on $ratio ratio for 20 seeds:\naccuracy = $(mean(acc))")

## previous kNN (normal Jaccard matrix)

using LinearAlgebra
Lprev = BSON.load(datadir("jaccard_matrix_strain.bson"))[:L]
L_full = Symmetric(Lprev)
ratio = 0.2

acc_prev = Float16[]
for seed in 1:20
    (Xtrain, ytrain), (Xtest, ytest), distance_matrix = train_test_split(X, y, L_full; ratio=ratio, seed=seed)
    pred = dist_knn(1, distance_matrix, ytrain, ytest);
    push!(acc_prev, pred[2])
end

println("Simple Jaccard\nMean test accuracy of kNN on $ratio ratio for 20 seeds:\naccuracy = $(mean(acc_prev))")