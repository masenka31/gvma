using DrWatson
@quickactivate

include(srcdir("init_strain.jl"))

### kNN ###
ratio = 0.2
seed = 1

using BSON
L = BSON.load(datadir("weigted_matrix.bson"))[:L]

acc = Float16[]
for seed in 1:20
    (Xtrain, ytrain), (Xtest, ytest), distance_matrix = train_test_split(X, y, L; ratio=ratio, seed=seed)
    pred = dist_knn(1, distance_matrix, ytrain, ytest)[2];
    push!(acc, pred)
end

using LinearAlgebra
Lprev = BSON.load(datadir("jaccard_matrix_strain.bson"))[:L]
L_full = Symmetric(Lprev)
ratio = 0.2

acc_prev = Float16[]
for seed in 1:20
    (Xtrain, ytrain), (Xtest, ytest), distance_matrix = train_test_split(X, y, L_full; ratio=ratio, seed=seed)
    pred = dist_knn(1, distance_matrix, ytrain, ytest)[2];
    push!(acc_prev, pred)
end

println("Simple Jaccard\nMean test accuracy of kNN on $ratio ratio for 20 seeds:\naccuracy = $(mean(acc_prev))")

println("Weighted Jaccard\nMean test accuracy of kNN on $ratio ratio for 20 seeds:\naccuracy = $(mean(acc))")
