using DrWatson
@quickactivate

include(srcdir("init_strain.jl"))
include(srcdir("flattening.jl"))
using LinearAlgebra
using UMAP
using BSON
using Random

using gvma: jaccard_distance, flatten_json

files = datadir.("samples_strain", dataset.samples .* ".json");
jsons = gvma.read_json.(files);
jsons_flatten = flatten_json.(jsons);

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

################################################
### Simple kNN using Jaccard distance matrix ###
################################################

# the new version of flatten_json
L = BSON.load(datadir("jaccard_matrix_new_flatten.bson"))[:L]

mean_acc = Float32[]
for ratio in [0.01, 0.05, 0.1, 0.2, 0.5, 0.8]
    acc = Float32[]
    for seed in 1:20
        (ytrain, ytest), distance_matrix = train_test_split(y, L; ratio=ratio, seed = seed)
        pred = dist_knn(1, distance_matrix, ytrain, ytest);
        push!(acc, Float32(pred[2]))
    end
    push!(mean_acc, mean(acc))
end

#############################################
### Clustering on Jaccard distance matrix ###
#############################################

using Clustering
using DataFrames
using gvma: encode
ynum = encode(y, labelnames)

c = kmedoids(L, 10)
clabels = assignments(c)
ri, adj_ri = randindex(c, ynum)[1:2]
counts(c, ynum)

# try how well is missing class separated
# for all classes
include(srcdir("confusion_matrix.jl"))

dfs = map(c -> evaluate_jaccard(y, c, L; k = 12), String.(labelnames));
df = vcat(dfs...)

###########################################
### Jaccard distance for small clusters ###
###########################################

# how to not calculate what we already know?
using BSON
L = BSON.load(datadir("jaccard_matrix_new_flatten.bson"))[:L]

include(srcdir("init_strain.jl"))
dataset = Dataset(datadir("samples_strain.csv"), datadir("schema.bson"), datadir("samples_strain"))
dataset_s = Dataset(datadir("samples_small.csv"), datadir("schema.bson"), datadir("samples_small"))

files = datadir.("samples_strain", dataset.samples .* ".json");
jsons = gvma.read_json.(files);
fps_old = flatten_json.(jsons);

files = datadir.("samples_small", dataset_s.samples .* ".json");
jsons = gvma.read_json.(files);
fps_new = flatten_json.(jsons);

fps = vcat(fps_new, fps_old)

n = length(y)
m = length(y) + length(ys)
new_L_init = zeros(Float64, m, m)
new_L_init[1:8000, 1:8000] .= L
using LinearAlgebra
newL = UpperTriangular(new_L_init)

using gvma: jaccard_distance

Threads.@threads for i in 1:m
    for j in i+1:m
        v = jaccard_distance(fps[i], fps[j])
        newL[i, j] = v
    end
end
L = collect(Symmetric(newL))

safesave(datadir("jaccard_matrix_all.bson"), Dict(:L => newL))
L = BSON.load(datadir("jaccard_matrix_all.bson"))[:L]

Xf, yf = cat(X, Xs), vcat(y, ys)

acc = zeros(Float64, 15, 10)
for seed in 1:10
    (ytrain, ytest), dm = train_test_split(yf, L; ratio = 0.2, seed = seed)
    a = map(k -> dist_knn(k, dm, ytrain, ytest)[2], 1:15)
    acc[:, seed] = a
end

acc_mean = mean(acc, dims=2)
df = DataFrame(:k => collect(1:15), :accuracy => acc_mean[:, 1])

# binary knn classification - large vs small clusters
ybinary = .!map(i -> any(i .== labelnames), yf)

acc_binary = zeros(Float64, 15, 10)
for seed in 1:10
    (ytrain, ytest), dm = train_test_split(ybinary |> Vector, L; ratio = 0.2, seed = seed)
    a = map(k -> dist_knn(k, dm, ytrain, ytest)[2], 1:15)
    acc_binary[:, seed] = a
end

acc_mean_binary = mean(acc_binary, dims=2)
df.binary_accuracy = acc_mean_binary[:, 1]

pretty_table(df, tf = tf_markdown, nosubheader=true, crop=:none)