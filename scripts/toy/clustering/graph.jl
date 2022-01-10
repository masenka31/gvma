using Graphs
using SimpleWeightedGraphs

using LinearAlgebra

D = rand(20,20)
D = Symmetric(D)
map(i -> D[i, i] = 0, 1:20)
D

gtest = SimpleWeightedGraph(M_test)
edges = kruskal_mst(gtest)
g = SimpleWeightedGraph(755)

map(i -> add_edge!(g, edges[i]), 1:length(edges)-1)
g

findmax(map(e -> e.weight, edges))

distM = M_train
D = sort(distM,dims=2)
type = "kappa"
# and the indices
if type == "kappa"
    kappa = D[:,k]
    inds = kappa .< quantile(kappa,proportion)
elseif type == "gamma"
    gamma = mean(D[:,1:k],dims=2)[:]
    inds = gamma .< quantile(gamma,proportion)
# can we somehow do this with only distance matrix?
elseif type == "delta"
    d = D[:, 1:k]
    delta = LinearAlgebra.norm(X - Statistics.mean(d,dims=2))
    inds = delta .< quantile(delta,proportion)
end

# prepare necessarry matrices
N = sum(inds)
#newX = X[:,inds]
newM = distM[inds,inds]

gtest = SimpleWeightedGraph(M_train)
#gtest = SimpleWeightedGraph(newM)
edges = kruskal_mst(gtest)
g = SimpleWeightedGraph(nv(gtest))

map(i -> add_edge!(g, edges[i]), 1:length(edges)-length(unique(ytrain))+1)

c = connected_components(g)

new_labels = map(i -> repeat([i], length(c[i])), 1:5)

ls = length.(c)

map(ci -> in(1,ci), c)

new_labels = similar(ytrain)
un = unique(ytrain)
for i in 1:length(new_labels)
    b = map(ci -> in(i,ci), c)
    new_labels[i] = un[b][1]
end

"""
    mst_clustering(M, y; k = nothing)

Using distance matrix `M`, calculates the Minimum Spanning Tree (MST)
and cuts the longest edges (the number is controlled with the number
of unique values in labels `y`). If you want to override the number
of final clusters, simply add `k`. Also calculates the RandIndex
(a clustering measure) between new and predicted labels.

Source: https://www.sciencedirect.com/science/article/pii/S092523120500264X.
"""
function mst_clustering(M, y; k = nothing)
    # add k
    isnothing(k) ? k = length(unique(y)) : nothing

    # create a graph and MST
    gs = SimpleWeightedGraph(M)
    edges = kruskal_mst(gs)
    g = SimpleWeightedGraph(nv(gs))
    map(i -> add_edge!(g, edges[i]), 1:length(edges)-k+1)

    # get the components
    c = connected_components(g)

    # create new labels
    new_labels = similar(y)
    un = unique(y)
    for i in 1:length(new_labels)
        b = map(ci -> in(i,ci), c)
        new_labels[i] = un[b][1]
    end
    return new_labels, randindex(new_labels, y)
end

"""
    mst_clustering(M, y, proportion; k = nothing, type="kappa")

Using distance matrix `M`, calculates the Minimum Spanning Tree (MST)
and cuts the longest edges (the number is controlled with the number
of unique values in labels `y`). If you want to override the number
of final clusters, simply add `k`. Also calculates the RandIndex
(a clustering measure) between new and predicted labels.

Note that first, only `proportion` of data is used, outliers are cut
based on `type` metric from the source article. Outliers are then
connected to created clusters with kNN algorithm.

Source: https://www.sciencedirect.com/science/article/pii/S092523120500264X.
"""
function mst_clustering(M, y, proportion; k = nothing, type="kappa")
    # add k
    isnothing(k) ? k = length(unique(y)) : nothing

    # get rule for outliers cut
    D = sort(M,dims=2)
    if type == "kappa"
        kappa = D[:,k]
        inds = kappa .< quantile(kappa,proportion)
    elseif type == "gamma"
        gamma = mean(D[:,1:k],dims=2)[:]
        inds = gamma .< quantile(gamma,proportion)
    else
        md = median(D[:,1:k],dims=2)[:]
        inds = md .< quantile(md,proportion)
    end

    # cut outliers
    newM = M[inds,inds]

    # create a graph and MST
    gs = SimpleWeightedGraph(newM)
    edges = kruskal_mst(gs)
    g = SimpleWeightedGraph(nv(gs))
    map(i -> add_edge!(g, edges[i]), 1:length(edges)-length(unique(y))+1)

    # get the components
    c = connected_components(g)

    # create new labels
    newy = y[inds]
    new_labels = similar(newy)
    un = unique(newy)
    for i in 1:length(new_labels)
        b = map(ci -> in(i,ci), c)
        new_labels[i] = un[b][1]
    end

    # use kNN to get the missing labels
    v = collect(1:length(ytest))
    _inds = setdiff(v, v[inds])

    distance_matrix = M[_inds, inds]
    pred, _ = dist_knn(k, distance_matrix, new_labels, ytest[_inds])

    yfull = zeros(Int, length(y))
    yfull[inds] = new_labels
    yfull[_inds] = pred

    return yfull, randindex(yfull, y)
end

l, ri = mst_clustering(M_train, ytrain)
l, ri = mst_clustering(E_test, ytest)

l, ri = mst_clustering(M_test, ytest, 10, 0.8, type="med")
countmap(l)