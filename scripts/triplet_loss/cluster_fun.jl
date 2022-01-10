# cluster
using Distances, Clustering
"""
    cluster_data(M, E, yl, k; type="train")

Performs k-medoids clustering and hierarchical clustering on given
distance matrices `M, E`. Needs labels `yl` and number of clusters `k`.
Since k-medoids is not very stable, performs the clustering 10x and
chooses the best result.

Returns a dictionary of results. Given `type`, names in the dictionary
are either "test" or "train".
"""
function cluster_data(M, E, yl, k; type="train")
    y = gvma.encode(yl, unique(yl))

    cM = kmedoids(M, k)
    vm = randindex(cM, y)[1]
    for i in 1:9
        _cM = kmedoids(M, k)
        _vm = randindex(_cM, y)[1]
        if _vm > vm
            vm = _vm
            cM = _cM
        end
    end

    cE = kmedoids(E, k)
    ve = randindex(cE, y)[1]
    for i in 1:9
        _cE = kmedoids(M, k)
        _ve = randindex(_cE, y)[1]
        if _ve > ve
            ve = _ve
            cE = _cE
        end
    end

    hM = cutree(hclust(M, linkage=:average), k=k)
    hmr = randindex(hM, y)[1]
    hE = cutree(hclust(E, linkage=:average), k=k)
    her = randindex(hE, y)[1]

    if type == "train"
        train_results = Dict(
            :ri_medoids_train_dm => vm,
            :ri_medoids_train_emb => ve,
            :ri_hclust_train_dm => hmr,
            :ri_hclust_train_emb => her
        )
        return train_results
    else
        test_results = Dict(
            :ri_medoids_test_dm => vm,
            :ri_medoids_test_emb => ve,
            :ri_hclust_test_dm => hmr,
            :ri_hclust_test_emb => her
        )
        return test_results
    end
end

cluster_data(M, E, y; type="train") = cluster_data(M, E, y, length(unique(y)); type = type)