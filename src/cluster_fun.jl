using Distances, Clustering

"""
    cluster_data(M, E, yl, k; type="train")

Performs k-medoids clustering and hierarchical clustering on given
distance matrices `M, E`. Matrix `M` is a distance matrix (e.g. Jaccard),
`E` is the Euclidean distance calculated on UMAP embedding created on
matrix `M`.

Needs labels `yl` and number of clusters `k`.

Since k-medoids is not very stable, performs the clustering 10x and
chooses the best result given the best silhouettes value.

Returns a dictionary of results. Given `type`, names in the dictionary
are either "test" or "train".
"""
function cluster_data(M, E, yl, k; type="train")
    y = gvma.encode(yl, unique(yl))

    # try k-medoids 10 times, choose the best using !silhouettes!
    cM = kmedoids(M, k)
    vm = randindex(cM, y)[1]
    slt_m = mean(silhouettes(cM, M))
    for i in 1:9
        _cM = kmedoids(M, k)
        _vm = randindex(_cM, y)[1]
        _slt_m = mean(silhouettes(_cM, M))
        if _slt_m > slt_m
            vm = _vm
            cM = _cM
            slt_m = _slt_m
        end
    end

    cE = kmedoids(E, k)
    ve = randindex(cE, y)[1]
    slt_e = mean(silhouettes(cE, E))
    for i in 1:9
        _cE = kmedoids(M, k)
        _ve = randindex(_cE, y)[1]
        _slt_e = mean(silhouettes(_cE, E))
        if _slt_e > slt_e
            ve = _ve
            cE = _cE
            slt_e = _slt_e
        end
    end

    hM = cutree(hclust(M, linkage=:average), k=k)
    hmr = randindex(hM, y)[1]
    hmslt = mean(silhouettes(hM, M))
    hE = cutree(hclust(E, linkage=:average), k=k)
    her = randindex(hE, y)[1]
    heslt = mean(silhouettes(hE, E))

    k = [
        Symbol("ri_medoids_"*type*"_dm"),
        Symbol("ri_medoids_"*type*"_emb"),
        Symbol("ri_hclust_"*type*"_dm"),
        Symbol("ri_hclust_"*type*"_emb"),
        Symbol("slt_medoids_"*type*"_dm"),
        Symbol("slt_medoids_"*type*"_emb"),
        Symbol("slt_hclust_"*type*"_dm"),
        Symbol("slt_hclust_"*type*"_emb")
    ]
    v = [vm, ve, hmr, her, slt_m, slt_e, hmslt, heslt]
    return Dict(k .=> v)
end

cluster_data(M, E, y; type="train") = cluster_data(M, E, y, length(unique(y)); type = type)

function cluster_data1(M, yl, k; type="train")
    y = gvma.encode(yl, unique(yl))

    # try k-medoids 10 times, choose the best using !silhouettes!
    cM = kmedoids(M, k)
    vm = randindex(cM, y)[1]
    slt_m = mean(silhouettes(cM, M))
    for i in 1:9
        _cM = kmedoids(M, k)
        _vm = randindex(_cM, y)[1]
        _slt_m = mean(silhouettes(_cM, M))
        if _slt_m > slt_m
            vm = _vm
            cM = _cM
            slt_m = _slt_m
        end
    end

    hM = cutree(hclust(M, linkage=:average), k=k)
    hmr = randindex(hM, y)[1]
    hmslt = mean(silhouettes(hM, M))

    k = [
        Symbol("ri_medoids_"*type*"_dm"),
        Symbol("ri_hclust_"*type*"_dm"),
        Symbol("slt_medoids_"*type*"_dm"),
        Symbol("slt_hclust_"*type*"_dm"),
    ]
    v = [vm, hmr, slt_m, hmslt]
    return Dict(k .=> v)
end
cluster_data1(M, y; type="train") = cluster_data1(M, y, length(unique(y)); type = type)