function evaluate_model(X::ProductNode, y::Vector, filename::String; k::Int=15)
    d = BSON.load(datadir("models/missing/$filename"))
    @unpack ratio, seed, α, missing_class, model = d
    (Xtrain, ytrain), (Xtest, ytest) = train_test_split(X, y, missing_class; ratio=ratio, seed=seed)
    full_model = model
    mill_model = model[1]

    # get the latent encoding
    enc_ts = mill_model(Xtest).data
    # calculate pairwise Euclidean distance
    M_test = pairwise(Euclidean(), enc_ts)

    # perform k-medoids clustering
    c = kmedoids(M_test, k)
    silh = silhouettes(c, L)
    clabels = assignments(c)
    yenc_binary = encode(ytest, missing_class)

    # get the counts for clusters and labels
    cs = counts(clabels, yenc_binary)

    # get adjusted randindex
    ri_adj = randindex(clabels, yenc_binary)[1]
    ri = randindex(clabels, yenc_binary)[2]

    # get the metrics for full clustering classification (given all labels)
    yfull = encode(yt, unique(yt))
    full_riadj, full_ri = randindex(clabels, yfull)[1:2]

    # get "purity" percentages of clusters with missing class
    s = sum(cs, dims=2)
    miss_cluster_ix = findall(x -> x != 0, cs[:, 2])
    percentages = cs[miss_cluster_ix, 2] ./ s[miss_cluster_ix]

    # get confusion matrix for binary classification
    new_bit = cs[:,1] .< cs[:,2]
    TN, FN = Tuple(sum(cs[.!new_bit, :], dims=1)) # gives TN, FN
    FP, TP = Tuple(sum(cs[new_bit, :], dims=1))   # gives FP, TP
    CM = ConfusionMatrix(TP, FP, FN, TN)

    # create a report (accuracy, recall, precision)
    r = report(CM)

    # get metrics
    df = DataFrame(
        :k => k,
        :k_positive => length(percentages),
        :percentages => [percentages],
        :randindex => ri,
        :adj_randindex => ri_adj,
        :full_randindex => full_ri,
        :full_adj_randindex => full_riadj,
        :silhouettes => mean(silh)
    )

    # add parameters to dataframe
    parameters = DataFrame([:ratio, :seed, :α, :missing_class] .=> [ratio, seed, α, missing_class])

    # put it all together and return
    hcat(r, df, parameters)
end

function evaluate_model(X::ProductNode, y::Vector, Xs::ProductNode, ys::Vector, filename::String; k::Int=15)
    d = BSON.load(datadir("models/missing/$filename"))
    @unpack ratio, seed, α, missing_class, model = d
    (_, ytrain), (Xtest, ytest) = train_test_split(X, y, missing_class; ratio=ratio, seed=seed)
    full_model = model
    mill_model = model[1]

    # cat the testing data and new small classes
    Xt = cat(Xtest, Xs)
    yt = vcat(ytest, ys)

    # get binary labels for test data
    ln = unique(ytrain)
    yenc_binary = .!map(i -> any(i .== ln), yt)

    # get the latent encoding
    enc_ts = mill_model(Xt).data
    # calculate pairwise Euclidean distance
    M_test = pairwise(Euclidean(), enc_ts)

    # perform k-medoids clustering
    c = kmedoids(M_test, k)
    silh = silhouettes(c, L)
    clabels = assignments(c)

    # get the metrics for full clustering classification (given all labels)
    yfull = encode(yt, unique(yt))
    full_riadj, full_ri = randindex(clabels, yfull)[1:2]

    # get the counts for clusters and labels
    cs = counts(clabels, yenc_binary)

    # get adjusted randindex
    ri_adj = randindex(clabels, yenc_binary)[1]
    ri = randindex(clabels, yenc_binary)[2]

    # get "purity" percentages of clusters with missing class
    s = sum(cs, dims=2)
    miss_cluster_ix = findall(x -> x != 0, cs[:, 2])
    percentages = cs[miss_cluster_ix, 2] ./ s[miss_cluster_ix]

    # get confusion matrix for binary classification
    new_bit = cs[:,1] .< cs[:,2]
    TN, FN = Tuple(sum(cs[.!new_bit, :], dims=1)) # gives TN, FN
    FP, TP = Tuple(sum(cs[new_bit, :], dims=1))   # gives FP, TP
    CM = ConfusionMatrix(TP, FP, FN, TN)

    # create a report (accuracy, recall, precision)
    r = report(CM)

    # get metrics
    df = DataFrame(
        :k => k,
        :k_positive => length(percentages),
        :percentages => [percentages],
        :randindex => ri,
        :adj_randindex => ri_adj,
        :full_randindex => full_ri,
        :full_adj_randindex => full_riadj,
        :silhouettes => mean(silh)
    )

    # add parameters to dataframe
    parameters = DataFrame([:ratio, :seed, :α, :missing_class] .=> [ratio, seed, α, missing_class])

    # put it all together and return
    hcat(r, df, parameters)
end

function evaluate_jaccard(y::Vector, missing_class::String, L::AbstractMatrix; k = 10)
    # perform clustering on distance matrix
    c = kmedoids(L, 10)
    silh = silhouettes(c, L)
    clabels = assignments(c)

    # k - RandIndex
    ynum = encode(y, labelnames)
    adj_ri, ri = randindex(c, ynum)[1:2]

    # binary RandIndex
    y_binary = encode(y, String(missing_class))
    binary_ri_adj, binary_ri = randindex(clabels, y_binary)[1:2]

    # get confusion matrix for binary classification
    cs = counts(clabels, y_binary)
    new_bit = cs[:,1] .< cs[:,2]
    TN, FN = Tuple(sum(cs[.!new_bit, :], dims=1)) # gives TN, FN
    FP, TP = Tuple(sum(cs[new_bit, :], dims=1))   # gives FP, TP
    CM = ConfusionMatrix(TP, FP, FN, TN)
    # create report
    r = report(CM)

    hcat(
        DataFrame(:class => missing_class),
        r,
        DataFrame(
            :ri => ri,
            :adj_ri => adj_ri,
            :b_ri => binary_ri,
            :b_ri_adj => binary_ri_adj,
            :silhouettes => mean(silh)
        )
    )
end