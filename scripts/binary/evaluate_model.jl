function evaluate_model(X::ProductNode, y::Vector, L::AbstractMatrix, filename::String; k::Int=15)
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
    clabels = assignments(c)
    yenc_binary = encode(ytest, missing_class)

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

    # create a report
    r = report(CM)

    # get everything together
    df = DataFrame(
        :k => k,
        :k_positive => length(percentages),
        :percentages => [percentages],
        :randindex => ri,
        :adj_randindex => ri_adj
    )

    parameters = DataFrame([:ratio, :seed, :α, :missing_class] .=> [ratio, seed, α, missing_class])

    hcat(r, df, parameters)
end


