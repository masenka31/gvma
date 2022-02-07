using Random

"""
    train_test_split(X, y; ratio=0.5, seed=nothing)

Classic train/test split with given ratio.
"""
function train_test_split(X, y; ratio=0.5, seed=nothing)
    n = length(y)
    n1 = Int(ratio*n)

    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    _ix = sample(1:n, n; replace=false)
    train_ix, test_ix = _ix[1:n1], _ix[n1+1:end]

    # reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    Xtrain, ytrain = X[train_ix], y[train_ix]
    Xtest, ytest = X[test_ix], y[test_ix]

    return (Xtrain, ytrain), (Xtest, ytest)
end

"""
    train_test_split(X, y, L_full::AbstractMatrix; ratio=0.5, seed=nothing)

Train/test split for both data and distance matrix (later used for kNN).
Distance matrix is of size (# test data, # train data).
"""
function train_test_split(X, y, L_full::AbstractMatrix; ratio=0.5, seed=nothing)
    n = length(y)
    n1 = Int(ratio*n)

    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    _ix = sample(1:n, n; replace=false)
    train_ix, test_ix = _ix[1:n1], _ix[n1+1:end]

    # reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    Xtrain, ytrain = X[train_ix], y[train_ix]
    Xtest, ytest = X[test_ix], y[test_ix]

    distance_matrix = L_full[test_ix, train_ix]

    return (Xtrain, ytrain), (Xtest, ytest), distance_matrix
end

"""
    train_test_split(X, y, missing_class::String; ratio=0.5, seed=nothing)

Train/test split for missing class in training data. Chosen missing class
is not put in training data, only in test data.
"""
function train_test_split(X, y, missing_class::String; ratio=0.5, seed=nothing)
    b = y .!= missing_class
    Xm, ym = X[b], y[b]
    Xc, yc = X[.!b], y[.!b]
    
    n = length(ym)
    n1 = Int(ratio*n)

    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    _ix = sample(1:n, n; replace=false)
    train_ix, test_ix = _ix[1:n1], _ix[n1+1:end]

    # reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    Xtrain, ytrain = Xm[train_ix], ym[train_ix]
    Xtest, ytest = cat(Xm[test_ix], Xc), vcat(ym[test_ix], yc)

    return (Xtrain, ytrain), (Xtest, ytest)
end

"""
    train_test_split_reg(X, y, L_full::AbstractMatrix; ratio=0.5, seed=nothing)

Train/test split for both data and distance matrix (later used for kNN).
Distance matrix is of size (# test data, # train data).
"""
function train_test_split_reg(X, y, L_full::AbstractMatrix; ratio=0.5, seed=nothing)
    n = length(y)
    n1 = Int(ratio*n)

    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    _ix = sample(1:n, n; replace=false)
    train_ix, test_ix = _ix[1:n1], _ix[n1+1:end]

    # reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    Xtrain, ytrain = X[train_ix], y[train_ix]
    Xtest, ytest = X[test_ix], y[test_ix]

    dm_train = L_full[train_ix, train_ix]
    dm_test = L_full[test_ix, train_ix]

    return (Xtrain, ytrain), (Xtest, ytest), (dm_train, dm_test)
end
function train_test_split_reg(X, y, L_full::AbstractMatrix, missing_class::String; ratio=0.5, seed=nothing)
    b = y .!= missing_class
    Xm, ym = X[b], y[b]
    Xc, yc = X[.!b], y[.!b]

    dm = L_full[b, b]
    dc = L_full[.!b, .!b]
    
    n = length(ym)
    n1 = Int(ratio*n)

    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    _ix = sample(1:n, n; replace=false)
    train_ix, test_ix = _ix[1:n1], _ix[n1+1:end]

    # reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    Xtrain, ytrain = Xm[train_ix], ym[train_ix]
    Xtest, ytest = cat(Xm[test_ix], Xc), vcat(ym[test_ix], yc)

    dm_train = dm[train_ix, train_ix]
    # we do not need a test distance matrix

    return (Xtrain, ytrain), (Xtest, ytest), (dm_train, )
end

"""
    train_test_split_ix(X, y, L_full::AbstractMatrix; ratio=0.5, seed=nothing)

Train/test split for both data and distance matrix and giving train and
test indexes.
"""
function train_test_split_ix(X, y, L_full::AbstractMatrix; ratio=0.5, seed=nothing)
    n = length(y)
    n1 = Int(ratio*n)

    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    _ix = sample(1:n, n; replace=false)
    train_ix, test_ix = _ix[1:n1], _ix[n1+1:end]

    # reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    Xtrain, ytrain = X[train_ix], y[train_ix]
    Xtest, ytest = X[test_ix], y[test_ix]

    dm_train = L_full[train_ix, train_ix]
    dm_test = L_full[test_ix, train_ix]

    return (Xtrain, ytrain), (Xtest, ytest), (dm_train, dm_test), (train_ix, test_ix)
end