using Random

"""
    random_ix(n::Int, seed=nothing)

This function generates random indexes based on the maximum number
and given seed. If no seed is set, samples randomly.
"""
function random_ix(n::Int, seed=nothing)
    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    _ix = sample(1:n, n; replace=false)

    # reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    return _ix
end

"""
    train_test_split(X::ProductNode, y; ratio=0.5, seed=nothing)

Classic train/test split with given ratio.
"""
function train_test_split(X::ProductNode, y::Vector; ratio=0.5, seed=nothing)
    n = length(y)
    n1 = Int(ratio*n)

    # get the indexes
    _ix = random_ix(n, seed)
    
    # split indexes to train/test
    train_ix, test_ix = _ix[1:n1], _ix[n1+1:end]

    # get data
    Xtrain, ytrain = X[train_ix], y[train_ix]
    Xtest, ytest = X[test_ix], y[test_ix]

    return (Xtrain, ytrain), (Xtest, ytest)
end

"""
    train_test_split(X::ProductNode, y::Vector, L_full::AbstractMatrix; ratio=0.5, seed=nothing)

Train/test split for both data and distance matrix (later used for kNN).
Distance matrix is of size (# test data, # train data).
"""
function train_test_split(X::ProductNode, y::Vector, L_full::AbstractMatrix; ratio=0.5, seed=nothing)
    n = length(y)
    n1 = Int(ratio*n)

    # get the indexes
    _ix = random_ix(n, seed)
    
    # split indexes to train/test
    train_ix, test_ix = _ix[1:n1], _ix[n1+1:end]

    # get data
    Xtrain, ytrain = X[train_ix], y[train_ix]
    Xtest, ytest = X[test_ix], y[test_ix]
    distance_matrix = L_full[test_ix, train_ix]

    return (Xtrain, ytrain), (Xtest, ytest), distance_matrix
end

"""
    train_test_split(L_full::AbstractMatrix; ratio=0.5, seed=nothing)

Train/test split for distance matrix (e.g. Jaccard distance matrix).
Returns distance matrix is of size (# test data, # train data).
"""
function train_test_split(y::Vector, L_full::AbstractMatrix; ratio=0.5, seed=nothing)
    n = size(L_full,1)
    n1 = round(Int, ratio*n)

    # get the indexes
    _ix = random_ix(n, seed)
    
    # split indexes to train/test
    train_ix, test_ix = _ix[1:n1], _ix[n1+1:end]

    # get data
    y_train, y_test = y[train_ix], y[test_ix]
    distance_matrix = L_full[test_ix, train_ix]

    return (y_train, y_test), distance_matrix
end

"""
    train_test_split(X::ProductNode, y::Vector, missing_class::String; ratio=0.5, seed=nothing)

Train/test split for missing class in training data. Chosen missing class
is not put in training data, only in test data.
"""
function train_test_split(X::ProductNode, y::Vector, missing_class::String; ratio=0.5, seed=nothing)
    b = y .!= missing_class
    Xm, ym = X[b], y[b]
    Xc, yc = X[.!b], y[.!b]
    
    n = length(ym)
    n1 = Int(ratio*n)

    # get the indexes
    _ix = random_ix(n, seed)
    train_ix, test_ix = _ix[1:n1], _ix[n1+1:end]

    Xtrain, ytrain = Xm[train_ix], ym[train_ix]
    Xtest, ytest = cat(Xm[test_ix], Xc), vcat(ym[test_ix], yc)

    return (Xtrain, ytrain), (Xtest, ytest)
end

"""
    train_test_split_reg(X::ProductNode, y::Vector, L_full::AbstractMatrix; ratio=0.5, seed=nothing)

Train/test split for both data and distance matrix (later used for kNN).
Distance matrix is of size (# test data, # train data).
"""
function train_test_split_reg(X::ProductNode, y::Vector, L_full::AbstractMatrix; ratio=0.5, seed=nothing)
    n = length(y)
    n1 = Int(ratio*n)

    # get the indexes
    _ix = random_ix(n, seed)
    train_ix, test_ix = _ix[1:n1], _ix[n1+1:end]

    Xtrain, ytrain = X[train_ix], y[train_ix]
    Xtest, ytest = X[test_ix], y[test_ix]

    dm_train = L_full[train_ix, train_ix]
    dm_test = L_full[test_ix, train_ix]

    return (Xtrain, ytrain), (Xtest, ytest), (dm_train, dm_test)
end
function train_test_split_reg(X::ProductNode, y::Vector, L_full::AbstractMatrix, missing_class::String; ratio=0.5, seed=nothing)
    b = y .!= missing_class
    Xm, ym = X[b], y[b]
    Xc, yc = X[.!b], y[.!b]

    dm = L_full[b, b]
    dc = L_full[.!b, .!b]
    
    n = length(ym)
    n1 = Int(ratio*n)

    # get the indexes
    _ix = random_ix(n, seed)
    train_ix, test_ix = _ix[1:n1], _ix[n1+1:end]

    Xtrain, ytrain = Xm[train_ix], ym[train_ix]
    Xtest, ytest = cat(Xm[test_ix], Xc), vcat(ym[test_ix], yc)

    dm_train = dm[train_ix, train_ix]
    # we do not need a test distance matrix

    return (Xtrain, ytrain), (Xtest, ytest), (dm_train, )
end

"""
    train_test_split_ix(X::ProductNode, y::Vector, L_full::AbstractMatrix; ratio=0.5, seed=nothing)

Train/test split for both data and distance matrix and giving train and
test indexes.
"""
function train_test_split_ix(X::ProductNode, y::Vector, L_full::AbstractMatrix; ratio=0.5, seed=nothing)
    n = length(y)
    n1 = Int(ratio*n)

    # get the indexes
    _ix = random_ix(n, seed)
    train_ix, test_ix = _ix[1:n1], _ix[n1+1:end]

    Xtrain, ytrain = X[train_ix], y[train_ix]
    Xtest, ytest = X[test_ix], y[test_ix]

    dm_train = L_full[train_ix, train_ix]
    dm_test = L_full[test_ix, train_ix]

    return (Xtrain, ytrain), (Xtest, ytest), (dm_train, dm_test), (train_ix, test_ix)
end

function train_test_split(X::ProductNode, y::Vector, Xs::ProductNode, ys::Vector; ratio=0.5, seed=nothing)
    (Xtrain, ytrain), (Xtest, ytest) = train_test_split(X, y; ratio=ratio, seed = seed)
    return (Xtrain, ytrain), (cat(Xtest, Xs), vcat(ytest, ys))
end