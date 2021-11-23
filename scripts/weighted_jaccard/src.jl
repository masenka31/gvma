############
### Data ###
############

"""
    load_gvma_jaccard()

Loads and flattens JSONs, gives labels, extracts the prefixes.
Be careful about `pwd()` which should be the `gvma` directory.
"""
function load_gvma_jaccard()
    # load data
    df = CSV.read("data/samples_strain.csv", DataFrame)
    files = Vector(df.sha256)
    y = Vector(df.strain)
    labelnames = unique(y)

    labels = encode(y, labelnames)
    samples = map(f -> joinpath("data/samples_strain", "$f.json"), files)

    read_json(file) = JSON.parse(read(file, String))
    jsons = read_json.(samples);
    jsons_flatten = flatten_json.(jsons);

    # create flattened jsons and prefixes
    fps = map(f -> map(j -> (js = j, pr = j[1:end-1]), f),  jsons_flatten)
    # this does not work right now -> should fix it, since it might speed up thing
    # fps = map(f -> Indices(map(j -> (js = j, pr = j[1:end-1]), f)),  jsons_flatten)
    all_prefixes = mapreduce(f -> map(j -> j.pr, collect(f)), vcat, fps) |> unique
    prefix2int = Dict(reverse.(enumerate(all_prefixes)))

    return fps, prefix2int, labels
end

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

Classic train/test split with given ratio. Also gives the cropped distance
matrix, rows correspond to test values, columns to train values.
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


#################
### Functions ###
#################

"""
Flattens a JSON.
"""
function flatten_json(sample::Dict)
    isempty(keys(sample)) && return(Vector{Pair}())
    mapreduce(vcat, collect(keys(sample))) do k 
        childs = flatten_json(sample[k])
        isempty(childs) && return(childs)
        map(childs) do c 
            (k, c...)
        end
    end
end
function flatten_json(sample::Vector)
    isempty(keys(sample)) && return(Vector{Pair}())
    mapreduce(vcat, collect(keys(sample))) do k 
        childs = flatten_json(sample[k])
        isempty(childs) && return(childs)
        map(childs) do c 
            (0, c...)
        end
    end
end
flatten_json(sample) = [(sample,)]


"""
    jaccard_distance(d1::Vector, d2::Vector)

Calculate the Jaccard distance between two vectors
as number of points in intersetion divided by number
of points in the union: d = # intersection / # union.
"""
function jaccard_distance(d1::Vector, d2::Vector)
    int = length(intersect(d1,d2))
    un = length(union(d1,d2))
    return (un - int)/un
end

"""
    jaccard_distance(d1, d2, prefix2int, w)

Calculate the Jaccard distance between two vectors
as number of points in intersetion divided by number
of points in the union: d = # intersection / # union.
Both union and intersection are weighted based on the
weights `w` (mapped from prefixes to indexes via prefix2int).
"""
function jaccard_distance(j1, j2, prefix2int, w)
    int = Zygote.@ignore map(j -> prefix2int[j.pr],intersect(j1,j2))
    un  = Zygote.@ignore map(j -> prefix2int[j.pr],union(j1,j2))
    # if intersection is empty, automatically return one (maximum)
    isempty(int) && return(one(eltype(w)))
    one(eltype(w)) - sum(w[int]) / sum(w[un])
end
function jaccard_distance(j1::Indices, j2::Indices, prefix2int, w)
    int = Zygote.@ignore map(j -> prefix2int[j.pr], collect(intersect(j1,j2)))
    un  = Zygote.@ignore map(j -> prefix2int[j.pr], collect(union(j1,j2)))
    # if intersection is empty, automatically return one (maximum)
    isempty(int) && return(one(eltype(w)))
    one(eltype(w)) - sum(w[int]) / sum(w[un])
end


"""
Zygote.Buffer makes it possible to differentiate setindex.
What needs to happen is for the matrix to be only changed
once and then it can work
"""

"""
    jpairwise(fps, prefix2int, w)

Calculates the pairwise weighted Jaccard distances between flatten JSONs
in `fps`. Differentiable version.
"""
function jpairwise(fps, prefix2int, w)
    d = Zygote.Buffer(zeros(Float32, length(fps), length(fps)))
    for i in 2:length(fps)
        for j in 1:i-1
            v = jaccard_distance(fps[i], fps[j], prefix2int, w)
            d[i,j] = v
            d[j,i] = v
        end
    end
    copy(d)
end

"""
    _jpairwise(fps, prefix2int, w)

Calculates the pairwise weighted Jaccard distances between flatten JSONs
in `fps`. Non-differentiable version - should also be faster.
"""
function _jpairwise(fps, prefix2int, w)
    d = zeros(Float32, length(fps), length(fps))
    Threads.@threads for i in 1:length(fps)
        for j in i+1:length(fps)
            v = jaccard_distance(fps[i], fps[j], prefix2int, w)
            d[i,j] = v
        end
    end
    Symmetric(d)
end

"""
    encode(labels, labelnames)

Encode `String` labels to numbers based on the order given `labelnames`.
"""
function encode(labels, labelnames)
    num_labels = ones(Int, length(labels))
    for i in 1:length(labels)
        v = findall(x -> x == labels[i], labelnames)
        num_labels[i] = v[1]
    end
    return num_labels
end