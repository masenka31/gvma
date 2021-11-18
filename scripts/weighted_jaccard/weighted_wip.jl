"""
This scripts needs Flux@12. Therefore, it is necessary to load
a different than GVMA directory and Manifest using DrWatson.

Do
`] activate`
to activate this environment.
"""

using IterTools
using Dictionaries
using Flux, Flux.Zygote
using ClusterLosses
using JSON

cd("..")

#################
### Functions ###
#################

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

function jaccard_distance(d1::Vector, d2::Vector)
    int = length(intersect(d1,d2))
    un = length(union(d1,d2))
    return (un - int)/un
end
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

function _jpairwise(fps, prefix2int, w)
    d = zeros(Float32, length(fps), length(fps))
    for i in 2:length(fps)
        for j in 1:i-1
            v = jaccard_distance(fps[i], fps[j], prefix2int, w)
            d[i,j] = v
            d[j,i] = v
        end
    end
    d
end

function encode(labels, labelnames)
    num_labels = ones(Int, length(labels))
    for i in 1:length(labels)
        v = findall(x -> x == labels[i], labelnames)
        num_labels[i] = v[1]
    end
    return num_labels
end

#################
### GVMA data ###
#################

using DrWatson
using DataFrames, CSV

function load_gvma_jaccard()
    # load data
    df = CSV.read("data\\samples_strain.csv", DataFrame)
    files = Vector(df.sha256)
    y = Vector(df.strain)
    labelnames = unique(y)

    labels = encode(y, labelnames)
    samples = map(f -> joinpath("data\\samples_strain", "$f.json"), files)

    read_json(file) = JSON.parse(read(file, String))
    jsons = read_json.(samples);
    jsons_flatten = flatten_json.(jsons);

    # create flattened jsons and prefixes
    fps = map(f -> map(j -> (js = j, pr = j[1:end-1]), f),  jsons_flatten)
    # fps = map(f -> Indices(map(j -> (js = j, pr = j[1:end-1]), f)),  jsons_flatten)
    all_prefixes = mapreduce(f -> map(j -> j.pr, collect(f)), vcat, fps) |> unique
    prefix2int = Dict(reverse.(enumerate(all_prefixes)))

    return fps, prefix2int, labels
end

using Random, StatsBase

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

# load data and split
fps, prefix2int, labels = load_gvma_jaccard();
(fps_train, l_tr), (fps_test, l_test) = train_test_split(fps, labels; ratio=0.2, seed=1);

# initialize parameters
w = ones(Float32, length(prefix2int))

# @time jpairwise(fps[1:100], prefix2int, w);
# @time _jpairwise(fps[1:100], prefix2int, w);

# batches
batchsize = 128
function prepare_minibatch()
    ix = rand(1:length(l_tr), batchsize)
    fps_train[ix], l_tr[ix]
end
mb_provider = IterTools.repeatedly(prepare_minibatch, 10)
batch = prepare_minibatch();

# loss function
lossf(x, y) = loss(Triplet(), jpairwise(x, prefix2int, σ.(w) .+ 1f-6), y)

# Flux parameters, optimizer
ps = Flux.params([w])
opt = ADAM()

# try training - does the loss decrease?
lossf(batch...)
Flux.train!(ps, mb_provider, opt) do x, y
    loss(Triplet(), jpairwise(x, prefix2int, σ.(w) .+ 1f-6), y)
end
lossf(batch...)

using Flux: @epochs

# train for given number of epochs
@epochs 10 begin
    Flux.train!(ps, mb_provider, opt) do x, y
        loss(Triplet(), jpairwise(x, prefix2int, σ.(w) .+ 1f-6), y)
    end
    @show lossf(batch...)
end

# calculate the resulting distance matrix
L_full_weighted = jpairwise(fps, prefix2int, σ.(w) .+ 1f-6)
safesave("weigted_matrix.bson", Dict(:L => L_full_weighted))