using gvma
using gvma: jaccard_distance, jpairwise

using Distributions
using StatsBase
using Mill
using Random
using Flux
using Flux: @epochs
using Flux.Zygote
using Base.Iterators: repeated

using IterTools
using Clustering, Distances
using ClusterLosses

using Plots, UMAP

#######################
### Data generation ###
#######################


"""
    get_obs(x)

Returns bag indices from an of iterable collection.
For a vector of lengths `l = [4,5,2]` returns `obs = [1:4, 5:9, 10:11]`.
"""
function get_obs(x)
    l = nobs.(x)
    n = length(l)
    lv = vcat(0,l)
    mp = map(i -> sum(lv[1:i+1]), 1:n)
    mpv = vcat(0,mp)
    obs = map(i -> mpv[i]+1:mpv[i+1], 1:n)
end

function generate_mill_noise(class_code::Vector{Vector{T}}, λ) where T <: Number
    space = [1:100,1:100]
    s = map(i -> sample.(space), 1:rand(Poisson(λ)))
    s = setdiff(s, class_code)
    oh = map(x -> Flux.onehotbatch(x,1:100), s)
    an = cat(ArrayNode.(oh)...)
    bn = BagNode(an, map(i -> 2i-1:2i, 1:length(s)))
end

function generate_mill_noise(code_instances::Vector{T}, λ) where T <: Number
    s = sample(1:100, rand(Poisson(λ))*2)
    s = setdiff(s, code_instances)

    while isempty(s)
        s = sample(1:100, rand(Poisson(λ))*2)
        s = setdiff(s, code_instances)
    end
    
    if mod(length(s),2) != 0
        s = s[1:end-1]
    end

    s2d = [[s[2i-1], s[2i]] for i in 1:length(s)÷2]
    oh = map(x -> Flux.onehotbatch(x,1:100), s2d)
    an = cat(ArrayNode.(oh)...)
    bn = BagNode(an, map(i -> 2i-1:2i, 1:length(s2d)))
end

function generate_mill_data(n_classes, n_bags; λ = 20, seed = nothing)
    # class code can be fixed with a seed, noise cannot
    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    space = [1:100,1:100]
    class_code = [sample.(space) for _ in 1:n_classes]

    # reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    data = BagNode[]
    labels = Int[]

    for i in 1:n_bags
        ns = generate_mill_noise(class_code, λ)
        ix = sample(1:n_classes)
        oh_code = Flux.onehotbatch(class_code[ix], 1:100)
        bn_code = BagNode(ArrayNode(oh_code), [1:2])
        push!(data, cat(ns, bn_code))
        
        push!(labels, collect(1:n_classes)[ix])
    end
    return BagNode(cat(data...), get_obs(data)), labels, class_code
end


function generate_mill_unique(n_classes, n_bags; λ = 20, seed = nothing)
    # class code can be fixed with a seed, noise cannot
    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    code_instances = sample(1:100, n_classes*2, replace=false)
    class_code = [[code_instances[2i-1], code_instances[2i]] for i in 1:length(code_instances)÷2]

    # reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    data = BagNode[]
    labels = Int[]

    for i in 1:n_bags
        ns = generate_mill_noise(code_instances, λ)
        ix = sample(1:n_classes)
        oh_code = Flux.onehotbatch(class_code[ix], 1:100)
        bn_code = BagNode(ArrayNode(oh_code), [1:2])
        push!(data, cat(ns, bn_code))
        
        push!(labels, collect(1:n_classes)[ix])
    end
    return BagNode(cat(data...), get_obs(data)), labels, class_code
end

"""
    unpack(X)

Unpacks a BagNode{BagNode{ArrayNode}} structure to vector of vectors...
"""
function unpack(X)
    x1 = map(i -> X[i].data, 1:nobs(X))
    x2 = map(x -> x.data.data, x1)
    x3 = map(bag -> map(i -> bag[:, 2i-1:2i], 1:size(bag,2)÷2), x2)
end

#########################
### Jaccard functions ###
#########################

"""
    weight_model(x, w, instance2int)

For given x, weights vector w and dictionary of known terms instance2int
returns either bounded weight if x is known and 1 if x is unknown.
"""
function weight_model(x, w, instance2int)
    if x in keys(instance2int)
        return σ(w[instance2int[x]])
    else
        return 1f0
    end
end


"""
    jw(x, y, instance2int, W)

Calculates the weighted Jaccard distance between x, y.
"""
function jw(x, y, instance2int, W)
    int = Zygote.@ignore intersect(x, y)
    un  = Zygote.@ignore union(x, y)

    int_w = map(i -> weight_model(i, W, instance2int), int)
    un_w = map(i -> weight_model(i, W, instance2int), un)

    isempty(int) && return(one(eltype(W)))
    one(eltype(int_w)) - sum(int_w) / sum(un_w)
end

"""
    jwpairwise(fps, instance2int, W)

For given vector of bags fps returns pairwise weighted Jaccard distance matrix.
"""
function jwpairwise(fps, instance2int, W)
    d = Zygote.Buffer(zeros(Float32, length(fps), length(fps)))
    for i in 2:length(fps)
        for j in 1:i-1
            v = jw(fps[i], fps[j], instance2int, W)
            d[i,j] = v
            d[j,i] = v
        end
    end
    copy(d)
end

"""
    _jwpairwise(fps, instance2int, W)

For given vector of bags fps returns pairwise weighted Jaccard distance matrix.
Distributed on threads.
"""
function _jwpairwise(fps, instance2int, W)
    d = Zygote.Buffer(zeros(Float32, length(fps), length(fps)))
    for i in 2:length(fps)
        Threads.@threads for j in 1:i-1
            v = jw(fps[i], fps[j], instance2int, W)
            d[i,j] = v
            d[j,i] = v
        end
    end
    copy(d)
end


# functions
"""
function jaccard_weighted(j1, j2)
    int = Zygote.@ignore intersect(j1,j2)
    un  = Zygote.@ignore union(j1,j2)

    isempty(int) && return(one(eltype(weight_model(un[1]))))
    one(eltype(weight_model(un[1]))) - sum(weight_model.(int)) / sum(weight_model.(un))
end

function jpairwise_weighted(data)
    d = Zygote.Buffer(zeros(Float32, length(data), length(data)))
    for i in 2:length(data)
        for j in 1:i-1
            v = jaccard_weighted(data[i], data[j])
            d[i,j] = v
            d[j,i] = v
        end
    end
    copy(d)
end
"""

######################
### Plotting utils ###
######################

function marker_shape(labels)
    shapes = [:circle, :square, :x, :utriangle, :dtriangle, :star, :+]
    s = Symbol[]
    for i in labels
        push!(s, shapes[i])
    end
    s
end