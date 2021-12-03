function scatter2(X, x=1, y=2; kwargs...)
    if size(X,1) > size(X,2)
        X = X'
    end
    scatter(X[x,:],X[y,:]; kwargs...)
end
function scatter2!(X, x=1, y=2; kwargs...)
    if size(X,1) > size(X,2)
        X = X'
    end
    scatter!(X[x,:],X[y,:]; kwargs...)
end

# encode labels to numbers
function encode(labels::Vector, labelnames::Vector)
    num_labels = ones(Int, length(labels))
    for i in 1:length(labels)
        v = findall(x -> x == labels[i], labelnames)
        num_labels[i] = v[1]
    end
    return num_labels
end

# encode labels to binary numbers
encode(labels::Vector, missing_class::String) = Int.(labels .== missing_class)

"""
    jaccard_distance(d1::Vector, d2::Vector)

Calculate the Jaccard distance between two vectors
as number of points in intersetion divided by number
of points in the union: d = # intersection / # union.
"""
function jaccard_distance(d1::Vector, d2::Vector)
    #dif = length(setdiff(d1,d2))
    int = length(intersect(d1,d2))
    un = length(union(d1,d2))
    return (un - int)/un
end

########### Flatten Jsons ##############

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
    jpairwise(fps)

Calculates the pairwise Jaccard distances between flatten JSONs
in `fps`. Use multithreading.
"""
function jpairwise(fps)
    d = zeros(Float32, length(fps), length(fps))
    Threads.@threads for i in 1:length(fps)
        for j in i+1:length(fps)
            v = jaccard_distance(fps[i], fps[j])
            d[i,j] = v
        end
    end
    collect(Symmetric(d))
end