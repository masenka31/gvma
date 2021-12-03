"""
Implements structure for a Triplet.

- x: the sample
- x⁺: sample with same class as x
- x⁻: sample with different class as x
"""
struct Triplet{T}
    x::T
    x⁺::T
    x⁻::T
end

function Base.show(io::IO, t::Triplet)
    tp = String(Symbol(typeof(t.x)))
    if length(tp) < 20
        print(io, "Triplet{$tp}")
    else
        print(io, "Triplet{$(tp[1:17])...}")
    end
end

"""
Implements structure for a Triplet.

- x: the sample
- x⁺: sample with same class as x
- x⁻: sample with different class as x
- l⁺: positive class label
- l⁻: negative class label

struct Triplet{T, S}
    x::T
    x⁺::T
    x⁻::T
    l⁺::S
    l⁻::S
end

# Constructor for Triplet without label classes.
# maybe later
# Triplet(x,y,z) = Triplet(x, y, z, 0, 1)
"""

using Flux

"""
Implements structure for a TripletModel. Is used also for pairwise
distance calculation.
"""
struct TripletModel{T<:Chain}
    f::T
end

# make the model trainable
Flux.trainable(tm::TripletModel) = (tm.f,)


"""
	labelmap(y)

Create a dictionary of labels => indexes.
"""
function labelmap(y::AbstractVector{T}) where {T}
	d = Dict{T, Vector{Int}}()
	for (i,v) in enumerate(y)
		if haskey(d, v)
			push!(d[v], i)
		else
			d[v] = [i]
		end
	end
	d
end

"""
    sample_triplet(data::T, labels::T) where T <: AbstractVector

Samples x, x⁺, x⁻ and creates a Triplet.
Here, data are indexed as a vector: `data[ix]`.
"""
function sample_triplet(data::T, labels::T) where T <: AbstractVector
    labelnames = unique(labels)
    mp = labelmap(labels)
    positive = sample(labelnames)
    negative = sample(setdiff(labelnames, positive))

    positive_ix = mp[positive]
    negative_ix = mp[negative]

    return Triplet(
        data[sample(positive_ix)],
        data[sample(positive_ix)],
        data[sample(negative_ix)]
    )
end
"""
    sample_triplet(data::T, labels::T) where T <: AbstractMatrix
    
Samples x, x⁺, x⁻ and creates a Triplet.
Here, data are indexed as a vector: `data[:, ix]`.
"""
function sample_triplet(data::T, labels::Vector) where T <: AbstractMatrix
    labelnames = unique(labels)
    mp = labelmap(labels)
    positive = sample(labelnames)
    negative = sample(setdiff(labelnames, positive))

    positive_ix = mp[positive]
    negative_ix = mp[negative]

    return Triplet(
        data[:, sample(positive_ix)],
        data[:, sample(positive_ix)],
        data[:, sample(negative_ix)]
    )
end
"""
    sample_triplet(data::T, labels::T) where T <: ProductNode
    
Samples x, x⁺, x⁻ and creates a Triplet.
Here, data are indexed as a vector: `data[:, ix]`.
"""
function sample_triplet(data::T, labels::Vector) where T <: ProductNode
    labelnames = unique(labels)
    mp = labelmap(labels)
    positive = sample(labelnames)
    negative = sample(setdiff(labelnames, positive))

    positive_ix = mp[positive]
    negative_ix = mp[negative]

    return Triplet(
        data[sample(positive_ix)],
        data[sample(positive_ix)],
        data[sample(negative_ix)]
    )
end

l2_dist(tm::TripletModel, x, y) = norm(tm.f(x) - tm.f(y))
l2_dist(x, y) = norm(x - y)
l2_dist(tm::TripletModel, x::Triplet) = [l2_dist(tm, x.x, x.x⁺), l2_dist(tm, x.x, x.x⁻)]

import Distances: pairwise
pairwise(tm::TripletModel, x::AbstractMatrix) = pairwise(Euclidean(), tm.f(x))
pairwise(tm::TripletModel, x::ProductNode) = pairwise(Euclidean(), tm.f(x))


triplet_loss(tm::TripletModel, x::Triplet) = Flux.mse(softmax(l2_dist(tm, x)), [0,1])

function triplet_loss(tm::TripletModel, x::Triplet)
    y⁺ = euclidean(tm.f(x.x), tm.f(x.x⁺))
    y⁻ = euclidean(tm.f(x.x), tm.f(x.x⁻))
    Flux.mse(softmax([y⁺, y⁻]), [0f0,1f0])
end