##################################
### Create Jaccard as a metric ###
##################################

using Distances: UnionMetric
import Distances: result_type

struct SetJaccard <: UnionMetric end

function (dist::SetJaccard)(x, y)
    int = length(intersect(x,y))
    un = length(union(x,y))
    return (un - int)/un
end
result_type(dist::SetJaccard, x, y) = Float32

struct WeightedJaccard <: UnionMetric
    instance2int::Dict
    W::Vector
end
WeightedJaccard(instance2int::Dict) = WeightedJaccard(instance2int, ones(Float32, length(instance2int)))
function WeightedJaccard(data::Vector)
    instance_dict = unique(data)
    instance2int = Dict(reverse.(enumerate(instance_dict)))
    WeightedJaccard(instance2int)
end
import Base: length
length(dist::WeightedJaccard) = length(dist.W)

function weight_model(x, w, instance2int)
    if x in keys(instance2int)
        return σ(w[instance2int[x]])
    else
        return 1f0
    end
end

function (dist::WeightedJaccard)(x, y)
    int = Zygote.@ignore intersect(x, y)
    un  = Zygote.@ignore union(x, y)

    int_w = map(i -> weight_model(i, dist.W, dist.instance2int), int)
    un_w = map(i -> weight_model(i, dist.W, dist.instance2int), un)

    isempty(int) && return(one(eltype(W)))
    one(eltype(int_w)) - sum(int_w) / sum(un_w)
end
result_type(dist::WeightedJaccard, x, y) = Float32

# @time pairwise(WeightedJaccard(instance2int, W), xun)
# @time jwpairwise(xun, instance2int, W)

"""
This does not work since parameters od WeightedJaccard need to be mutable
and currently cannot be differentiated through.

Flux.trainable(dist::WeightedJaccard) = (W = dist.W, )

# create weight model
model = WeightedJaccard(vcat(xun...))
ps = Flux.params(model)
opt = ADAM(0.005)

# create loss and compute it on a batch
# lossf(x, y) = ClusterLosses.loss(Triplet(), jwpairwise(x, instance2int, W), y)
lossf(x, y) = ClusterLosses.loss(Triplet(), pairwise(model, x), y)
lossf(batch...)

@epochs 5 begin
    Flux.train!(ps, mb_provider, opt) do x, y
        lossf(x, y)
    end
    @show lossf(batch...)
end
"""