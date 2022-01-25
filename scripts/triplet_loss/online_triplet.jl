using DrWatson
@quickactivate

using gvma
using ClusterLosses
using Distances
using Flux

# plotting on headless environment
using Plots
ENV["GKSwstype"] = "100"

# matrix sampling
import StatsBase: sample
function sample(x::AbstractMatrix)
    ix = sample(1:size(x,2))
    return x[:, ix]
end
sample(x::AbstractMatrix, n::Int) = hcat([sample(x) for _ in 1:n]...)
function sample(x::AbstractMatrix, labels::Vector, n::Int)
    ix = sample(1:size(x,2), n)
    (x[:, ix], labels[ix])
end

# create some sample data and labels
x1 = randn(2,300) .+ [5,3]
x2 = randn(2,800) .- [1,1]
l1 = zeros(Int, size(x1,2))
l2 = ones(Int, size(x2,2))
data = hcat(x1,x2)
labels = vcat(l1,l2)

# function to create minibatches
minibatch(batchsize=128) = sample(data, labels, batchsize)
batch = minibatch()

model = Chain(Dense(2,20,tanh), Dense(20,2))
loss(Triplet(), SqEuclidean(), model(batch[1]), batch[2])

import ClusterLosses: loss
lossf(x, y) = loss(Triplet(), SqEuclidean(), model(x), y)
lossf(batch) = lossf(batch...)

opt = ADAM()

using IterTools
mb_provider = IterTools.repeatedly(minibatch, 100)

ps = Flux.params(model)
Flux.train!(ps, mb_provider, opt) do x, y
    loss(Triplet(), SqEuclidean(), model(x), y)
    @show loss(Triplet(), SqEuclidean(), model(batch[1]), batch[2])    
end

loss(Triplet(), SqEuclidean(), model(batch[1]), batch[2])

enc = model(data)
scatter2(enc, color=labels)
savefig("encoding.png")

x3 = randn(2,400) .+ [6,-3]
newdata = hcat(data, x3)
newlabels = vcat(labels, ones(Int, 400) .+ 1)

enc = model(newdata)
scatter2(enc, color=newlabels)
savefig("encoding.png")