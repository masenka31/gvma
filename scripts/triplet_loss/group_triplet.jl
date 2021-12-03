using DrWatson
@quickactivate

using gvma
using Flux
using StatsBase: sample
using LinearAlgebra: norm
using Statistics: mean

include(srcdir("triplet.jl"))

x1 = randn(2,300) .+ [5,3]
x2 = randn(2,800) .- [1,1]
l1 = zeros(Int, size(x1,2))
l2 = ones(Int, size(x2,2))

data = hcat(x1,x2)
labels = vcat(l1,l2)

f = Chain(Dense(2,20,relu), Dense(20,2))
model = TripletModel(f)
opt = ADAM()

loss(x::Triplet) = triplet_loss(model, x)
loss(x::Vector) = mean(map(xi -> loss(xi), x))

minibatch(;batchsize=128) = [sample_triplet(data, labels) for i in 1:batchsize]
minibatch(;batchsize=128) = [sample_triplet(newdata, newlabels) for i in 1:batchsize]

batch = minibatch()
loss(batch)

for i in 1:100
    batch = minibatch()
    Flux.train!(loss, Flux.params(model), batch, opt)
    if mod(i,10) == 1
        @show loss(batch)
    end
end

# clustering power evaluation
using Clustering, Distances
l = labels
c1 = kmedoids(pairwise(Euclidean(), data),2)
c2 = kmedoids(pairwise(model, data), 2)
randindex(c1,c2)
randindex(c1,l .+ 1)
randindex(c2,l .+ 1)

# add new cluster
x3 = randn(2,400) .+ [6,-3]
newdata = hcat(data, x3)
newlabels = vcat(labels, ones(Int, size(x3,2)) .+ 1)

c1 = kmedoids(pairwise(Euclidean(), newdata),3)
c2 = kmedoids(pairwise(model, newdata), 3)
randindex(c1,c2)
randindex(c1,newlabels .+ 1)
randindex(c2,newlabels .+ 1)

using Plots
ENV["GKSwstype"] = "100"

plt = scatter2(newdata, color=c2.assignments)
wsave("gaussian3_triplet.png", plt)

# umap embedding
using UMAP
emb1 = umap(pairwise(Euclidean(), newdata), 2; metric=:precomputed)
emb2 = umap(pairwise(model, newdata), 2; metric=:precomputed)

e1 = scatter2(emb1, color=newlabels)
e2 = scatter2(emb2, color=newlabels)
ee = plot(e1,e2,layout=(1,2))

wsave("embedding.png", ee)

enc = map(x -> model.f(x), eachcol(newdata))
enc = hcat(enc...)

plt = scatter2(enc, color=newlabels)
wsave("encoding.png", plt)
