using DrWatson
@quickactivate

using Flux, Flux.Zygote

x1 = randn(2,300) .+ [5,3]
x2 = randn(2,800) .- [1,1]

l1 = zeros(Int, size(x1,2))
l2 = ones(Int, size(x2,2))
l = vcat(l1,l2)

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

function sample_triplet(x1,x2)
    if rand() > 0.5
        x, xp = eachcol(sample(x1,2))
        xn = sample(x2)
        return (x, xp, xn)
    else
        x, xp = eachcol(sample(x2,2))
        xn = sample(x1)
        return (x, xp, xn)
    end
end

using LinearAlgebra

minibatch(;batchsize=128) = [sample_triplet(x1,x2) for i in 1:batchsize]

d(x, y) = norm(f(x) - f(y))
sm(x, y, z) = softmax(map(s -> d(x, s), [y, z]))
loss(x, y, z) = Flux.mse(sm(x, y, z), [0,1])

loss(x::Vector) = mean(map(xi -> loss(xi...), x))

f = Chain(Dense(2,10,sigmoid), Dense(10,2))
opt = ADAM()
for i in 1:100
    batch = minibatch()
    Flux.train!(loss, Flux.params(f), batch, opt)
    @show loss(batch)
end

XY = hcat(x1,x2)
M = zeros(size(XY,2), size(XY,2))

for i in 1:size(XY,2)
    for j in i:size(XY,2)
        m = d(XY[:,i], XY[:, j])
        M[i,j] = m
        M[j,i] = m
    end
end

using Plots
ENV["GKSwstype"] = "100"

using gvma

plt = scatter2(XY, color=vcat(l1,l2))
wsave("gaussian_test.png", plt)

using Clustering, Distances
c1 = kmedoids(pairwise(Euclidean(), XY, XY),2)
c2 = kmedoids(M, 2)
randindex(c1,c2)
randindex(c1,l .+ 1)
randindex(c2,l .+ 1)

plt = scatter2(XY, color=c2.assignments)
wsave("gaussian_triplet.png", plt)

# and now what if I add a cluster that has not been there before...
# what is going to happen, how will the distance matrix be affected?

x3 = randn(2,400) .+ [6,-3]

XYZ = hcat(XY,x3)
l3 = vcat(l, ones(Int, size(x3,2)) .+ 1)

plt = scatter2(XYZ, color=l3)
wsave("gaussian3.png", plt)

M3 = zeros(size(XYZ,2), size(XYZ,2))

for i in 1:size(XYZ,2)
    for j in i:size(XYZ,2)
        m = d(XYZ[:,i], XYZ[:, j])
        M3[i,j] = m
        M3[j,i] = m
    end
end

c3 = kmedoids(M3, 3)
randindex(c3, l3 .+ 1)

plt = scatter2(XYZ, color=c3.assignments)
wsave("gaussian3_triplet.png", plt)

# UMAP??

emb1 = umap(pairwise(Euclidean(), XYZ), 2; metric=:precomputed)
emb2 = umap(M3, 2; metric=:precomputed)

e1 = scatter2(emb1, color=l3 .+ 1)
e2 = scatter2(emb2, color=l3 .+ 1)
ee = plot(e1,e2,layout=(1,2))

wsave("embedding.png", ee)
