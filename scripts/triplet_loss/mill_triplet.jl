using DrWatson
@quickactivate

using gvma
using Flux
using StatsBase: sample
using LinearAlgebra: norm
using Statistics: mean

include(srcdir("init_strain.jl"))

using gvma: Triplet, TripletModel, sample_triplet, triplet_loss


(Xtrain, ytrain), (Xtest, ytest) = train_test_split(X, y; seed = 1, ratio = 0.5)
model = triplet_mill_constructor(Xtrain, 5, relu, meanmax_aggregation, 1; odim=2)

tr = sample_triplet(Xtrain, ytrain)

minibatch(;batchsize=8) = [sample_triplet(Xtrain, ytrain) for i in 1:batchsize]
batch = minibatch()

opt = ADAM()

loss(x::Triplet) = triplet_loss(model, x)
loss(x::Vector) = mean(map(xi -> loss(xi), x))

batch = minibatch()
loss(batch)

for i in 1:10
    batch = minibatch()
    Flux.train!(loss, Flux.params(model), batch, opt)
    if mod(i,1) == 0
        @show loss(batch)
    end
end

# clustering power evaluation
using Clustering, Distances
D = pairwise(model, x)
Dsimd = _simd_pairwise(model, x)
c2 = kmedoids(D, 10)
labels = gvma.encode(ytrain, labelnames)
randindex(c2,labels)

enc = hcat(map(i -> model.f(Xtrain[i]), 1:nobs(Xtrain))...)

scatter2(enc, zcolor=labels, color=:tab10)
savefig("enc_mill.png")