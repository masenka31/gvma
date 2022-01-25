using DrWatson
@quickactivate

using gvma
using Flux
using Plots

trimmed_acos(x) = acos(σ(x))
trimmed_acos.(randn(10))

x1 = randn(2,200)
x2 = randn(2,300) .+ [-1.7,3.5]
X = hcat(x1,x2)
y = vcat(zeros(Int, 200), ones(Int, 300))

x3 = randn(2,250) .- [2.9,4.1]
y3 = ones(Int, 250) .+ 1

Xf = hcat(X, x3)
yf = vcat(y, y3)

# simple classifier
yoh = Flux.onehotbatch(y, [0,1])

using Flux: logitcrossentropy
lossf(x, y) = logitcrossentropy(W*x, y)
# W = randn(2,2)
W = kaiming_uniform(2,2)
lossf(X,yoh)

opt = Descent()
for i in 1:10
    Flux.train!(lossf, Flux.params(W), Flux.Data.DataLoader((X, yoh), batchsize=32), opt)
    @show lossf(X,yoh)
end

scatter2(W*X, color=y)
scatter2(W*Xf, color=yf)
scatter2(Xf, color=yf)

"""
function lossf(x, y)
    Wn = W ./ sqrt.(sum(W .^2, dims=2))
    logitcrossentropy(Wn*x, y)
end
"""
# what if - normalization?
# how do I do normalization?
normalizeW(W::AbstractMatrix) = W ./ sqrt.(sum(W .^2, dims=2))
normalizes(x::AbstractVector, s::Real) = x ./ sqrt(sum(abs2,x)) * s
normalizes(x::AbstractMatrix, s::Real) = x ./ sqrt.(sum(x .^ 2, dims=1)) .* s

s = 3
costheta(W, x) = normalizeW(W) * normalizes(x, s)

L(x, y) = logitcrossentropy(costheta(W, x), y)

using Flux: kaiming_uniform
W = kaiming_uniform(2,2)
opt = ADAM(10e-2)

L(X[:,1], yoh[:,1])

for i in 1:10
    Flux.train!(L, Flux.params(W), Flux.Data.DataLoader((X, yoh), batchsize=32), opt)
    @show L(X, yoh)
end

scatter2(costheta(W, X), color=y, label="")
scatter2(costheta(W, Xf), color=yf, label="")

scatter2(normalize(W)*X, color=y, label="")
scatter2(normalize(W)*Xf, color=yf, label="")
savefig("costheta.png")


tx1 = randn(2,200)
tx2 = randn(2,300) .+ [1.2,3.5]
tX = hcat(tx1,tx2)

scatter2(costheta(W, tX), color=y)

function logsumexp(x::AbstractArray; dims = :)
    max_ = maximum(x; dims = dims)
    max_ .+ log.(sum(exp.(x .- max_); dims = dims))
end
function sumexp(x::AbstractArray; dims = :)
    max_ = maximum(x; dims = dims)
    max_ .+ sum(exp.(x .- max_); dims = dims)
end

x = randn(10)
logsumexp(x)

x = randn(2)
cs = costheta(W, x)
softmax(cs)

softmax_margin(x) = exp.(x .- m) ./ sum(exp.(x))

using Flux: crossentropy
one_softmax_margin(x, ix, m = 1f0) = exp(x[ix] - m) ./ (exp(x[ix] - m) + sum(exp.(x[Not(ix)])))
all_softmax_margin(x, m) = hcat(map(col -> map(i -> one_softmax_margin(col, i, m), 1:length(col)), eachcol(x))...)
ls(x, y) = crossentropy(all_softmax_margin(costheta(W, x), 1.0), y)

s = 3.0
costheta(W, x) = normalizeW(W) * normalizes(x, s)
W = kaiming_uniform(2,2)
opt = ADAM()

for i in 1:300
    Flux.train!(ls, Flux.params(W), Flux.Data.DataLoader((X, yoh), batchsize=32), opt)
    mod(i,10) == 1 ? println("epoch $i: ", ls(X, yoh)) : nothing
end

scatter2(costheta(W, X), color=y, label="")
scatter2(costheta(W, Xf), color=yf, label="", opacity=0.5)

scatter2(W*X, color=y, label="")
scatter2(W*Xf, color=yf, label="")

scatter2(Xf, color=yf, label="")