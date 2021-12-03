trimmed_acos(x) = acos(σ(x))
trimmed_acos.(randn(10))

x1 = randn(2,200)
x2 = randn(2,300) .+ [1.2,3.5]
X = hcat(x1,x2)
y = vcat(zeros(Int, 200), ones(Int, 300))

# simple classifier
yoh = Flux.onehotbatch(y, [0,1])

using Flux: logitcrossentropy
lossf(x, y) = logitcrossentropy(W*x, y)
lossf(x, y) = logitcrossentropy(normalize(W)*x, y)
# W = randn(2,2)
W = kaiming_uniform(2,2)
lossf(X,y)

opt = Descent()
for i in 1:10
    Flux.train!(lossf, Flux.params(W), Flux.Data.DataLoader((X, y), batchsize=32), opt)
    @show lossf(X,y)
end

# what if - normalization?
# how do I do normalization?
normalizeW(W::AbstractMatrix) = W ./ sqrt.(sum(W .^2, dims=2))
lossf(x, y) = logitcrossentropy(normalizeW(W)*x, y)
"""
function lossf(x, y)
    Wn = W ./ sqrt.(sum(W .^2, dims=2))
    logitcrossentropy(Wn*x, y)
end
"""

normalizes(x::AbstractVector, s::Real) = x ./ sqrt(sum(abs2,x)) * s
normalizes(x::AbstractMatrix, s::Real) = x ./ sqrt.(sum(x .^ 2, dims=1)) .* s

s = 2
costheta(W, x) = normalizeW(W) * normalizes(x, s)
costheta(W, X)

L(x, y) = logitcrossentropy(costheta(W, x), y)

W = kaiming_uniform(2,2)
opt = ADAM(10e-2)
for i in 1:10
    Flux.train!(L, Flux.params(W), Flux.Data.DataLoader((X, yoh), batchsize=32), opt)
    @show L(X, yoh)
end

scatter2(costheta(W, X), color=y)
savefig("costheta.png")

x