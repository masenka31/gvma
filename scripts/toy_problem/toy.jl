using SumProductTransform, Unitary, Flux, Setfield, Mill, Distributions
using Flux: throttle
using SumProductTransform: fit!
using SumProductTransform: ScaleShift, SVDDense
using DistributionsAD: TuringMvNormal

using Plots
ENV["GKSwstype"] = "100"

function toyproblem(;bagsize = 100, minorities = 1, nbags = 100)
	x = randn(Float64, 2,bagsize*nbags)
	x[:,1:2:end] = x[:,1:2:end] .* [0.5, 2] .+ [3,0]
	x[:,2:2:end] = x[:,1:2:end] .* [2, 0.5] .- [0,3]
	x[:,1:100:end] = randn(Float64, 2, nbags) ./ 2 .+ [6,6]
	bags = [((i-1)*bagsize+1):i*bagsize for i in 1:nbags]
	BagNode(ArrayNode(x), bags)
end
function toyproblem(;bagsize = 100, minorities = 1, nbags = 100)
	x = randn(Float64, 2,bagsize*nbags)
	x[:,1:2:end] = x[:,1:2:end] .* [0.5, 2] .+ [3,0]
	x[:,2:2:end] = x[:,1:2:end] .* [2, 0.5] .- [0,3]
	x[:,1:100:end] = randn(Float64, 2, nbags) ./ 2 .+ [6,6]
	bags = [((i-1)*bagsize+1):i*bagsize for i in 1:nbags]
	[x[:,bags[i]] for i in 1:100]
end

function plot_contour(m, x, title = nothing)
	levels = quantile(exp.(logpdf(m, x)), 0.001:0.09:0.99)
	δ = levels[1] / 20
	levels = vcat(collect(levels[1] - 10δ:δ:levels[1] - δ), levels)
	xr = range(minimum(x[1,:]) - 1 , maximum(x[1,:])+ 1 , length = 200)
	yr = range(minimum(x[2,:]) - 1 , maximum(x[2,:])+ 1 , length = 200)
	p1 = contour(xr, yr, (x...) ->  exp(logpdf(m, [x[1],x[2]])[1]))
	p2 = deepcopy(p1)
	xx = x[:,sample(1:size(x,2), 100, replace = false)]
	scatter!(p2, x[1,:], x[2,:], alpha = 0.4)
	p = plot(p1, p2)
	!isnothing(title) && title!(p, title)
	p
end

function gmm(d, n, unitary = :butterfly)
  SumNode([TransformationNode(SVDDense(d, identity, unitary), TuringMvNormal(d, 1f0)) for i in 1:n])
end


###############################################################################
#			Fitting Instances
###############################################################################
xtrn = toyproblem()

model = gmm(2, 9)
history = fit!(model, xtrn.data.data, 100, 20000, 0; gradmethod = :exact, minimum_improvement = -1e10, opt = ADAM())
p₁ = plot_contour(model, xtrn, "gmm")

###############################################################################
#			Fitting Instances
###############################################################################
using IterTools
model = f64(gmm(2, 9))
function bagloss(model, x) 
	l = reshape(logpdf(model, x.data.data), 1, :)
	o = Mill.segmented_mean_forw(l, [0.0], x.bags, nothing) - Mill.segmented_max_forw(-l, [0.0], x.bags)
	-mean(o)
end

Flux.Optimise.train!((x...) -> bagloss(model, x...), Flux.params(model), repeatedly(() -> (xtrn[sample(1:100,10, replace = false)],), 20000), ADAM())
p₁ = plot_contour(model, xtrn.data.data)

###############################################################################
#           Fitting MGMM model
###############################################################################

function toyproblem(;bagsize = 100, minorities = 1, nbags = 100)
	x = randn(Float64, 2,bagsize*nbags)
	x[:,1:2:end] = x[:,1:2:end] .* [0.5, 2] .+ [3,0]
	x[:,2:2:end] = x[:,1:2:end] .* [2, 0.5] .- [0,3]
	x[:,1:100:end] = randn(Float64, 2, nbags) ./ 2 .+ [6,6]
	bags = [((i-1)*bagsize+1):i*bagsize for i in 1:nbags]
	[x[:,bags[i]] for i in 1:100]
end

"""
	loss(m::MGMM, x)

Returns the negative log-likelihood of a bag as a sum of instance log-likelihoods.
"""
function loss(m::MGMM, x)
    MM = toMixtureModel(m)
    -sum(logpdf(MM, x))
end

lossf(x) = loss(tr_model, x)
lossf(x)

parameters = (K = 9, T = 2)
tr_x = toyproblem(;minorities=5)
tr_model = MGMM_constructor(tr_x; parameters...)
x = tr_x[1]

best_likelihood = Inf
best_model = deepcopy(tr_model)
# training loop
for i in 1:100

    # do EM for MGMM
    tr_model = star(tr_x, tr_model)

    if isnan(tr_model)
        @warn "NaN alpha values. Training stopped."
        break
    end

    tr_model.β = beta_star(tr_x, tr_model)
    train_loss = mean(lossf.(tr_x))
    @info train_loss
    if train_loss < best_likelihood
        best_likelihood = train_loss
        best_model = deepcopy(tr_model)
    end
end

using KernelDensity
using StatsPlots

best_model.χ
MM = toMixtureModel(best_model)
Y = rand(MM, 2000)
scatter2(hcat(tr_x...),opacity=0.5)
contour!(kde(Y'),levels=30)
savefig("contour_best.png")

contour(kde(Y'),levels=30)
savefig("contour_best.png")


tr_model.χ
MM = toMixtureModel(tr_model)
Y = rand(MM, 2000)
scatter2(hcat(tr_x...),opacity=0.5)
contour!(kde(Y'),levels=30)
savefig("contour.png")

contour(kde(Y'),levels=30)
savefig("contour.png")