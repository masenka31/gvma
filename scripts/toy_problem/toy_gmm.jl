using DrWatson
using Distributions, GaussianMixtures

using KernelDensity
using Plots, StatsPlots
ENV["GKSwstype"] = "100"
include("/home/maskomic/projects/gvma/src/plotting.jl")

function toyproblem(;bagsize = 100, minorities = 1, nbags = 100)
	x = randn(Float64, 2,bagsize*nbags)
	x[:,1:2:end] = x[:,1:2:end] .* [0.5, 2] .+ [3,0]
	x[:,2:2:end] = x[:,1:2:end] .* [2, 0.5] .- [0,3]
	x[:,1:100:end] = randn(Float64, 2, nbags) ./ 2 .+ [6,6]
	bags = [((i-1)*bagsize+1):i*bagsize for i in 1:nbags]
	[x[:,bags[i]] for i in 1:100]
end
function toyproblem(;bagsize = 100, minorities = 1, nbags = 100)
	x = randn(Float64, 2,bagsize*nbags)
	x[:,1:2:end] = x[:,1:2:end] .* [0.5, 2] .+ [3,0]
	x[:,2:2:end] = x[:,1:2:end] .* [2, 0.5] .- [0,3]
	x[:,1:200:end] = randn(Float64, 2, nbags÷2) ./ 2 .+ [6,6]
    x[:,1:201:end] = randn(Float64, 2, nbags÷2) ./ 2 .+ [6,6]
	bags = [((i-1)*bagsize+1):i*bagsize for i in 1:nbags]
	[x[:,bags[i]] for i in 1:100]
end

data = toyproblem()
X = hcat(data...)

scatter2(X)
savefig("data.png")

g = GMM(9, collect(X'), kind=:full)
for i in 1:100
    em!(g, collect(data[i]'), nIter=1)
end

mm = MixtureModel(g)
plot_contour(mm, X)
savefig("contour.png")

Y = rand(mm, 10000)
contour(kde(Y'))
savefig("kernel_density.png")

### two gaussian mixture model
function em2g(g1,g2,data;niter=20)
    for k in 1:niter
        m1 = MixtureModel(g1);
        m2 = MixtureModel(g2);

        l1 = map(i -> sum(logpdf(m1,data[i])), 1:100);
        l2 = map(i -> sum(logpdf(m2,data[i])), 1:100);
        b = l1 .> l2;
        d1 = hcat(data[b]...);
        d2 = hcat(data[b .== 0]...);

        em!(g1, collect(d1'), nIter=1);
        em!(g2, collect(d2'), nIter=1);
    end
    m1 = MixtureModel(g1)
    m2 = MixtureModel(g2)
    return m1,m2
end

g1 = GMM(3, collect(X[:, 1:5000]'), kind=:full)
g2 = GMM(3, collect(X[:, 5001:end]'), kind=:full)
m1,m2 = em2g(g1,g2,data)
l1 = map(i -> sum(logpdf(m1,data[i])), 1:100);
l2 = map(i -> sum(logpdf(m2,data[i])), 1:100);
b = l1 .> l2
sum(b)

scatter2(hcat(data[b .== 1]...),opacity=0.5)
contour!(kde(rand(m1,10000)'),levels=40,color=:jet)
savefig("m1.png")

scatter2(hcat(data[b .== 0]...),opacity=0.5)
contour!(kde(rand(m2,10000)'),levels=40,color=:jet)
savefig("m2.png")

plot_contour(m1, hcat(data[b .== 1]...))
savefig("mc1.png")

plot_contour(m2, hcat(data[b .== 0]...))
savefig("mc2.png")