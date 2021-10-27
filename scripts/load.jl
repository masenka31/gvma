using DrWatson
@quickactivate

include("utilities.jl")
using UMAP, Plots
include("plotting.jl")
using Statistics
using UMAP
using Flux

dataset = Dataset("./data/samples.csv", "./data/schema.bson")
#x, y = dataset[1:10]

# create data, labels
data = tmap(dataset.extractor, dataset.samples)
labelnames = unique(dataset.targets)
@time X, y = dataset[:]
labels = zeros(Int, 8000)
for i in 1:8000
    idx = findall(x -> x == 1, y[i] .== labelnames)
    labels[i] = idx[1]
end

m = reflectinmodel(
    X,
    d -> Dense(d, 10, tanh),
    d -> meanmax_aggregation(d)
)
enc = m(X).data
p = scatter2(enc; color=labels);
display("image/png", p)

embedding = umap(enc, 2)
p = scatter2(embedding; color=labels, markerstrokewidth=0.2, color_palette=:seaborn_bright, opacity=0.5);
display("image/png", p)


# první část modelu odpovídající behavior_summary
x = X[200]
function ex(x)
    a = m.ms[:behavior_summary](x[:behavior_summary])
    b = m.ms[:signatures](x[:signatures])
    c = m.ms[:network_http](x[:network_http])
    mean(hcat(a,b,c).data,dims=2)
end

vectors = map(i -> ex(X[i]), 1:8000)
matrix = hcat(vectors...)

new_emb = umap(matrix, 2)
scatter(new_emb[1,:], new_emb[2,:], zcolor=labels, color=:jet)

using GenerativeModels
using Flux
using StatsBase
include("vae.jl")

x = X[sample(1:8000)]
mil_model = reflectinmodel(X)
vae = vae_constructor(;idim=10, zdim=8, activation = "swish", hdim=32, nlayers=3)
loss(x) = - elbo(vae,mil_model(x).data)
loss(x)
opt = ADAM()
ps = Flux.params(vae, mil_model)

Flux.train!(loss, ps, [x], opt)
for i in 1:100
    batch = map(i -> X[i], sample(1:8000, 64, replace=false));
    Flux.train!(loss, ps, batch, opt)
    println("Epoch $i finished.")
end

enc = mil_model(X).data
embedding = umap(enc, 2)
scatter(embedding[1,:], embedding[2,:], zcolor=labels, color=:jet, markersize=2, markerstrokewidth=0)
scatter(enc[4,:], enc[2,:], zcolor=labels, color=:jet, markersize=2, markerstrokewidth=0)

latent = latent_space(vae, enc)
scatter(latent[1,:], latent[2,:], zcolor=labels, color=:jet, markersize=2, markerstrokewidth=0)
emb_latent = umap(latent, 2)
scatter(emb_latent[1,:], emb_latent[2,:], zcolor=labels, color=:jet, markersize=2, markerstrokewidth=0)

enc = m(X).data
mapping = umap(enc,2)
scatter2(mapping; color=labels)