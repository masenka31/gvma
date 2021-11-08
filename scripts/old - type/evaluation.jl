using DrWatson
using Mill, Flux, BSON, DataFrames
include("utilities.jl")
using Plots
ENV["GKSwstype"] = "100"

using UMAP
include("plotting.jl")

# load results and best model
df = collect_results(datadir("classifier"))
sort!(df, :train_accuracy, rev=true)
best_model = df[1,:model]

# load dataset
dataset = Dataset("./data/samples.csv", "./data/schema.bson")
# create data, labels
labelnames = unique(dataset.targets)
@time X, y = dataset[:]

# get the ProductModel
pm = best_model[1]
enc = pm(X).data

for a in 1:2:15
    for b in 2:4:20
        p = scatter2(enc, a, b; zcolor=encode(y, labelnames), label="",color=:tab10)
        wsave("encoding/enc_x=$(a)_y=$(b).png", p)
    end
end

emb = umap(enc, 2)
p = scatter2(emb; zcolor=encode(y, labelnames), label="",color=:tab10)
wsave("encoding/umap_x=$(a)_y=$(b).png", p)


# validation data
tix, vix = df[1, :indices]
X_val, y_val = X[vix], y[vix]

enc_v = pm(X_val).data
for (a,b) in [(1,2), (1,14), (5,10)]
    println(a, ", ", b)
    p = scatter2(enc_v, a, b; zcolor=encode(y_val, labelnames), label="",color=:tab10)
    wsave("encoding/val_x=$(a)_y=$(b).png", p)
end

for nn in [5,15,30,50]
    emb_v = umap(enc_v, 2, n_neighbors=nn)
    p = scatter2(emb_v; zcolor=encode(y_val, labelnames), label="",color=:tab10, size=(800,800),aspect_ratio=:equal,markersize=3,markerstrokewidth=0,opacity=0.7)
    wsave("encoding/umap_val_n=$nn.png", p)
end