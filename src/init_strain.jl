using DrWatson
@quickactivate

using gvma

using JsonGrinder
using Mill
using Statistics
using Flux
using Flux: throttle, @epochs
using StatsBase
using Base.Iterators: repeated

using UMAP, Plots
ENV["GKSwstype"] = "100"

# load dataset
dataset = Dataset(datadir("samples_strain.csv"), datadir("schema.bson"))
X, type, y = dataset[:]

# create data, labels
labelnames = unique(dataset.strain)
@info "Data loaded and prepared."