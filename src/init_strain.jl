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

using Plots
ENV["GKSwstype"] = "100"

# load dataset
dataset = Dataset(datadir("samples_strain.csv"), datadir("schema.bson"))
X, type, y = dataset[:]
#using BSON
#using InlineStrings
#@unpack X, y = BSON.load(datadir("Xy.bson"))

# create data, labels
labelnames = unique(y)
@info "Data loaded and prepared."