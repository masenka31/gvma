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

function load_dataset(small=false)
    # load strain dataset
    dataset = Dataset(datadir("samples_strain.csv"), datadir("schema.bson"), datadir("samples_strain"))
    X, type, y = dataset[:]
    
    if small
        # load small classes dataset
        dataset_s = Dataset(datadir("samples_small.csv"), datadir("schema.bson"), datadir("samples_small"))
        Xs, type_s, ys = dataset_s[:]
        return (X, y, unique(y)), (Xs, ys, unique(ys))
    else
        return X, y, unique(y)
    end
end

# X, y, labelnames = load_dataset()
(X, y, labelnames), (Xs, ys, labelnames_s) = load_dataset(true)

@info "Data loaded and prepared."