module gvma

using Plots
using StatsBase
using JsonGrinder
using Flux
using Distances

include("utils.jl")
#include("flattening.jl")
include("dataset.jl")
include("data2.jl")
include("knn.jl")
include("constructors.jl")
#include("triplet.jl")

export scatter2, scatter2!
export flatten_json, read_json
export Dataset
export dist_knn
export train_test_split
export train_test_split_reg, train_test_split_ix

export classifier_constructor, triplet_mill_constructor

end # module