module gvma

using Plots
using StatsBase
using JsonGrinder
using Flux

include("plotting.jl")
include("flattening.jl")
include("utilities.jl")
include("data.jl")
include("knn.jl")
include("classifier.jl")

export scatter2, scatter2!
export flatten_json, read_json
export Dataset
export dist_knn
export train_test_split
export train_test_split_reg, train_test_split_ix

export classifier_constructor

end # module