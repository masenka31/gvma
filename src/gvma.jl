module gvma

using Plots
using StatsBase
using JsonGrinder

include("plotting.jl")
include("flattening.jl")
include("utilities.jl")
include("data.jl")
include("knn.jl")

export scatter2, scatter2!
export flatten_json, read_json
export Dataset
export dist_knn
export train_test_split

end # module