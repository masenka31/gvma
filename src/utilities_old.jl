using BSON
using CSV
using DataFrames
using JSON
using JsonGrinder
using Mill
using ThreadTools

read_json(file) = JSON.parse(read(file, String))

struct Dataset{T}
    dir::String
    samples::Vector{String}
    targets::Vector{T}
    schema
    extractor
end

function Dataset(
    file_csv::String,
    file_schema::String;
    dir = joinpath(dirname(file_csv), "samples")
)

    df = CSV.read(file_csv, DataFrame)
    schema = BSON.load(file_schema)[:schema]
    extractor = suggestextractor(schema)
    return Dataset(abspath(dir), Vector(df.sha256), Vector(df.type), schema, extractor)
end

Base.show(io::IO, d::Dataset) = print(io, "Dataset with $(length(d.samples)) samples")

function Base.getindex(d::Dataset, inds)
    files = joinpath.(d.dir, string.(d.samples[inds], ".json"))
    data = reduce(catobs, tmap(x -> d.extractor(read_json(x)), files))
    targets = d.targets[inds]
    return data, targets
end