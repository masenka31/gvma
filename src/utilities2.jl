using BSON
using CSV
using DataFrames
using JSON
using JsonGrinder
using Mill
using ThreadTools

read_json(file) = JSON.parse(read(file, String))

struct Dataset3{T, S, W}
    dir::String
    samples::Vector{String}
    type::Vector{T}
    severity::Vector{S}
    strain::Vector{W}
    schema
    extractor
end

function Dataset3(
    file_csv::String,
    file_schema::String;
    dir = joinpath(dirname(file_csv), "samples")
)

    df = CSV.read(file_csv, DataFrame)
    schema = BSON.load(file_schema)[:schema]
    extractor = suggestextractor(schema)
    return Dataset3(abspath(dir), Vector(df.sha256), Vector(df.type), Vector(df.severity), schema, extractor)
end

Base.show(io::IO, d::Dataset3) = print(io, "Dataset with $(length(d.samples)) samples.")

function Base.getindex(d::Dataset3, inds)
    files = joinpath.(d.dir, string.(d.samples[inds], ".json"))
    data = reduce(catobs, tmap(x -> d.extractor(read_json(x)), files))
    type, severity = d.type[inds], d.severity[inds]
    return data, type, severity
end