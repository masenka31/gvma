using BSON
using CSV
using DataFrames
using JSON
using JsonGrinder
using Mill
using ThreadTools

read_json(file) = JSON.parse(read(file, String))

##############################
### Initial strain dataset ###
##############################

struct Dataset{T, S}
    dir::String
    samples::Vector{String}
    type::Vector{T}
    strain::Vector{S}
    schema
    extractor
end

function Dataset(
    file_csv::String,
    file_schema::String,
    folder::String
)

    dir = joinpath(dirname(file_csv), folder)
    df = CSV.read(file_csv, DataFrame)
    schema = BSON.load(file_schema)[:schema]
    extractor = suggestextractor(schema)
    return Dataset(abspath(dir), Vector(df.sha256), Vector(df.type), Vector(df.strain), schema, extractor)
end

Base.show(io::IO, d::Dataset) = print(io, "Dataset with $(length(d.samples)) samples.")

function Base.getindex(d::Dataset, inds)
    files = joinpath.(d.dir, string.(d.samples[inds], ".json"))
    data = reduce(catobs, tmap(x -> d.extractor(read_json(x)), files))
    type, strain = d.type[inds], d.strain[inds]
    return data, type, strain
end

################################
### Time-data strain dataset ###
################################

struct TDataset
    samples
    strain
    timestamp
    type
    schema
    extractor
end

using JSON, HDF5
using ProgressMeter
using DrWatson

"""
function load_samples()
    jsons = h5read(datadir("timedata","samples.h5"),"samples")
    n = length(jsons)
    dicts = Vector{Dict}(undef, n)

    p = Progress(n)
    Threads.@threads for i in 1:n
        dicts[i] = JSON.parse(jsons[i])
        next!(p)
    end
    return dicts
end
"""

function TDataset()
    # load samples, schema, extractor
    df = CSV.read(datadir("timedata/samples.csv"), DataFrame)
    sch = BSON.load(datadir("schema.bson"))[:schema]
    extractor = suggestextractor(sch)

    # load samples
    d = load_samples()

    # return the dataset structure
    return TDataset(d, String.(df.strain), df.timestamp, df.type, sch, extractor)
end

Base.show(io::IO, d::TDataset) = print(io, "Dataset with $(length(d.samples)) observations.")

function Base.getindex(d::TDataset, inds)
    data = reduce(catobs, tmap(x -> d.extractor(x), d.samples[inds]))
    type, strain, timestamp = d.type[inds], d.strain[inds], d.timestamp[inds]
    return data, strain, timestamp, type
end


load_extractor(file::String) = BSON.load(file)[:extractor]

function load_jsons(file::String, inds::Union{AbstractRange,Colon})
    return h5open(file, "r") do data
        data["samples"][inds]
    end
end

function load_jsons(file::String, inds)
    return h5open(file, "r") do data
        [data["samples"][i] for i in inds]
    end
end

function load_samples(file::String, extractor; inds = :)
    jsons = load_jsons(file, inds)
    n = length(jsons)
    dicts = Vector{ProductNode}(undef, n)

    p = Progress(n)
    Threads.@threads for i in 1:n
        dicts[i] = extractor(JSON.parse(jsons[i]))
        next!(p)
    end
    return reduce(catobs, dicts)
end

struct Info
    strain
    timestamp
    type
    schema
    extractor
end

function InfoDataset()
    # load samples, schema, extractor
    df = CSV.read(datadir("timedata/samples.csv"), DataFrame)
    sch = BSON.load(datadir("schema.bson"))[:schema]
    extractor = suggestextractor(sch)

    # return the dataset structure
    return Info(String.(df.strain), df.timestamp, df.type, sch, extractor)
end

struct TimeDataset{T<:AbstractNode}
    data::T
    strain
    timestamp
end

function TimeDataset(d::Info, inds)
    samples = load_samples(datadir("timedata/samples.h5"), d.extractor; inds = inds)
    strain = d.strain[inds]
    timestamp = d.timestamp[inds]
    TimeDataset(samples, strain, timestamp)
end

function Base.getindex(d::TimeDataset, inds)
    data, strain, timestamp = d.data[inds], d.strain[inds], d.timestamp[inds]
    return data, strain, timestamp
end

Base.show(io::IO, d::TimeDataset) = print(io, "Dataset with $(nobs(d.data)) observations.")