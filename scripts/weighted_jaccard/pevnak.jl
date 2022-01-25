"""
This scripts needs Flux@12. Therefore, it is necessary to load
a different than GVMA directory and Manifest using DrWatson.

Do
`] activate`
to activate this environment.
"""

using IterTools
using Dictionaries
using Flux, Flux.Zygote
using ClusterLosses
using JSON
# load data and necessary packages
function flatten_json(sample::Dict)
    isempty(keys(sample)) && return(Vector{Pair}())
    mapreduce(vcat, collect(keys(sample))) do k 
        childs = flatten_json(sample[k])
        isempty(childs) && return(childs)
        map(childs) do c 
            (k, c...)
        end
    end
end

function flatten_json(sample::Vector)
    isempty(keys(sample)) && return(Vector{Pair}())
    mapreduce(vcat, collect(keys(sample))) do k 
        childs = flatten_json(sample[k])
        isempty(childs) && return(childs)
        map(childs) do c 
            (0, c...)
        end
    end
end
flatten_json(sample) = [(sample,)]

function jaccard_distance(d1::Vector, d2::Vector)
    int = length(intersect(d1,d2))
    un = length(union(d1,d2))
    return (un - int)/un
end


js1 = """{ "a" : {"b" : ["v", "p"], "c" : 1}}"""
js2 = """{ "a" : {"b" : "vx", "c" : 2}}"""
js3 = """{ "a" : {"b" : "v", "c" : 2}}"""
js4 = """{ "a" : {"b" : "v", "c" : 2}}"""

#sch = schema([js1,js2,js3,js4])

j1, j2, j3, j4 = JSON.parse.([js1,js2,js3,js4])
# prefixes = [("a", "b"), ("a", "c")]

f1, f2, f3, f4 = map(j -> flatten_json(j), [j1,j2,j3,j4])
fp1, fp2, fp3, fp4 = map(f -> map(j -> (js = j, pr = j[1:end-1]), f),  [f1, f2, f3, f4])
fps = map(f -> Indices(map(j -> (js = j, pr = j[1:end-1]), f)),  [f1, f2, f3, f4])

all_prefixes = mapreduce(f -> map(j -> j.pr, collect(f)), vcat, fps) |> unique

prefix2int = Dict(reverse.(enumerate(all_prefixes)))
w = ones(Float32, length(prefix2int))


function jaccard_distance(j1, j2, prefix2int, w)
    int = Zygote.@ignore map(j -> prefix2int[j.pr],intersect(j1,j2))
    un  = Zygote.@ignore map(j -> prefix2int[j.pr],union(j1,j2))
    # if intersection is empty, automatically return one (maximum)
    isempty(int) && return(one(eltype(w)))
    one(eltype(w)) - sum(w[int]) / sum(w[un])
end

function jaccard_distance(j1::Indices, j2::Indices, prefix2int, w)
    int = Zygote.@ignore map(j -> prefix2int[j.pr], collect(intersect(j1,j2)))
    un  = Zygote.@ignore map(j -> prefix2int[j.pr], collect(union(j1,j2)))
    # if intersection is empty, automatically return one (maximum)
    isempty(int) && return(one(eltype(w)))
    one(eltype(w)) - sum(w[int]) / sum(w[un])
end

"""
Zygote.Buffer makes it possible to differentiate setindex.
What needs to happen is for the matrix to be only changed
once and then it can work
"""

function jpairwise(fps, prefix2int, w)
    d = Zygote.Buffer(zeros(Float32, length(fps), length(fps)))
    for i in 2:length(fps)
        for j in 1:i-1
            v = jaccard_distance(fps[i], fps[j], prefix2int, w)
            d[i,j] = v
            d[j,i] = v
        end
    end
    copy(d)
end
function _jpairwise(fps, prefix2int, w)
    d = zeros(Float32, length(fps), length(fps))
    for i in 2:length(fps)
        for j in 1:i-1
            v = jaccard_distance(fps[i], fps[j], prefix2int, w)
            d[i,j] = v
            d[j,i] = v
        end
    end
    d
end

labels = [0,0,1,1]
gradient(w -> sum(jpairwise(fps, prefix2int, w)), w)[1]
gradient(w) do w 
    loss(Triplet(), jpairwise(fps, prefix2int, w), labels)
end[1]


ps = Flux.params([w])
opt = ADAM()

function prepare_minibatch()
    fps, labels
end

mb_provider = IterTools.repeatedly(prepare_minibatch, 100)

loss(Triplet(), jpairwise(fps, prefix2int, σ.(w) .+ 1f-6), labels)
Flux.train!(ps, mb_provider, opt) do fps, labels
    loss(Triplet(), jpairwise(fps, prefix2int, σ.(w) .+ 1f-6), labels)
end
loss(Triplet(), jpairwise(fps, prefix2int, σ.(w) .+ 1f-6), labels)

### full data ###
using DrWatson
using DataFrames, CSV
df = CSV.read("..\\..\\data\\samples_strain.csv", DataFrame)

files = Vector(df.sha256)
y = Vector(df.strain)
labelnames = unique(y)

function encode(labels, labelnames)
    num_labels = ones(Int, length(labels))
    for i in 1:length(labels)
        v = findall(x -> x == labels[i], labelnames)
        num_labels[i] = v[1]
    end
    return num_labels
end

labels = encode(y, labelnames)
samples = map(f -> joinpath("C:\\Users\\masen\\Desktop\\gvma\\data\\samples_strain", "$f.json"), files)

read_json(file) = JSON.parse(read(file, String))
#files = datadir.("samples_strain", dataset.samples .* ".json");
jsons = read_json.(samples);
jsons_flatten = flatten_json.(jsons);

fps = map(f -> map(j -> (js = j, pr = j[1:end-1]), f),  jsons_flatten)
# fps = map(f -> Indices(map(j -> (js = j, pr = j[1:end-1]), f)),  jsons_flatten)
all_prefixes = mapreduce(f -> map(j -> j.pr, collect(f)), vcat, fps) |> unique

prefix2int = Dict(reverse.(enumerate(all_prefixes)))
w = ones(Float32, length(prefix2int))

@time jpairwise(fps[1:1000], prefix2int, w)

ps = Flux.params([w])
opt = ADAM()

batchsize = 128
function prepare_minibatch()
    ix = rand(1:length(labels), batchsize)
    fps[ix], labels[ix]
end

mb_provider = IterTools.repeatedly(prepare_minibatch, 10)

batch = prepare_minibatch();

lossf(x, y) = loss(Triplet(), jpairwise(x, prefix2int, σ.(w) .+ 1f-6), y)

lossf(batch...)
Flux.train!(ps, mb_provider, opt) do x, y
    loss(Triplet(), jpairwise(x, prefix2int, σ.(w) .+ 1f-6), y)
end
lossf(batch...)

using Flux: @epochs

@epochs 10 begin
    Flux.train!(ps, mb_provider, opt) do x, y
        loss(Triplet(), jpairwise(x, prefix2int, σ.(w) .+ 1f-6), y)
    end
    @show lossf(batch...)
end

L_full_weighted = jpairwise(fps, prefix2int, σ.(w) .+ 1f-6)