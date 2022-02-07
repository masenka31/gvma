using JSON 
import Base.==

const DictOrVector = Union{Dict, Vector}

"""
    _flatten2string(sample)

    convert JSON to a list of key-values as a string
"""
function _flatten2string(sample::Dict, add_delimiter=true)
    delimiter = add_delimiter ? "," : ""
    isempty(keys(sample)) && return("")
    mapreduce(vcat, collect(keys(sample))) do k 
        childs = _flatten2string(sample[k], add_delimiter)
        isempty(childs) && return(childs)
        map(c -> delimiter * string(JSON.json(k)) * string(c), childs)
    end |> unique |> x -> filter(!isempty, x)
end

function _flatten2string(sample::Vector, add_delimiter=true)
    isempty(sample) && return("")
    delimiter = add_delimiter ? "," : ""
    r = mapreduce(s -> _flatten2string(s, add_delimiter), vcat, sample)
    r = map(c -> delimiter * "[]" * string(c), unique(r))
    filter(!isempty, r)
end
_flatten2string(::Missing, add_delimiter=true) = ""

function _flatten2string(x::T, add_delimiter=true)  where T <: Union{Number, String}
    x = JSON.json(x) |> string
    x = add_delimiter ? "," * x : x
    [x]
end

flatten2string(x) = _flatten2string(x, false)

"""
    function flatten_json(sample::DictOrVector, others...)

    Convert the structure (or the same structures) into 
    a vector of key-value pairs.

    see `unflatten` for the inverse operation

for example
```
julia> a = Dict(:a => [1,2,3], :b => "hello", :c => Dict(:d => "daemon"))
Dict{Symbol, Any} with 3 entries:
  :a => [1, 2, 3]
  :b => "hello"
  :c => Dict(:d=>"daemon")

julia> flatten_json(a)
5-element Vector{Pair{Tuple{Symbol, Tuple}, Tuple{Any}}}:
  (:a, (1, ())) => 1
  (:a, (2, ())) => 2
  (:a, (3, ())) => 3
       (:b, ()) => "hello"
 (:c, (:d, ())) => "daemon"
```

    Flattening multiple structures is supported
```
julia> flatten_json(a,a)
5-element Vector{Pair{Tuple{Symbol, Tuple}, Tuple{Any, Any}}}:
  (:a, (1, ())) => (1, 1)
  (:a, (2, ())) => (2, 2)
  (:a, (3, ())) => (3, 3)
       (:b, ()) => ("hello", "hello")
 (:c, (:d, ())) => ("daemon", "daemon")
 ```
"""
function flatten_json(sample::DictOrVector)
    isempty(keys(sample)) && return(Vector{Pair}())
    mapreduce(vcat, collect(keys(sample))) do k 
        childs = flatten_json(sample[k])
        isempty(childs) && return(childs)
        map(childs) do c 
            (k, c...)
        end
    end
end

flatten_json(sample) = [(sample,)]

"""
  Summary
  ≡≡≡≡≡≡≡≡≡

  JsonTerm{T, V} holds one key / value / index in the json. The type of the key
  is held in T, which has one of three values :key, :index, :leaf

  Fields
  ≡≡≡≡≡≡≡≡
  v::V        value of the element

"""
struct JsonTerm{T, V}
  v::V
end


IndexTerm = JsonTerm{:index, <:Any}

JsonTerm(T::Symbol, v::V) where {V} = JsonTerm{T,V}(v)

Base.show(io::IO, t::IndexTerm) = print(io, "[", t.v, "]")
Base.show(io::IO, t::JsonTerm) = print(io, t.v)
Base.isless(a::JsonTerm{T,V}, b::JsonTerm{T,V}) where {T,V} = a.v < b.v
Base.isless(a::JsonTerm{T1,<:Any}, b::JsonTerm{T2,<:Any}) where {T1,T2} = T1 < T2
value(t::JsonTerm) = t.v 
value(t) = t

==(a::IndexTerm, b::IndexTerm) = true
==(a::JsonTerm, b::JsonTerm) = false
==(a::JsonTerm{T,V}, b::JsonTerm{T,V}) where {T,V} = a.v == b.v
Base.hash(a::IndexTerm, seed::UInt64) = return(hash(seed))
Base.hash(a::JsonTerm, seed::UInt64) = hash(a.v, seed)

"""
  Summary
  ≡≡≡≡≡≡≡≡≡

  JsonPath{T} holds one key or value in the json. JsonTerm 

  Fields
  ≡≡≡≡≡≡≡≡
  p::T
"""
struct JsonPath{T}
  p::T
end

Base.show(io::IO, jp::JsonPath) = print(io, join(jp.p, "."))
Base.length(jp::JsonPath) = length(jp.p)
Base.only(jp::JsonPath) = only(jp.p)
Base.getindex(jp::JsonPath, i::Int) = jp.p[i]
Base.getindex(jp::JsonPath, i) = JsonPath(jp.p[i])
Base.lastindex(jp::JsonPath) = length(jp)
Base.first(jp::JsonPath) = jp[1]
Base.tail(jp::JsonPath) = jp[2:end]
Base.hash(jp::JsonPath, seed::UInt64) = foldl((seed, t) -> hash(t, seed), jp.p, init = seed)

function Base.isless(a::JsonPath, b::JsonPath)
  for i in 1:min(length(a.p), length(b.p))
    a[i] > b[i] && return(false)
  end
  return(true)
end

function ==(a::JsonPath, b::JsonPath)
  length(a) != length(b) && return(false)
  for i in 1:length(a)
    a[i] != b[i] && return(false)
  end
  return(true)
end

struct FlatJson{T}
  paths::T
  function FlatJson(paths::T) where {T}
    new{T}(sort(paths))
  end
end

Base.isempty(a::FlatJson) = isempty(a.paths)
Base.length(a::FlatJson) = length(a.paths)
Base.unique(a::FlatJson) = FlatJson(unique(a.paths))
Base.intersect(a::FlatJson, b::FlatJson) = FlatJson(intersect(a.paths, b.paths))
Base.setdiff(a::FlatJson, b::FlatJson) = FlatJson(setdiff(a.paths, b.paths))
Base.symdiff(a::FlatJson, b::FlatJson) = FlatJson(symdiff(a.paths, b.paths))

function _flatten_json(t::Symbol, sample::DictOrVector)
    isempty(keys(sample)) && return(Vector{Pair}())
    mapreduce(vcat, collect(keys(sample))) do k 
        childs = flatten_json(t, sample[k])
        isempty(childs) && return(childs)
        map(childs) do c 
            (JsonTerm(t, k), c...)
        end
    end
end

flatten_json(t::Symbol, sample::Vector) = _flatten_json(:index, sample)
flatten_json(t::Symbol, sample::Dict) = _flatten_json(:key, sample)
flatten_json(t::Symbol, sample) = [(JsonTerm(:leaf, sample),)]

FlatJson(js::Dict) = FlatJson(map(JsonPath, flatten_json(:key, js)))

unflatten_json(fj::FlatJson) = unflatten_json(fj.paths)


"""
    function unflatten_json(d::Vector{<:Pair})

    An inverse operation to flattening.

```
julia> a = Dict(:a => [1,2,3], :b => "hello", :c => Dict(:d => "daemon"))
Dict{Symbol, Any} with 3 entries:
  :a => [1, 2, 3]
  :b => "hello"
  :c => Dict(:d=>"daemon")

julia> flatten_json(a)
5-element Vector{Tuple{Symbol, Any, Vararg{Any, N} where N}}:
 (:a, 1, 1)
 (:a, 2, 2)
 (:a, 3, 3)
 (:b, "hello")
 (:c, :d, "daemon")

julia> unflatten_json(ans)
Dict{Symbol, Any} with 3 entries:
  :a => [1, 2, 3]
  :b => "hello"
  :c => Dict(:d=>"daemon")
```

"""
function unflatten_json(d::Vector)
    @assert length(d) > 0
    if all(length(k) == 1 for k in d)
        @assert length(d) == 1
        return(value(only(only(d))))   #this is bizzare, yet it make sense as it is an array containing single tuple
    end

    s = Dict{typeof(value(d[1][1])), Vector{Int}}()
    for (i, ks) in enumerate(d)
        k = value(first(ks))
        s[k] = push!(get(s,k, Vector{Int}()), i)
    end
    if (d[1][1] isa IndexTerm) || (d[1][1] isa Integer)
        map(sort(collect(keys(s)))) do i
          ii = s[i]
          unflatten_json(map(Base.tail, @view d[ii]))
        end
    else
        Dict(k => unflatten_json(map(Base.tail, @view d[ii])) for (k,ii) in s)
    end
end
