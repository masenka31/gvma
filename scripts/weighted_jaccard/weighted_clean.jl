using DrWatson
@quickactivate

# load data and necessary packages
include(srcdir("init_strain.jl"))

"""
    jaccard_distance(d1::Vector, d2::Vector)

Calculate the Jaccard distance between two vectors
as number of points in intersetion divided by number
of points in the union:

`d = (# union - # intersection) / # union`.
"""
function jaccard_distance(d1::Vector, d2::Vector)
    int = length(intersect(d1,d2))
    un = length(union(d1,d2))
    return (un - int)/un
end

using BSON
using LinearAlgebra
using ClusterLosses
using JSON

js1 = """{ "a" : {"b" : ["v", "p"], "c" : 1}}"""
js2 = """{ "a" : {"b" : "vx", "c" : 2}}"""
js3 = """{ "a" : {"b" : "v", "c" : 2}}"""
js4 = """{ "a" : {"b" : "v", "c" : 2}}"""

sch = schema([js1,js2,js3,js4])

j1, j2, j3, j4 = JSON.parse.([js1,js2,js3,js4])
prefixes = [("a", "b"), ("a", "c")]
f1, f2, f3, f4 = map(j -> flatten_json(j), [j1,j2,j3,j4])

function jaccard_distance(j1, j2, prefixes, w)
    # if the jsons are the same, automatically return zero (minimum)
    if j1 == j2
        return 0.0
    end

    int = intersect(j1,j2)
    # if intersection is empty, automatically return one (maximum)
    if isempty(int)
        return 1.0
    end

    # else calculate the weighted distance
    un = union(j1,j2)

    m_un = map(pr -> map(j -> issubset(pr, j), un), prefixes)
    m_int = map(pr -> map(j -> issubset(pr, j), int), prefixes)
    
    M_un = hcat(m_un...) * w
    M_int = hcat(m_int...) * w

    return (sum(M_un) - sum(M_int)) / sum(M_un)
end

function jaccard_binaries(j1, j2, prefixes)
    int = intersect(j1,j2)
    un = union(j1,j2)

    m_un = map(pr -> map(j -> issubset(pr, j), un), prefixes)
    m_int = map(pr -> map(j -> issubset(pr, j), int), prefixes)
    
    M_un = hcat(m_un...)
    M_int = hcat(m_int...)
    return M_un, M_int
end

function g(j1,j2,prefixes,w)
    M_un, M_int = jaccard_binaries(j1,j2, prefixes)
    (sum(M_un * w) - sum(M_int * w)) / sum(M_un * w)
end

labels = [0,0,1,1]
f_vec = [f1,f2,f3,f4]
# d = UpperTriangular(zeros(4,4))
d = zeros(4,4)
for i in 1:4
    for j in 1:4
        # d[i,j] = jaccard_distance(f_vec[i], f_vec[j], prefixes, w)
        d[i, j] = g(f_vec[i], f_vec[j], prefixes, w)
    end
end
# d_full = collect(Symmetric(d))

loss(Triplet(), d_full, labels)
# gradient(x -> loss(Triplet(), d_full, labels), d_full)