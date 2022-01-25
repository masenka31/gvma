using DrWatson
@quickactivate

# load data and necessary packages
include(srcdir("init_strain.jl"))

"""
    jaccard_distance(d1::Vector, d2::Vector)

Calculate the Jaccard distance between two vectors
as number of points in intersetion divided by number
of points in the union: d = # intersection / # union.
"""
function jaccard_distance(d1::Vector, d2::Vector)
    #dif = length(setdiff(d1,d2))
    int = length(intersect(d1,d2))
    un = length(union(d1,d2))
    return (un - int)/un
end

# load the Jaccard matrix
using BSON
using LinearAlgebra
L = BSON.load(datadir("jaccard_matrix_strain.bson"))[:L]
# create full matrix
L_full = Symmetric(L)

using ClusterLosses

js1 = """{ "a" : {"b" : ["v", "p"], "c" : 1}}"""
js2 = """{ "a" : {"b" : "vx", "c" : 2}}"""
js3 = """{ "a" : {"b" : "v", "c" : 2}}"""
js4 = """{ "a" : {"b" : "v", "c" : 2}}"""

sch = schema([js1,js2,js3,js4])

using JSON
j1, j2, j3, j4 = JSON.parse.([js1,js2,js3,js4])


# can be converted to a dictionary of DictEntry with 
# sch.childs
# then we can go further down with keys
# how to do this effectively?
# use some kind of unflatten jsons system?
# when you flatten json, you get all of it

w = rand(2)

# we should create a structure that would be a dictionary of prefixes
# how to get the prefixes?
# and the structure would have methods to access the weights in the dictionary

prefixes = [("a", "b"), ("a", "c")]

# where the prefix matches the flatten json
map(j -> issubset(prefixes[1], j), flatten_json(j1))
# where each of the prefixes matches the flatten json
map(pr -> map(j -> issubset(pr, j), flatten_json(j1)), prefixes)

# this defines a matrix which multiplied with weights gives the weight vector
m = map(pr -> map(j -> issubset(pr, j), flatten_json(j1)), prefixes)
M = hcat(m...)

# weights
w = rand(2)
# weight vector for all parts of the flatten json
w_vec = M * w

# flatten jsons
f1, f2, f3, f4 = map(j -> flatten_json(j), [j1,j2,j3,j4])
# how about the jaccard?
jaccard_distance(f1,f2)
jaccard_distance(f2,f3)

# weighted jaccard?
# for each point we need the weight -> which comes from the prefix
# then we can replace the union/intersect with the weights of the union/intersect summed up

int = intersect(f2,f3)
un = union(f2,f3)

# weight matrix for intersection
m_int = hcat(map(pr -> map(j -> issubset(pr, j), int), prefixes)...)
M_int = m_int * w

# weight matrix for union
m_un = hcat(map(pr -> map(j -> issubset(pr, j), un), prefixes)...)
M_un = m_un * w

# and now do the sum
jc_dist = (sum(M_un) - sum(M_int)) / sum(M_int)

# not sure if I can use this if I want to differentiate with respect to the weights?
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

function jaccard_distance2(j1, j2, prefixes, w)
    int = intersect(j1,j2)
    un = union(j1,j2)

    m_un = map(pr -> map(j -> issubset(pr, j), un), prefixes)
    M_un = hcat(m_un...) * w

    m_int = map(pr -> map(j -> issubset(pr, j), int), prefixes)
    M_int = hcat(m_int...) * w

    return (sum(M_un) - sum(M_int)) / sum(M_un)
end

"""
# benchmark for the first jaccard
julia> @btime jaccard_distance(f4,f3,prefixes, w)
  139.373 ns (0 allocations: 0 bytes)
0.0

julia> @btime jaccard_distance(f1,f3,prefixes, w)
  2.511 μs (31 allocations: 1.50 KiB)
1.0

julia> @btime jaccard_distance(f2,f3,prefixes, w)
  5.567 μs (71 allocations: 3.69 KiB)
0.7733125884630444

# benchmark for the second jaccard
julia> @btime jaccard_distance(f4,f3,prefixes, w)
  5.257 μs (69 allocations: 3.62 KiB)
0.0

julia> @btime jaccard_distance(f1,f3,prefixes, w)
  7.300 μs (93 allocations: 4.52 KiB)
1.0

julia> @btime jaccard_distance(f2,f3,prefixes, w)
  5.400 μs (71 allocations: 3.69 KiB)
0.7733125884630444
"""

fjs = [f1,f2,f3,f4]
dm = UpperTriangular(zeros(4,4))
for i in 1:4
    for j in i+1:4
        dm[i,j] = jaccard_distance(fjs[i], fjs[j])
    end
end

loss(Triplet(), dm, rand(0:1,4))

# triplet loss - how it works
d = Symmetric(rand(10,10))
for i in 1:10
    d[i,i] = 0
end
loss(Triplet(), d, rand(0:1,10))

# differentiation?
using Flux, ClusterLosses, Distances
y = [1,1,2,2];
x = rand(2,4)
map(l -> gradient(x -> loss(l, CosineDist(), x, y), x)[1], [Triplet(), NCA(), NCM()])
map(l -> gradient(x -> loss(l, SqEuclidean(), x, y), x)[1], [Triplet(), NCA(), NCM()])

# how do you differentiate with respect to the weights?
# you need to recalculate the jaccard distance all the time
# implement somethig as
# loss(Triplet, Jaccard(), jsons, y) ?


dm_full = Symmetric(dm)
gradient(x -> loss(Triplet(), dm_full, [1,1,2,2]), dm_full)


# getting the prefixes?
types = map(x -> typeof.(x), jsons_flatten[1])
map((x, t) -> x[t .== String], jsons_flatten[1])

# why not do it by hand :D
prefixes = [
    ("behavior_summary", "files"),
    ("behavior_summary", "read_files"),
    ("behavior_summary", "resolved_apis"),
    ("info", "source"),
    ("network_http"),
    ("signatures", "description"),
    ("signatures", "severity"),
    ("signatures", "name")
]

jsons_flatten
# initialize the weights
w = softmax(rand(length(prefixes)))

# randomly sample and calculate weighted jaccard distance
k, l = rand(1:8000, 2)
jaccard_distance(jsons_flatten[k], jsons_flatten[l], prefixes, w)

# but what about the weights?
# how do you make it such that the weights are optimized?
# create a neural network for them? what would be the input?
# the input could be the number of values with the prefix in the union/intersection?

function jaccard_binaries(j1, j2, prefixes)
    int = intersect(j1,j2)
    un = union(j1,j2)

    m_un = map(pr -> map(j -> issubset(pr, j), un), prefixes)
    m_int = map(pr -> map(j -> issubset(pr, j), int), prefixes)
    
    M_un = hcat(m_un...)
    M_int = hcat(m_int...)
    return M_un, M_int
end

M_un, M_int = jaccard_binaries(f2,f3, prefixes)
f(M_un, M_int) = (sum(M_un * w) - sum(M_int * w)) / sum(M_un * w)
gradient(f, M_un, M_int)
f(w) = (sum(M_un * w) - sum(M_int * w)) / sum(M_un * w)
gradient(f, rand(2))



# triplet loss - how it works
f(w) = (sum(M_un * w) - sum(M_int * w)) / sum(M_un * w)
function g(j1,j2,prefixes,w)
    M_un, M_int = jaccard_binaries(j1,j2, prefixes)
    (sum(M_un * w) - sum(M_int * w)) / sum(M_un * w)
end

MUN, MINT = 

f_vec = [f1,f2,f3,f4]
d = UpperTriangular(zeros(4,4))
for i in 1:4
    for j in i+1:4
        d[i,j] = jaccard_d
end
gradient(x -> loss(Triplet(), dm_full, [1,1,2,2]), dm_full)