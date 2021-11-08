########################
### Weighted Jaccard ###
########################

altup = vcat(jsons_flatten...)
un = unique(altup)
c = countmap(altup)

h = histogram(collect(values(c)), ylims=(0,80))
wsave(plotsdir("stidf", "count_hist.png"), h)


a = [1,2,3]
b = [2,3,4]
c = [2,4,7]
d = [2,8,9]

function weighted_jaccard(a, b, c)
    u = union(a, b)
    w_u = (8000 .- [c[ui] for ui in u]) ./ 8000
    int = intersect(a, b)
    w_i = (8000 .- [c[ii] for ii in int]) ./ 8000

    wu = sum(w_u)
    wint = sum(w_i)

    (wu - wint)/wu
end

wji = map(x -> weighted_jaccard(a, x, c), jsons_flatten)

mean(wji[1:800])
map(i -> mean(wji[i:i+799]), 1:800:8000)

n = 1000
_ix = sample(1:8000, n; replace=false)
ratio = 0.4
n1 = Int(ratio*n)

train_ix, test_ix = _ix[1:n1], _ix[n1+1:end]
Xtrain, ytrain = X[train_ix], y[train_ix]
Xtest, ytest = X[test_ix], y[test_ix]

WJI = []
for a in jsons_flatten[_ix]
    wji = map(x -> weighted_jaccard(a, x, c), jsons_flatten[_ix])
    push!(WJI, wji)
end

Lsmall = hcat(WJI...)

distance_matrix = Lsmall[n1+1:end, 1:n1]
dist_knn(1, distance_matrix, ytrain, ytest)

n = 8000
WJI = zeros(n,n) |> UpperTriangular
for i in 1:n
    for j in i+1:n
        wji = weighted_jaccard(jsons_flatten[i], jsons_flatten[j], c)
        WJI[i, j] = wji
    end
end

wsave(datadir("jaccard_weighted.bson"), Dict(:Lw => WJI))
WJI = BSON.load(datadir("jaccard_weighted.bson"))[:Lw]

Lfull = Symmetric(WJI)

plt = plot()
for seed in 1:5
    (Xtrain, ytrain), (Xtest, ytest), distance_matrix = train_test_split(X, y, Lfull; ratio=0.01, seed=seed)
    acc = Float16[]
    for kn in 1:20
        pred = dist_knn(kn, distance_matrix, ytrain, ytest);
        push!(acc, pred[2])
    end
    plt = plot!(1:20, acc, marker=:circle, ylims=(0,1), legend=:bottomleft,
    xlabel="k neighbors", ylabel="accuracy", label="seed = $seed")
end
wsave(plotsdir("strain_knn", "k-accuracy_ntr=$(8000*0.01)_weighted.png"), plt)