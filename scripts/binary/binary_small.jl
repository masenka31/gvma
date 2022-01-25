using DrWatson
@quickactivate

# load data and necessary packages
include(srcdir("init_strain.jl"))
include(srcdir("confusion_matrix.jl"))
include(scriptsdir("binary", "evaluate_model.jl"))
using Distances, Clustering
using gvma: encode
using DataFrames, BSON

_names = readdir(datadir("models", "missing"))
filtered_names = filter(name -> occursin("HarHar", name), _names)
filtered_names = filter(name -> occursin("ratio=0.05", name), filtered_names)

dfs = DataFrame[]
for n in filtered_names
    df = evaluate_model(X, y, Xs, ys, n; k = 27)
    push!(dfs, df)
end
res = vcat(dfs...)

g = groupby(res[:, Not(:percentages)], [:ratio, :α])
nm = names(g[1])[1:end-4]
cdf = combine(g, nm .=> mean, renamecols=false)

dfs2 = DataFrame[]
for n in filtered_names
    df = evaluate_model(X, y, n; k = 15)
    push!(dfs2, df)
end
res2 = vcat(dfs2...)

g2 = groupby(res2[:, Not(:percentages)], [:ratio, :α])
nm2 = DataFrames.names(g2[1])[1:end-4]
cdf2 = combine(g2, nm2 .=> mean, renamecols=false)

using PrettyTables
pretty_table(cdf, nosubheader=true)
pretty_table(cdf2, nosubheader=true)

# parallelized calculation of the DataFrames
using Base.Threads: @threads
@show Threads.nthreads();

files = readdir(datadir("models", "missing"))

results = repeat([DataFrame()], 10)
@threads for i in 1:10
    filtered_names = filter(name -> occursin(labelnames[i], name), files)
    dfs = DataFrame[]
    for n in filtered_names
        df = evaluate_model(X, y, Xs, ys, n; k = 27)
        push!(dfs, df)
    end
    res = vcat(dfs...)
    results[i] = res
end

safesave(datadir("results_dataframes", "binary_small.bson"), Dict(:results => results))

function nan_mean(x)
    b = isnan.(x)
    return mean(x[.!b])
end


function comb(results, i)
    df = results[i]
    g = groupby(df[:, Not(:percentages)], [:ratio, :α])
    nm = names(g[1])[1:end-4]
    cdf = combine(g, nm .=> nan_mean, renamecols=false)
    cdf = round.(cdf, digits=3)
    return hcat(cdf, DataFrame(:class => repeat([df[1,:missing_class]], size(cdf,1))))
end

combined_results = map(i -> comb(results, i), 1:10)
comb_mater = vcat(combined_results...)

using PrettyTables
map(i -> pretty_table(comb(results, i), tf = tf_markdown, crop=:none, nosubheader=true), 1:10);

fdf = filter(:ratio => r -> r == 0.2, comb_mater)
pretty_table(fdf[:, Not([:ratio, :k])], tf = tf_markdown, nosubheader=true, hlines=collect(4:3:30), crop=:none)