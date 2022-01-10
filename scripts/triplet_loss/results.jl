using DrWatson
@quickactivate

using DataFrames, Flux, Mill

df = collect_results(datadir("triplet_clustering"))
g = groupby(df, :all_classes)
df_all = g[2] |> DataFrame
df_small = g[1] |> DataFrame

names(df_small)

df1 = df_small[:, Not([:all_classes, :opt, :path, :model])]
df1 = filter(:distance => x -> x != "CosineDist", df1)[:, Not(:distance)]
combine(df1, names(df1) .=> mean)

using Statistics

df0 = df_all[:, Not([:all_classes, :opt, :path, :model])]
df0 = filter(:distance => x -> x != "CosineDist", df0)[:, Not(:distance)]

combine(df0, names(df0) .=> mean)
