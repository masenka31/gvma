using DrWatson
@quickactivate

using gvma
using Flux, DataFrames, BSON
using PrettyTables

# load data
df = collect_results(datadir("regularization"))
seeds = parse.(Int64, map(i -> match(r".*seed=([0-9]{1,2}).*", df[i,:path]).captures[1], 1:size(df, 1)))
df[!,:seed] = seeds     # add seed
sort!(df, [:seed, :reg])

# group and combine
g = groupby(df, :reg)
cdf = combine(
    g,
    :train_acc => mean,
    :test_acc => mean,
    :randindex => mean,
    :ajd_randindex => mean,
    :hubert => mean,
    :mirkin => mean,
    :varinfo => mean,
    :mutualinfo => mean,
    :vmeasure => mean,
    renamecols = false
)
cdf_rounded = round.(cdf, digits=4)

# print markdown table
pretty_table(cdf_rounded, tf = tf_markdown, nosubheader=true)