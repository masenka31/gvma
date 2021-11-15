using DrWatson
@quickactivate

using gvma
using Flux, DataFrames, BSON
using PrettyTables
using Statistics

# load data
df = collect_results(datadir("regularization"))
# when loaded from regularization_first, add seed
# seeds = parse.(Int64, map(i -> match(r".*seed=([0-9]{1,2}).*", df[i,:path]).captures[1], 1:size(df, 1)))
# df[!,:seed] = seeds     # add seed

# group and combine
g = groupby(df, [:α, :ratio])
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
pretty_table(cdf_rounded[:, [2,1,3,4,5,6,7,8,9,10,11]],
                tf = tf_markdown, nosubheader=true, body_hlines=collect(7:7:60))