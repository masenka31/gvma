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
                tf = tf_markdown, nosubheader=true, body_hlines=collect(7:7:60)
)

cdf_lean = filter(:α => α -> α == 0.1f0, cdf)[:, [:ratio, :train_acc, :test_acc, :randindex, :ajd_randindex]]

#########################
### for missing class ###
#########################

using DrWatson
@quickactivate
using DataFrames
using Statistics

df2 = collect_results(datadir("regularization_missing"), subfolders=true)
df_filt = filter(:α => α -> α == 0.1f0, df2)
g2 = groupby(df_filt, [:class, :ratio])
cdf2 = combine(g2,
    [:train_acc, :test_acc, :randindex, :ajd_randindex] .=> mean,
    renamecols=false
)

cdf2_lean = filter(:ratio => x -> x == 0.2, cdf2)
cdf_compare = vcat(
    hcat(DataFrame(:class => "all"), DataFrame(cdf_lean[3,:])),
    cdf2_lean
)

pretty_table(sort(cdf_compare, :ajd_randindex, rev=true); nosubheader=true, crop=:none)

df3 = collect_results(datadir("regularization_cosine"), subfolders=true)
df_filt3 = filter(:α => α -> α == 0.1f0, df3)
g3 = groupby(df_filt3, [:ratio])
cdf3 = combine(g3,
    [:train_acc, :test_acc, :randindex, :ajd_randindex] .=> mean,
    renamecols=false
)

cdf3_lean = filter(:ratio => x -> x == 0.2, cdf3)
cdf_compare = vcat(
    hcat(DataFrame(:class => "all"), DataFrame(cdf_lean[3,:])),
    cdf2_lean
)