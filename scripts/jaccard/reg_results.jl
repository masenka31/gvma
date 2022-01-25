using DrWatson
@quickactivate

using gvma
using Flux, DataFrames, BSON
using PrettyTables
using Statistics
using Mill

# load data
_df = collect_results(datadir("regularization"))
# when loaded from regularization_first, add seed
# seeds = parse.(Int64, map(i -> match(r".*seed=([0-9]{1,2}).*", df[i,:path]).captures[1], 1:size(df, 1)))
# df[!,:seed] = seeds     # add seed

df = filter(:α => x -> x in [0,0.1f0,1], _df)

# group and combine
g = groupby(df, [:α, :ratio])
cdf = combine(
    g,
    [:train_acc, :test_acc, :randindex, :ajd_randindex] .=> mean,
    renamecols=false
)
cdf_rounded = round.(cdf, digits=3)

# print markdown table
pretty_table(cdf_rounded, tf = tf_markdown, nosubheader=true, body_hlines=collect(4:4:60), crop=:none)

cdf_lean = filter(:α => α -> α == 0.1f0, cdf)
pretty_table(cdf_lean, tf = tf_markdown, nosubheader=true)

#########################
### for missing class ###
#########################

using DrWatson
@quickactivate
using DataFrames
using Statistics

df2 = collect_results(datadir("regularization_missing"), subfolders=true)
# df_filt = filter(:α => α -> α == 0.1f0, df2)
g2 = groupby(df2, [:class, :α, :ratio])
cdf2 = combine(g2,
    [:train_acc, :test_acc, :randindex, :ajd_randindex] .=> mean
    #renamecols=false
)

cdf2_lean = filter(:α => y -> y == 0.1f0, filter(:ratio => x -> x == 0.2, cdf2))
cdf_compare = vcat(
    hcat(DataFrame(:class => "all"), DataFrame(cdf_lean[3,:])),
    cdf2_lean
)

pretty_table(sort(cdf_compare, :ajd_randindex_mean, rev=true); tf = tf_markdown, nosubheader=true, crop=:none)

df3 = collect_results(datadir("regularization_cosine"), subfolders=true)
df_filt3 = filter(:α => α -> α != 0.2f0, df3)
g3 = groupby(df_filt3, [:α, :ratio])
cdf3 = combine(g3,
    [:train_acc, :test_acc, :randindex, :ajd_randindex] .=> mean,
    renamecols=false
)

cdf3_rounded = round.(cdf3, digits=3)
pretty_table(cdf3_rounded, tf = tf_markdown, nosubheader=true, body_hlines=collect(4:4:12), crop=:none)

cdf3_lean = filter(:α => α -> α == 0.1f0, filter(:ratio => x -> x == 0.2, cdf3))
cdf_compare = vcat(
    hcat(DataFrame(:class => "all"), DataFrame(cdf_lean[3,:])),
    cdf2_lean
)

### Comparison of regularization : no, Euclidean , Cosine
e_df = filter(:α => x -> x == 0.1f0 || x == 0.0f0, filter(:ratio => r -> r == 0.2, cdf))
c_df = filter(:α => x -> x == 0.1f0, cdf3_lean)

final_df = hcat(DataFrame(:distance => ["Euclidean", "Euclidean", "Cosine"]), vcat(e_df, c_df))
pretty_table(final_df, tf = tf_markdown, nosubheader=true, crop=:none)

####################################################################################
# new regularization with new Jaccard matrix
# load data
df = collect_results(datadir("regularization"))

# group and combine
g = groupby(df, [:α, :ratio])
cdf = combine(
    g,
    [:train_acc, :test_acc, :ri, :adj_ri, :silhouettes] .=> mean,
    renamecols=false
)
cdf_rounded = round.(cdf, digits=3)

# print markdown table
pretty_table(cdf_rounded, tf = tf_markdown, nosubheader=true, body_hlines=collect(3:3:9), crop=:none)

cdf_lean = filter(:α => α -> α == 0.1f0, cdf)
pretty_table(cdf_lean, tf = tf_markdown, nosubheader=true)