using DrWatson
@quickactivate

include(scriptsdir("toy", "src.jl"))
include(scriptsdir("toy", "cluster_fun.jl"))
include(scriptsdir("toy", "jaccard_metric.jl"))

using DataFrames
using DataFrames: groupby
using PrettyTables

using Plots, StatsPlots
ENV["GKSwstype"] = "100"


"""
    function generate_results(type::String="weighted", folder::String="toy_max=1000"; pretty = true)

Generates a results table for the toy problem.
"""
function generate_results(type::String="weighted", folder::String="toy_max=1000"; pretty = true)
    if type=="weighted"
        df_weighted = collect_results(datadir("toy_max=1000", "weighted"))
        df = df_weighted[:, Not([:opt, :model, :code, :path])]
        gdf = groupby(df, [:λ, :n_classes, :unq])
    elseif type=="triplet"
        _df = collect_results(datadir(folder, "triplet"))
        df = _df[:, Not([:opt, :model, :code, :path])]
        gdf = groupby(df, [:λ, :n_classes, :unq, :activation])
    elseif type=="jaccard"
        _df = collect_results(datadir(folder, "jaccard"))
        df = _df[:, Not([:code, :path])]
        gdf = groupby(df, [:λ, :n_classes, :unq])
    else
        println("This type is not supported.")
        return
    end

    metrics = ["ri_hclust_test_dm", "ri_hclust_test_emb", "ri_medoids_test_dm", "ri_medoids_test_emb",
               "ri_hclust_train_dm", "ri_hclust_train_emb", "ri_medoids_train_dm", "ri_medoids_train_emb"]
               
    if type=="triplet"
        new_names = ["λ", "n_classes", "unq", "activation",
        "hclust_test_dm", "hclust_test_emb", "medoids_test_dm", "medoids_test_emb",
        "hclust_train_dm", "hclust_train_emb", "medoids_train_dm", "medoids_train_emb"]
    elseif type=="weighted" || type=="jaccard"
        new_names = ["λ", "n_classes", "unq",
        "hclust_test_dm", "hclust_test_emb", "medoids_test_dm", "medoids_test_emb",
        "hclust_train_dm", "hclust_train_emb", "medoids_train_dm", "medoids_train_emb"]
    end
    
    cdf = combine(gdf, metrics .=> mean, renamecols=false)
    rename!(cdf, new_names)

    if pretty
        return pretty_table(cdf, nosubheader=true, crop=:none, body_hlines = collect(3:3:1000))
    else
        return cdf
    end
end

# results for triplet
table_triplet = generate_results("triplet", "toy_max=10000", pretty=false)
df_relu = filter(:activation => x -> x == "relu", table_triplet)
df_swish = filter(:activation => x -> x == "swish", table_triplet)
pretty_table(
    df_relu, nosubheader=true, crop=:none,# tf=tf_markdown,
    body_hlines = collect(3:3:1000),
    formatters = ft_printf("%5.3f", 5:12)
)

pretty_table(
    df_swish, nosubheader=true, crop=:none,# tf=tf_markdown,
    body_hlines = collect(3:3:1000),
    formatters = ft_printf("%5.3f", 5:12)
)

# results for weighted Jaccard model

table_weighted = generate_results("weighted", "toy_max=10000", pretty=false)
sort!(table_weighted, [:n_classes, :unq, :λ])
pretty_table(
    table_weighted, nosubheader=true, crop=:none,
    body_hlines = collect(3:3:1000),
    formatters = ft_printf("%5.3f", 4:11)
)

# results for simple Jaccard
table_jaccard = generate_results("jaccard", "toy_max=1000", pretty=false)
pretty_table(
    table_jaccard, nosubheader=true, crop=:none,
    body_hlines = collect(3:3:1000),
    formatters = ft_printf("%5.3f", 4:11)
)

###################################################
############## Old results gathering ##############
###################################################

# df_triplet = collect_results(datadir("toy", "triplet"))
df_triplet = collect_results(datadir("toy_max=1000", "triplet"))
# println(names(df_triplet))
df = df_triplet[:, Not([:opt, :model, :code, :path])]

# println(names(df))
gdf = groupby(df, [:λ, :n_classes, :unq, :activation])
# g = gdf[1]
# println(names(g))

metrics0 = ["ri_hclust_test_dm", "slt_medoids_test_dm", "ri_medoids_train_emb", "slt_medoids_test_emb",
           "ri_hclust_test_emb", "slt_medoids_train_emb", "slt_hclust_train_dm",
           "slt_hclust_train_emb", "slt_medoids_train_dm", "ri_medoids_test_emb",
           "slt_hclust_test_emb", "ri_medoids_test_dm", "slt_hclust_test_dm",
           "ri_hclust_train_dm", "ri_hclust_train_emb", "ri_medoids_train_dm"
]

metrics = ["ri_hclust_test_dm", "ri_medoids_train_emb", "ri_hclust_test_emb",
           "ri_medoids_test_emb", "ri_medoids_test_dm", "ri_hclust_train_dm",
           "ri_hclust_train_emb", "ri_medoids_train_dm"
]


sort!(metrics)
cdf = combine(gdf, metrics .=> mean, renamecols=false)
# cdf = combine(gdf, metrics .=> mean)

gdf_un = groupby(cdf, :unq)
cdf_un = combine(gdf_un, metrics .=> mean, renamecols=false)
# cdf_un = combine(gdf_un, metrics .=> mean)

using PrettyTables
pretty_table(cdf, nosubheader=true, crop=:none, body_hlines = collect(3:3:1000))
pretty_table(cdf_un, nosubheader=true, crop=:none)

cols = ["ri_hclust_test_dm", "ri_hclust_test_emb",
    "ri_medoids_test_dm", "ri_medoids_test_emb"
]


# filter true
filt_df = filter(:unq => x -> x == true, df)

@df pivot(filt_df, cols) groupedboxplot(:λ, :value, group=:name, legend=:bottomleft, ylims=(0,1.05),
                    xlabel="λ", ylabel="RandIndex", title="Test data results", titlefontsize=10,
                    label=["HClust (DM)" "HClust (UMAP)" "k-medoids (DM)" "k-medoids (UMAP)"])
savefig("triplet_box_test_lambda_unq=true.png")

@df pivot(filt_df, cols) groupedboxplot(:n_classes, :value, group=:name, legend=:bottomleft, ylims=(0,1.05),
                    xlabel="n classes", ylabel="RandIndex", title="Test data results", titlefontsize=10,
                    label=["HClust (DM)" "HClust (UMAP)" "k-medoids (DM)" "k-medoids (UMAP)"])
savefig("triplet_box_test_classes_unq=true.png")

# filter false
filt_df = filter(:unq => x -> x == false, df)

@df pivot(filt_df, cols) groupedboxplot(:λ, :value, group=:name, legend=:bottomleft, ylims=(0,1.05),
                    xlabel="λ", ylabel="RandIndex", title="Test data results", titlefontsize=10,
                    label=["HClust (DM)" "HClust (UMAP)" "k-medoids (DM)" "k-medoids (UMAP)"])
savefig("triplet_box_test_lambda_unq=false.png")

@df pivot(filt_df, cols) groupedboxplot(:n_classes, :value, group=:name, legend=:bottomleft, ylims=(0,1.05),
                    xlabel="n classes", ylabel="RandIndex", title="Test data results", titlefontsize=10,
                    label=["HClust (DM)" "HClust (UMAP)" "k-medoids (DM)" "k-medoids (UMAP)"])
savefig("triplet_box_test_classes_unq=false.png")

################################
######### Weight Model #########
################################

# df_weighted = collect_results(datadir("toy", "weighted"))
df_weighted = collect_results(datadir("toy_max=1000", "weighted"))
df = df_weighted[:, Not([:opt, :model, :code, :path])]
gdf = groupby(df, [:λ, :n_classes, :unq])

cdf = combine(gdf, metrics .=> mean, renamecols=false)
gdf_un = groupby(cdf, :unq)
cdf_un = combine(gdf_un, metrics .=> mean, renamecols=false)

using PrettyTables
pretty_table(cdf, nosubheader=true, crop=:none, body_hlines = collect(3:3:1000))
pretty_table(cdf_un, nosubheader=true, crop=:none)

cols = ["ri_hclust_test_dm", "ri_hclust_test_emb",
    "ri_medoids_test_dm", "ri_medoids_test_emb"
]

# filter true
filt_df = filter(:unq => x -> x == true, df)

@df pivot(filt_df, cols) groupedboxplot(:λ, :value, group=:name, legend=:bottomleft, ylims=(0,1.05),
                    xlabel="λ", ylabel="RandIndex", title="Test data results", titlefontsize=10,
                    label=["HClust (DM)" "HClust (UMAP)" "k-medoids (DM)" "k-medoids (UMAP)"])
savefig("box_test_lambda_unq=true.png")

@df pivot(filt_df, cols) groupedboxplot(:n_classes, :value, group=:name, legend=:bottomleft, ylims=(0,1.05),
                    xlabel="n classes", ylabel="RandIndex", title="Test data results", titlefontsize=10,
                    label=["HClust (DM)" "HClust (UMAP)" "k-medoids (DM)" "k-medoids (UMAP)"])
savefig("box_test_classes_unq=true.png")

# filter false
filt_df = filter(:unq => x -> x == false, df)

@df pivot(filt_df, cols) groupedboxplot(:λ, :value, group=:name, legend=:bottomleft, ylims=(0,1.05),
                    xlabel="λ", ylabel="RandIndex", title="Test data results", titlefontsize=10,
                    label=["HClust (DM)" "HClust (UMAP)" "k-medoids (DM)" "k-medoids (UMAP)"])
savefig("box_test_lambda_unq=false.png")

@df pivot(filt_df, cols) groupedboxplot(:n_classes, :value, group=:name, legend=:bottomleft, ylims=(0,1.05),
                    xlabel="n classes", ylabel="RandIndex", title="Test data results", titlefontsize=10,
                    label=["HClust (DM)" "HClust (UMAP)" "k-medoids (DM)" "k-medoids (UMAP)"])
savefig("box_test_classes_unq=false.png")



"""
    pivot(df, cols::Vector{T}; whichcols = nothing, newname = :name, newvalue = :value) where T <: Union{Symbol, String}

Takes `cols` and reshapes the dataframe such that cols are combined to two columns:
- name: the former column name
- value: the former column value

Everything else is copied from the original dataframe.
"""
function pivot(df, cols::Vector{T}; whichcols = nothing, newname = :name, newvalue = :value) where T <: Union{Symbol, String}
    n = length(cols)
    subdf = [df[:, cols[i]] for i in 1:n]
    sz = length(subdf[1])

    new_col = DataFrame(newname => ["str" for _ in 1:n*sz], newvalue => zeros(eltype(subdf[1]), n*sz))
    for i in 1:n
        for j in 1:sz
            k = j + (i-1)*sz
            new_col[k, :name] = String(cols[i])
            new_col[k, :value] = subdf[i][j]
        end
    end

    isnothing(whichcols) ? whichcols = Not(cols) : nothing

    new_df = vcat([df for _ in 1:n]...)
    hcat(new_df, new_col)
end



pretty_table(cdf_triplet, nosubheader=true, crop=:none, body_hlines = collect(3:3:1000))
pretty_table(cdf_weight, nosubheader=true, crop=:none, body_hlines = collect(3:3:1000))