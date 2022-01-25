using DrWatson
@quickactivate
using Mill

# data
include(srcdir("init_strain.jl"))
"""
julia>labelnames
10-element Vector{InlineStrings.String7}:
 "clean"
 "Dinwod"
 "HarHar"
 "Upatre"
 "Pakes"
 "Ulise"
 "CTS"
 "Kraton"
 "Waski"
 "Lamer"
"""
println(ARGS)
missing_class = ARGS[1]
ratio = parse(Float64, ARGS[2])
(Xtrain, ytrain), (Xtest, ytest) = train_test_split(X, y, missing_class; ratio=ratio, seed=1)
y_oh = Flux.onehotbatch(ytrain, labelnames)
y_oh_t = Flux.onehotbatch(ytest, labelnames)
unique(ytrain)
unique(ytest)

# Parameters
mdim, activation, aggregation, nlayers = 32, relu, meanmax_aggregation, 2

# model
m = reflectinmodel(
    Xtrain[1],
    k -> Dense(k, mdim, activation),
    d -> aggregation(d)
)

if nlayers == 1
    net = Dense(mdim, 10)
elseif nlayers == 2
    net = Chain(Dense(mdim, mdim, activation), Dense(mdim, 10))
else
    net = Chain(Dense(mdim, mdim, activation), Dense(mdim, mdim, activation), Dense(mdim, 10))
end

full_model = Chain(m, Mill.data, net, softmax)

# try that the model works
try
    full_model(X[1])
catch e
    error("Model wrong, error: $e")
end

opt = ADAM()

# loss and accuracy
loss(X, y_oh) = Flux.logitcrossentropy(full_model(X), y_oh)
accuracy(x, y) = mean(labelnames[Flux.onecold(full_model(x))] .== y)

@info "Model created, starting training..."

# train the model
@epochs 30 begin
    Flux.train!(loss, Flux.params(full_model), repeated((Xtrain, y_oh), 10), opt)
    println("train: ", loss(Xtrain, y_oh))
    println("val: ", loss(Xtest, y_oh_t))
    println("accuracy train: ", accuracy(Xtrain, ytrain))
    println("accuracy validation: ", accuracy(Xtest, ytest))
end

# encode labels to numbers
function encode(labels, labelnames)
    num_labels = ones(Int, length(labels))
    for i in 1:length(labels)
        v = findall(x -> x == labels[i], labelnames)
        num_labels[i] = v[1]
    end
    return num_labels
end

# test data UMAP for different number of neighbors
enc_tst = m(Xtest).data
misix = findall(x -> x == missing_class, labelnames)[1]

for nn in [1,2,3,4,5,10,15,20]
    emb_tst = umap(enc_tst, 2, n_neighbors=nn)
    p = scatter2(emb_tst; zcolor=encode(ytest, labelnames),
                    color=:tab10, markerstrokewidth=0, label="",
                    title="Missing class: $missing_class (index = $misix)\nn_neighbors = $nn");

    name = savename("nn=$nn", Dict(:ratio => ratio), "png")
    safesave(plotsdir("strain_missing", missing_class, name), p)
end
@info "Test UMAP encoding saved."