using Flux
using Flux: throttle, @epochs
using StatsBase
using Base.Iterators: repeated

ix = sample(1:8000, 8000, replace=false)
tix = ix[1:6000]
vix = ix[6001:end]

X_train, y_train = X[tix], y[tix]
X_val, y_val = X[vix], y[vix]
y_oh = Flux.onehotbatch(y_train, labelnames)
y_oh_val = Flux.onehotbatch(y_val, labelnames)

minibatchsize = 6000
function minibatch()
	idx = sample(tix, minibatchsize, replace = false)
	reduce(catobs, data[idx]), Flux.onehotbatch(y[idx], labelnames)
end
batch = minibatch()

# model
m = reflectinmodel(
    X,
    k -> Dense(k, 10, relu),
    d -> meanmax_aggregation(d)
)
m2 = reflectinmodel(
    X,
    k -> Dense(k, 32, relu),
    d -> meanmax_aggregation(d)
)

opt = ADAM()
net = Dense(32,10)
full_model = Chain(m2, Mill.data, Dense(32,10),softmax)
ps = Flux.params(full_model);

#loss(a, b) = Flux.logitcrossentropy(m(a).data, b)
loss(X, y_oh) = Flux.logitcrossentropy(full_model(X), y_oh)
loss(batch...)

accuracy(x,y) = mean(labelnames[Flux.onecold(m(x).data)] .== y)
accuracy(x,y) = mean(labelnames[Flux.onecold(full_model(x))] .== y)
accuracy(X_val, y_val)

using EvalMetrics

@epochs 30 begin
    #ps_old = deepcopy(ps)
    #Flux.train!(loss, ps, repeated(minibatch(), 1), opt)
    #Flux.train!(loss, Flux.params(m), repeated((X_train, y_oh), 10), opt)
    Flux.train!(loss, Flux.params(full_model), repeated((X_train, y_oh), 10), opt)
    #println(ps_old .== ps)
    println("train: ", loss(X_train, y_oh))
    println("val: ", loss(X_val, y_oh_val))
    println("accuracy train: ", accuracy(X_train, y_train))
    println("accuracy validation: ", accuracy(X_val, y_val))
end

val_labels = zeros(Int, 2000)
for (k,i) in enumerate(vix)
    idx = findall(x -> x == 1, y[i] .== labelnames)
    val_labels[k] = idx[1]
end

enc = m(X_val).data
p = scatter2(enc, 5, 3; color=val_labels, label="",markerstrokewidth=0,markersize=2);
display("image/png", p)

emb = umap(enc, 2)
p = scatter2(emb; color=val_labels,markerstrokewidth=0,markersize=4,opacity=0.5);
display("image/png", p)

enc = net(m(X).data)
emb = umap(enc, 2)
p = scatter2(emb; color=val_labels,markerstrokewidth=0,markersize=4,opacity=0.5,xlims=(-25,25));
display("image/png", p)

emb = umap(enc, 3)
p = scatter(emb[1,:], emb[2,:], emb[3,:]; color=val_labels,markerstrokewidth=0,markersize=4,opacity=0.5);
display("image/png", p)



# train data
enc = m2(X_train).data
p = scatter2(enc, 1, 3; color=labels, label="")
emb = umap(enc, 2)
p = scatter2(emb; zcolor=encode(y_train, labelnames), color=:tab10, markerstrokewidth=0);
display("image/png", p)

function encode(labels, labelnames)
    num_labels = ones(Int, length(labels))
    for i in 1:length(labels)
        v = findall(x -> x == labels[i], labelnames)
        num_labels[i] = v[1]
    end
    return num_labels
end

DataFrame(hcat(labelnames,map(x -> count(==(x), y), labelnames)), [:labelnames, :count])