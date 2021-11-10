# loads necesarry packages, functions
# creates data
include("init.jl")
# other packages for VAE usage
using GenerativeModels, DistributionsAD, ConditionalDists, IPMeasures
include("vae.jl")

# Mill model
mdim, activation, aggregation = 32, tanh, max_aggregation
m = reflectinmodel(
    X,
    k -> Dense(k, mdim, activation),
    d -> aggregation(d)
)

### load existing model
df = collect_results(datadir("classifier"))
sort!(df, :train_accuracy, rev=true)
best_model = df[1,:model]
m = best_model[1]

tix, vix = df[1,:indices]
X_train, y_train = X[tix], y[tix]
X_val, y_val = X[vix], y[vix]

# VAE model
vae = vae_constructor(;idim=mdim, zdim = 2, activation="tanh", hdim = 32, nlayers=3)
loss(x) = -elbo(vae, m(x).data)
opt = ADAM()
loss(x)
best_mse = Inf
vae_best = deepcopy(vae)

@epochs 100 begin
    Flux.train!(loss, Flux.params(vae), repeated(X, 1), opt)
    x_new = mean(vae.decoder, mean(vae.encoder, m(X_train).data))
    err = Flux.mse(m(X_train).data, x_new)
    println("MSE: $err")
    if err < best_mse
        vae_best = deepcopy(vae)
        best_mse = err
    end
end

rep = latent_space(vae_best, m(X).data)
p = scatter2(rep; zcolor=encode(y, labelnames), color=:tab10, markerstrokewidth=0);
wsave("vae_enc_new200epochs.png", p)

emb = m(X).data
p = scatter2(emb; zcolor=encode(y, labelnames), color=:tab10, markerstrokewidth=0);
wsave("m_enc.png", p)