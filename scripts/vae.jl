"""
	safe_softplus(x::T)

Safe version of softplus.	
"""
safe_softplus(x::T) where T  = softplus(x) + T(0.000001)

"""
	function build_mlp(idim::Int, hdim::Int, odim::Int, nlayers::Int; activation::String="relu", lastlayer::String="")

Creates a chain with `nlayers` layers of `hdim` neurons with transfer function `activation`.
input and output dimension is `idim` / `odim`
If lastlayer is no specified, all layers use the same function.
If lastlayer is "linear", then the last layer is forced to be Dense.
It is also possible to specify dimensions in a vector.

```juliadoctest
julia> build_mlp(4, 11, 1, 3, activation="relu")
Chain(Dense(4, 11, relu), Dense(11, 11, relu), Dense(11, 1, relu))

julia> build_mlp([4, 11, 11, 1], activation="relu")
Chain(Dense(4, 11, relu), Dense(11, 11, relu), Dense(11, 1, relu))

julia> build_mlp(4, 11, 1, 3, activation="relu", lastlayer="tanh")
Chain(Dense(4, 11, relu), Dense(11, 11, relu), Dense(11, 1, tanh))
```
"""
build_mlp(ks::Vector{Int}, fs::Vector) = Flux.Chain(map(i -> Dense(i[2],i[3],i[1]), zip(fs,ks[1:end-1],ks[2:end]))...)

build_mlp(idim::Int, hdim::Int, odim::Int, nlayers::Int; kwargs...) =
	build_mlp(vcat(idim, fill(hdim, nlayers-1)..., odim); kwargs...)

function build_mlp(ks::Vector{Int}; activation::String = "relu", lastlayer::String = "")
	activation = (activation == "linear") ? "identity" : activation
	fs = Array{Any}(fill(eval(:($(Symbol(activation)))), length(ks) - 1))
	if !isempty(lastlayer)
		fs[end] = (lastlayer == "linear") ? identity : eval(:($(Symbol(lastlayer))))
	end
	build_mlp(ks, fs)
end

"""
	vae_constructor(;idim::Int=1, zdim::Int=1, activation = "relu", hdim=128, nlayers::Int=3, 
		init_seed=nothing, prior="normal", pseudoinput_mean=nothing, k=1, kwargs...)

Constructs a classical variational autoencoder.

# Arguments
	- `idim::Int`: input dimension.
	- `zdim::Int`: latent space dimension.
	- `activation::String="relu"`: activation function.
	- `hdim::Int=128`: size of hidden dimension.
	- `nlayers::Int=3`: number of decoder/encoder layers, must be >= 3. 
	- `init_seed=nothing`: seed to initialize weights.
	- `prior="normal"`: one of ["normal", "vamp"].
	- `pseudoinput_mean=nothing`: mean of data used to initialize the VAMP prior.
	- `k::Int=1`: number of VAMP components. 
	- `var="scalar"`: decoder covariance computation, one of ["scalar", "diag"].
"""
function vae_constructor(;idim::Int=1, zdim::Int=1, activation="relu", hdim=128, nlayers::Int=3, 
	init_seed=nothing, prior="normal", pseudoinput_mean=nothing, k=1, var="scalar", kwargs...)
	(nlayers < 3) ? error("Less than 3 layers are not supported") : nothing
	
	# if seed is given, set it
	(init_seed != nothing) ? Random.seed!(init_seed) : nothing
	
	# construct the model
	# encoder - diagonal covariance
	encoder_map = Chain(
		build_mlp(idim, hdim, hdim, nlayers-1, activation=activation)...,
		ConditionalDists.SplitLayer(hdim, [zdim, zdim], [identity, safe_softplus])
		)
	encoder = ConditionalMvNormal(encoder_map)
	
	# decoder - we will optimize only a shared scalar variance for all dimensions
	if var=="scalar"
		decoder_map = Chain(
			build_mlp(zdim, hdim, hdim, nlayers-1, activation=activation)...,
			ConditionalDists.SplitLayer(hdim, [idim, 1], [identity, safe_softplus])
			)
	else
		decoder_map = Chain(
				build_mlp(zdim, hdim, hdim, nlayers-1, activation=activation)...,
				ConditionalDists.SplitLayer(hdim, [idim, idim], [identity, safe_softplus])
				)
	end
	decoder = ConditionalMvNormal(decoder_map)

	# prior
	if prior == "normal"
		prior_arg = zdim
	elseif prior == "vamp"
		(pseudoinput_mean === nothing) ? error("if `prior=vamp`, supply pseudoinput array") : nothing
		prior_arg = init_vamp(pseudoinput_mean, k)
	end

	# reset seed
	(init_seed !== nothing) ? Random.seed!() : nothing

	# get the vanilla VAE
	model = VAE(prior_arg, encoder, decoder)
end

function latent_space(m::VAE, data)
    rand(m.encoder, data)
end