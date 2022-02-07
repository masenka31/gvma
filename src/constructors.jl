"""
    classifier_constructor(Xtrain, mdim, activation, aggregation, nlayers; seed = nothing)

Constructs a classifier as a model composed of Mill model and simple Chain of Dense layers.
The output dimension is fixed to be 10, `mdim` is the hidden dimension in both Mill model
the Chain model.
"""
function classifier_constructor(Xtrain, mdim, activation, aggregation, nlayers; odim = 10, seed = nothing)
    
    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    # mill model
    m = reflectinmodel(
        Xtrain[1],
        k -> Dense(k, mdim, activation),
        d -> aggregation(d)
    )

    # create the net after Mill model
    if nlayers == 1
        net = Dense(mdim, odim)
    elseif nlayers == 2
        net = Chain(Dense(mdim, mdim, activation), Dense(mdim, odim))
    else
        net = Chain(Dense(mdim, mdim, activation), Dense(mdim, mdim, activation), Dense(mdim, odim))
    end

    # connect the full model
    full_model = Chain(m, Mill.data, net, softmax)

    # reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    # try that the model works
    try
        full_model(Xtrain[1])
    catch e
        error("Model wrong, error: $e")
    end

    return full_model
end

"""
    triplet_mill_constructor(Xtrain, mdim, activation, aggregation, nlayers; odim = 10, seed = nothing)

Constructs a model to use for Triplet metric learning as a model composed of Mill model and
simple Chain of Dense layers.
"""
function triplet_mill_constructor(Xtrain, mdim, activation, aggregation, nlayers; odim = 10, seed = nothing)
    
    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    # mill model
    m = reflectinmodel(
        Xtrain[1],
        k -> Dense(k, mdim, activation),
        d -> aggregation(d)
    )

    # create the net after Mill model
    if nlayers == 1
        net = Dense(mdim, odim)
    elseif nlayers == 2
        net = Chain(Dense(mdim, mdim, activation), Dense(mdim, odim))
    elseif nlayers == 3
        net = Chain(Dense(mdim, mdim, activation), Dense(mdim, mdim, activation), Dense(mdim, odim))
    end

    # connect the full model
    if nlayers == 0
        full_model = Chain(m, Mill.data)
    else
        full_model = Chain(m, Mill.data, net)
    end

    # reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    # try that the model works
    try
        full_model(Xtrain[1])
    catch e
        error("Model wrong, error.")
    end

    return full_model
end