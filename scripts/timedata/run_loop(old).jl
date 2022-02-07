function fit_triplet(data::TDataset, vec, splits; max_train_time=60*60*2, lr=0.001, batchsize=64, patience=100, kwargs...)
    # dataset split indexes
    tr_x, val_x, ts_x = splits

    # create train minibatch function
    function minibatch_train()
        x, y, t = minibatch(dataset, tr_x, batchsize)
        return x, y
    end
    batch = minibatch_train()

    # validation minibatch function
    function minibatch_val()
        x, y, t = minibatch(dataset, val_x, 500)
        return x, y
    end
    valbatch = minibatch_val()

    # minibatch iterator
    mb_provider = IterTools.repeatedly(minibatch_train, 10)

    # parameters and optimiser
    activation = eval(Symbol(vec.activation))
    agg = eval(Symbol(vec.agg))
    margin, batchsize = vec.margin, vec.batchsize
    model = triplet_mill_constructor(batch[1], vec.mdim, activation, agg, vec.nlayers; odim = vec.odim, seed = vec.init_seed)
    ps = Flux.params(model)
    opt = ADAM(lr)

    # training parameters
    best_loss = Inf
    best_model = deepcopy(model)
    patience = 100
    _patience = 0
    
	# create loss function
    triplet_loss(x, y) = ClusterLosses.loss(Triplet(margin), SqEuclidean(), model(x), y)

    # train
    start_time = time()
    while time() - start_time < max_train_time
        Flux.train!(ps, mb_provider, opt) do x, y
            triplet_loss(x, y)
        end

        l = mean(triplet_loss(minibatch_val()...) for _ in 1:10)

        if l < best_loss
            @show l
            best_loss = l
            best_model = deepcopy(model)
            _patience = 0
        else
            _patience += 1
        end
        if _patience == patience
            @info "Loss did not improve for $patience number of epochs, stopped training."
            break
        end
        if l < 5e-6
            @info "Training loss descreased enough, stopped training."
            break
        end
    end

    return best_model
end

"""
experimental_loop(data::TDataset, sample_params_f, max_seed::Int)
"""
function experimental_loop(data::TDataset, sample_params_f, max_seed::Int; max_train_time = 60*60*2)

    # sample the random hyperparameters
    parameters = sample_params_f()

    # with these hyperparameters, train and evaluate the model on different train/val/tst splits
    for seed in 1:max_seed
        # define where data is going to be saved
        savepath = datadir("timedata_results/triplet/models/seed=$seed")
        mkpath(savepath)

        # get data
        splits = time_split(dataset; seed=seed)

        @info "Trying to fit Triplet model on time-GVMA with parameters $parameters..."
    
        # fit
        best_model = fit_triplet(data, parameters, splits; max_train_time=max_train_time)

        @info "Model fitted."

        params_dict = Dict(keys(parameters) .=> values(parameters))
        dict = merge(Dict(:model => best_model), params_dict)
        safesave(joinpath(savepath, savename("model", parameters, "bson")), dict)

        @info "Saved results and model for seed no. $seed."
    end
    @info "Experiment finished."
end


"""
    sample_results_data(dataset::TDataset, split_seed=7177, seed=2053)

This function samples subset of dataset based on the time and returns three subdatasets
for results calculations.
"""
function sample_results_data(dataset::TDataset, split_seed=7177, seed=2053)
    tr_x, val_x, ts_x = time_split(dataset; seed = split_seed)
    Xtr, ytr, _ = minibatch(dataset, tr_x, 30000, seed)
    Xv, yv, _ = minibatch(dataset, val_x, 30000, seed)
    Xts, yts, _ = minibatch(dataset, ts_x, 30000, seed)
    return Xtr, ytr, Xv, yv, Xts, yts
end

"""
    calculate_results(Xtr, ytr, Xv, yv, Xts, yts, model_file::String;seed=2053)

Calculates results for a results file with the saved model. This version saves time because creating
a dataset to evaluate results on is costly. Therefore, we can create the dataset first, and loop over
model files and calculate just the encodings and clustering + kNN results.
"""
function calculate_results(Xtr, ytr, Xv, yv, Xts, yts, model_file::String;seed=2053)
    d = BSON.load(model_file)
    parameters = copy(d)
    delete!(parameters, :model)
    best_model = d[:model]

    # create triplet encodings and perform clustering
    enc_train = best_model(Xtr)
    M_train = pairwise(Euclidean(), enc_train)
    train_res = cluster_data1(M_train, ytr, type="train")

    enc_val = best_model(Xv)
    M_val = pairwise(Euclidean(), enc_val)
    val_res = cluster_data1(M_val, yv, type="val")

    enc_test = best_model(Xts)
    M_test = pairwise(Euclidean(), enc_test)
    test_res = cluster_data1(M_test, yts, type="test")

    # kNN clustering
    DMv = pairwise(Euclidean(), enc_val, enc_train)
    knn_val = maximum(k -> dist_knn(k, DMv, ytr, yv)[2], 1:10)

    DMt = pairwise(Euclidean(), enc_test, enc_train)
    knn_test = maximum(k -> dist_knn(k, DMt, ytr, yts)[2], 1:10)

    @info "Results calculated."

    full_results = merge(
        train_res, val_res, test_res,
        Dict(:knn_val => knn_val, :knn_test => knn_test)
    )
    
    results = merge(full_results, parameters)
    name = savename(parameters, "bson")

    savepath = datadir("timedata_results/results")
    safesave(joinpath(savepath, name), results)
end