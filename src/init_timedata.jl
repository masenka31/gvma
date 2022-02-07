using DrWatson
@quickactivate

# using JSON, HDF5
# using ProgressMeter
using gvma
using Dates
using StatsBase

using gvma: random_ix, Info, TimeDataset
using Random


function time_split(d::Info, breaktime=DateTime(2020,11,30,23,59,59,999); ratio=0.5, seed=nothing)
    # indexes to work with
    timestamp = d.timestamp
    sorted_ix = sortperm(timestamp)
    IX = 1:length(timestamp)
    
    # find breakpoint index
    break_ix = findfirst(x -> x > breaktime, timestamp[sorted_ix])
    # get train+val and test indexes
    ix1, test_ix = IX[sorted_ix[1:break_ix-1]], IX[sorted_ix[break_ix:end]]

    # get length of train/val indexes
    n = length(ix1)
    n1 = round(Int, ratio*n)
    # randomly sample and split indexes
    _ix = random_ix(n, seed)
    # split indexes to train/val
    train_ix, val_ix = ix1[_ix[1:n1]], ix1[_ix[n1+1:end]]

    # return only the split indexes
    return train_ix, val_ix, test_ix
end

"""
function minibatch(d::TDataset, ixs, batchsize=64)
    ix = sample(ixs, batchsize)
    x, y, t, _ = d[ix]
    return x, y, t
end
function minibatch(d::TDataset, ixs, batchsize, seed)
    # fix and reset seed
    Random.seed!(seed)
    ix = sample(ixs, batchsize)
    Random.seed!()

    x, y, t, _ = d[ix]
    return x, y, t
end
"""

function minibatch(d::TimeDataset, batchsize=64)
    ix = sample(1:nobs(d.data), batchsize)
    x, y, t = d.data[ix], d.strain[ix], d.timestamp[ix]
    return x, y, t
end