using DrWatson
@quickactivate

using JSON, HDF5
using ProgressMeter
using gvma
using Dates
using StatsBase

using gvma: TDataset, random_ix


function time_split(d::TDataset, breaktime=DateTime(2020,11,30,23,59,59,999); ratio=0.5, seed=nothing)
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

function minibatch(d::TDataset, ixs, batchsize=64)
    ix = sample(ixs, batchsize)
    x, y, t, _ = d[ix]
end


dataset = TDataset()
tr_x, val_x, ts_x = time_split(dataset; seed=1)
train_minibatch = minibatch(dataset, tr_x)

