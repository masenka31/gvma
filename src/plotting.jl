function scatter2(X, x=1, y=2; kwargs...)
    if size(X,1) > size(X,2)
        X = X'
    end
    scatter(X[x,:],X[y,:]; kwargs...)
end
function scatter2!(X, x=1, y=2; kwargs...)
    if size(X,1) > size(X,2)
        X = X'
    end
    scatter!(X[x,:],X[y,:]; kwargs...)
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