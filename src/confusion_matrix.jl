
"""
Confusion matrix in the form of:

         true:    0   1

    predicted: 0  TP  FP

    predicted: 1  FN  TN
"""
struct ConfusionMatrix{T}
    TP::T
    FP::T
    FN::T
    TN::T
end

function Base.show(io::IO, m::ConfusionMatrix)
    str = """
    ConfusionMatrix:
                true:   |   0   1
        -------------------------------
        predicted:  0   |   $(m.TN) $(m.FN)
        predicted:  1   |   $(m.FP) $(m.TP)
    """
    print(io, str)
end


import Base: sum
sum(CM::ConfusionMatrix) = sum([CM.TP, CM.TN, CM.FP, CM.FN])
accuracy(CM::ConfusionMatrix) = (CM.TP + CM.TN) / sum(CM)
recall(CM::ConfusionMatrix) = CM.TP / (CM.TP + CM.FN)
precision(CM::ConfusionMatrix) = CM.TP / (CM.TP + CM.FP)

function report(m::ConfusionMatrix)
    a = accuracy(m)
    r = recall(m)
    p = precision(m)

    DataFrame(:accuracy => a, :recall => r, :precision => p)
end