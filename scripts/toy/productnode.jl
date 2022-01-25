X = ProductNode((x = ArrayNode([1f0]), y = ArrayNode([12f0])))

x, y = sample(1:100, 2)
xoh = Flux.onehot(x, 1:100)
yoh = Flux.onehot(y, 1:100)

X = ProductNode((x = ArrayNode(xoh), y = ArrayNode(yoh)))

function Dict(X::ProductNode)