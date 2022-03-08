import numpy as n

def debugInitializeWeights(fan_out,fan_in):
    w = n.zeros((fan_out, 1 + fan_in))
    w = n.reshape(n.sin(n.array(range(w.size))), n.shape(w)) / 10.0
    return w