import numpy as n
import math
import random as r



#Function to initialize random thetas for symmetry breaking
def initializeRandWeights(L_in,L_out):
    epsilon = (math.sqrt(6))/(math.sqrt(L_in + L_out))
    w = n.random.random(size = (L_out, 1 + L_in)) * 2 * epsilon - epsilon
    return w
