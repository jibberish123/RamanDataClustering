import numpy as n
import math


def sigmoid(x):
    answer = (1)/(1 + n.exp(-x))
    return answer

def sigmoidGradient(z):
    answer = n.multiply(sigmoid(z), (1 - sigmoid(z)))
    return answer
    
