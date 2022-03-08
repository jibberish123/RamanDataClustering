import numpy as n

from debugInitializeWeights import debugInitializeWeights
from costFunction import costFunction
def checkGradients(lmbda):
    
    #Definitions:
    inputLayerSize = 3
    hiddenLayerSize = 5
    numLabels = 3
    m = 5

    #Create debugging random thetas
    Theta1 = debugInitializeWeights(hiddenLayerSize,inputLayerSize)
    Theta2 = debugInitializeWeights(numLabels,hiddenLayerSize)

    #Create debugging random X and y
    X = debugInitializeWeights(m, inputLayerSize - 1)
    y = 1 + n.mod(range(1,m), numLabels)

    #unroll Thetas
    nnParams = n.concatenate((Theta1.ravel(), Theta2.ravel()))
    #manually compute gradients
    numgrad = computeGradients(inputLayerSize,hiddenLayerSize,numLabels,X,y,lmbda,nnParams)

    J = costFunction(nnParams,inputLayerSize,hiddenLayerSize,numLabels,X,y,lmbda)
    #calculate difference
    diff = n.linalg.norm(numgrad - J[1]) / n.linalg.norm(numgrad + J[1])
    print numgrad
    print J[1]


    print diff
        

#------------------------------------------------------------------------------------
#Compute gradient function, essentially calculating derivative
def computeGradients(inputLayerSize,hiddenLayerSize,numLabels,X,y,lmbda,theta):
    e = .0001
    perturb = n.zeros(n.shape(theta))
    numgrad = n.zeros(n.shape(theta))

    for p in range(theta.size):
        perturb[p] = e
        loss1 = costFunction(theta - perturb,inputLayerSize,hiddenLayerSize,numLabels,X,y,lmbda)[0]
        loss2 = costFunction(theta + perturb,inputLayerSize,hiddenLayerSize,numLabels,X,y,lmbda)[0]

        numgrad[p] = (loss2 - loss1) / (2.0 * e)
        perturb[p] = 0
    return numgrad

    


