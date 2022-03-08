import numpy as n
from sigmoid import sigmoid, sigmoidGradient

#-------------------------------------------------------------------------------
#Cost function using feedforwarding and backpropogation for a neural network, also regularized
def costFunction(nnParams,inputLayerSize,hiddenLayerSize,numLabels,X,y,lmbda):

    #roll thetas
    Theta1 = n.reshape(nnParams[ :hiddenLayerSize * (inputLayerSize + 1)], (hiddenLayerSize, inputLayerSize + 1))
    Theta2 = n.reshape(nnParams[hiddenLayerSize * (inputLayerSize + 1): ], (numLabels, hiddenLayerSize + 1))

    #definitions:
    m = X.shape[0]

    #-----------------------------------------------------------------------------------------
    #ALGORITHMS:

    #feedforward:
    y_matrix = n.eye(numLabels)[y-1,:][0]
    a1 = n.hstack((n.ones((m,1),dtype = float),X))
    a1 = n.array(a1, dtype = float)
    z2 = n.dot(a1, n.transpose(Theta1))
    a2 = n.hstack((n.ones((m,1),dtype = float),sigmoid(z2)))
    z3 = n.dot(a2, n.transpose(Theta2))
    a3 = sigmoid(z3)
    h = a3

    #Cost function calculation:
    J = (1.0 / m) * n.sum(n.sum(n.multiply(-y_matrix,n.log(h)) - n.multiply(1 - y_matrix, n.log(1 - h))))
    J += (lmbda / (2.0 * m)) * (n.sum(n.sum(n.power(Theta1[:,1:],2))) + n.sum(n.sum(n.power(Theta2[:,1:],2))))    

    #Backpropogation:
    d3 = (a3 - y_matrix)
    d2 = n.multiply(n.dot(d3, Theta2[:,1:]), sigmoidGradient(z2))
    
    D1 = n.dot(n.transpose(d2),a1)
    D2 = n.dot(n.transpose(d3),a2)

    #Calculate gradients:
    Theta1_grad = D1 / m
    Theta2_grad = D2 / m

    Theta1[:,0] = 0
    Theta2[:,0] = 0

    Theta1_grad += Theta1 * (lmbda / m)
    Theta2_grad += Theta2 * (lmbda / m)

    #Unroll Gradients:
    grad = n.concatenate((Theta1_grad.ravel(),Theta2_grad.ravel()))

    print "cost: \n"
    print J
    print " \n Gradients: \n"
    print grad 
    return J, grad
