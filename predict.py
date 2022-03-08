import numpy as n

from sigmoid import sigmoid


#--------------------------------------------------------------------
#Predict function
def predict(Theta1, Theta2, X):
    m = X.shape[0]
    numLabels = Theta2.shape[0]

    Theta1 = n.array(Theta1,dtype = float)
    Theta2 = n.array(Theta2,dtype = float)
    X = n.array(X, dtype = float)
    


    h1 = sigmoid(n.dot(n.hstack((n.ones((m,1),dtype = float),X)),n.transpose(Theta1)))
    h2 = sigmoid(n.dot(n.hstack((n.ones((m,1),dtype = float), h1)),n.transpose(Theta2)))
    p = n.argmax(h2,axis = 1) + 1
    p = p.astype(int)
    print p
    return p
    

