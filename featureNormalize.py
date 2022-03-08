import numpy as n



#Normalize features mean 0 standard deviation 1
def featureNormalize(X):
    X = n.array(X,dtype = float)
    mu = n.ones(X.shape,dtype = float) * n.mean(X)
    Xnorm = X - mu
    sigma = n.ones(Xnorm.shape,dtype = float) * n.std(Xnorm)
    Xnorm = Xnorm / sigma
    
    return Xnorm,mu,sigma
    



