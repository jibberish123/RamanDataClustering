import numpy as n
import random
import scipy as s
from scipy import optimize
from matplotlib import pyplot
import csv
from Convert import convert
from YConvert import convertY
from sigmoid import sigmoid, sigmoidGradient
from randInitializeWeights import initializeRandWeights
from costFunction import costFunction
from checkGradients import checkGradients
from predict import predict
from featureNormalize import featureNormalize
from sklearn import model_selection


#-----------------------------------------------
#Data conversion from csv to numpy.arrays

X,y_name = convert('RamanData.csv')

#Convert labels to numerical logistic classification(aka: E Coli --> 4)

y, num_labels = convertY(y_name)
y = y[0].astype(int)

#----------------------------------------------------------
#Normalize X
X,mu,sigma = featureNormalize(X)



#---------------------------------------------------
print(n.shape(X))

xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.33,shuffle=True)



#definitions:
shapeX = n.shape(X)
inputLayerSize = shapeX[1]
hiddenLayerSize = 50
m = shapeX[0]


#--------------------------------------------------------------------------
#initializing random thetas for both Theta1 and Theta2 using initializeRandWeights
initialTheta1 = initializeRandWeights(inputLayerSize, hiddenLayerSize)
initialTheta2 = initializeRandWeights(hiddenLayerSize, num_labels)


print (initialTheta1.shape())
print (initialTheta2.shape())
#unrolling the Thetas into a long array
nnParams = n.concatenate((initialTheta1.ravel(), initialTheta2.ravel()))
#set regularization parameter lambda
lmbda = 1

#--------------------------------------- ------------------------------------
#Manual calculations of gradients to verify gradients, only run to verify, otherwise very costly
#checkGradients(lmbda)


def costFunc(nnparams):
    return costFunction(nnparams,inputLayerSize,hiddenLayerSize,num_labels,X,y,lmbda)[0]
def gradFunc(nnparams):
    return costFunction(nnparams,inputLayerSize,hiddenLayerSize,num_labels,X,y,lmbda)[1]

#-----------------------------------------------------------------------------------------------------
#minimization function
optimize = optimize.fmin_cg(costFunc,nnParams,fprime = gradFunc,full_output = True,maxiter = 2000)
nnparams = optimize[0]
cost = optimize[1]

#Roll Thetas from optimized
Theta1 = n.reshape(nnparams[ :hiddenLayerSize * (inputLayerSize + 1)], (hiddenLayerSize, inputLayerSize + 1))
Theta2 = n.reshape(nnparams[hiddenLayerSize * (inputLayerSize + 1): ], (num_labels, hiddenLayerSize + 1))

#-----------------------------------------------------------------------
#Predict

pred = predict(Theta1,Theta2,X)

#Display training accuracy
print ('\n Training Set Accuracy: ')
print (n.mean((pred == y) * 100,dtype = float))
