 
import numpy as np 
import logging
import json
from utility import *

FILE_NAME_TRAIN = 'train.csv' #replace this file name with the train file
FILE_NAME_TEST = 'test.csv' #replace
ALPHA = 2.5
EPOCHS = 70000#keep this greater than or equl to 5000 strictly otherwise you will get an error
MODEL_FILE = 'models/model8'
train_flag = False
logging.basicConfig(filename='output.log',level=logging.DEBUG)

np.set_printoptions(suppress=True)

def appendIntercept(X):
    #steps
    #make a column vector of ones
    #stack this column vector infront of the main X vector using hstack
    #return the new matrix
    #remove this line once you finish writing
    n = np.shape(X)
    col = np.ones((n[0],1))
    new_arr = np.hstack((col,X))
    return new_arr

def initialGuess(n_thetas):
 row = np.zeros((n_thetas,))
 return row


def train(theta, X, y, model):
     J = [] #this array should contain the cost for every iteration so that you can visualize it later when you plot it vs the ith iteration
     #train for the number of epochs you have defined
     m = len(y)
     #your  gradient descent code goes here
     #steps
     #run you gd loop for EPOCHS that you have defined
        #calculate the predicted y using your current value of theta
        # calculate cost with that current theta using the costFunc function
        #append the above cost in J
        #calculate your gradients values using calcGradients function
        # update the theta using makeGradientUpdate function (don't make a new variable assign it back to theta that you received)
     for i in range(0,EPOCHS):
      y_p = predict(X,theta)
      cost = costFunc(m,y,y_p)
      J.append(cost)	
      grads = calcGradients(X,y,y_p,m)
      theta = makeGradientUpdate(theta,grads)
     print theta
     model['J'] = J
     model['theta'] = list(theta)
     return model


def predict(X,theta):
    arr = np.dot(X,theta.T)
    return 1.0 / (1.0 + np.exp(-1.0 * arr))

def costFunc(m,y,y_predicted):
    J = np.multiply(y,np.log(y_predicted)) + np.multiply((1 - y),np.log(1 - y_predicted))
    J = np.sum(J)
    J/=m
    J = (-1.0)*J
    return J

def calcGradients(X,y,y_predicted,m):
    new_arr = np.multiply(np.transpose(X),np.subtract(y_predicted,y))/m
    new_arr = np.sum(new_arr,axis=1) 

    return np.transpose(new_arr)

def makeGradientUpdate(theta, grads):
    theta = np.subtract(theta,ALPHA*grads)
    return theta


########################main function###########################################
def main():
    if(train_flag):
        model = {}
        X_df,y_df = loadData(FILE_NAME_TRAIN)
        X,y, model = normalizeData(X_df, y_df, model)
        X = appendIntercept(X)
        theta = initialGuess(X.shape[1])
        model = train(theta, X, y, model)
        with open(MODEL_FILE,'w') as f:
            f.write(json.dumps(model))
        print "Accuracy :",accuracy(X,y,model)

    else:
        model = {}
        with open(MODEL_FILE,'r') as f:
            model = json.loads(f.read())
            X_df, y_df = loadData(FILE_NAME_TEST)
            X,y = normalizeTestData(X_df, y_df, model)
            X = appendIntercept(X)
            print "Accuracy: ",accuracy(X,y,model)

if __name__ == '__main__':
    main()           



