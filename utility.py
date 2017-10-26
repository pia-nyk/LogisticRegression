
import numpy as np
import pandas as pd #not of your use
import logging
import json

FILE_NAME_TRAIN = 'train.csv' #replace this file name with the train file
FILE_NAME_TEST = 'test.csv' #replace
ALPHA = 2
EPOCHS = 50000
MODEL_FILE = 'models/model8'
train_flag = True

logging.basicConfig(filename='output.log',level=logging.DEBUG)


#utility functions
def loadData(file_name):
    df = pd.read_csv(file_name)
    logging.info("Number of data points in the data set "+str(len(df)))
    y_df = df['output']
    keys = ['company_rating','model_rating', 'bought_at', 'months_used', 'issues_rating','resale_value']
    X_df = df.get(keys)
    return X_df, y_df


def normalizeData(X_df, y_df, model):
    #save the scaling factors so that after prediction the value can be again rescaled
    model['input_scaling_factors'] = [list(X_df.mean()),list(X_df.std())]
    #model['output_scaling_factors'] = [y_df.mean(), y_df.std()]
    X = np.array((X_df-X_df.mean())/X_df.std())
    y = np.array(y_df)
    return X, y, model

def normalizeTestData(X_df, y_df, model):
    meanX = model['input_scaling_factors'][0]
    stdX = model['input_scaling_factors'][1]
    #meany = model['output_scaling_factors'][0]
    #stdy = model['output_scaling_factors'][1]

    X = 1.0*(X_df - meanX)/stdX
    y = y_df

    return X, y


def accuracy(X, y, model):

    y_predicted = predict(X,np.array(model['theta']))
   # acc = np.sqrt(1.0*(np.sum(np.square(y_predicted - y)))/len(X))
    y = 1.0 * y
    #print y
    c=0

    for i in range(0,len(y)):
        if y_predicted[i] > 0.5:
          y_predicted[i] = 1
        else:  
          y_predicted[i] = 0
        if y_predicted[i] == y[i]:
          c=c+1

    #print y_predicted      

    acc = (1.0 * c)/len(y) * 100           

    return acc

def predict(X,theta):
    arr = np.dot(X,theta)
    return 1 / (1.0 + np.exp(-1.0 * arr))  
