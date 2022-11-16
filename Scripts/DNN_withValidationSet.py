#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## load module tensorflow

from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold


# In[21]:


print(os.getcwd())
seed = 7
# load the dataset
# "brca_class_in.txt"
# "brca_AFE_class_in.txt"
def get_data(path):
    dataset = loadtxt(path, delimiter=',')
    size = dataset.shape[1]-1
    b_size = round(dataset.shape[0]/10)

    # split into input (X) and output (y) variables
    X = dataset[:,0:size]
    Y = dataset[:,size]


    # split into test and train sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)
    return X, Y, size, b_size, X_train, X_test, y_train, y_test


# In[22]:


# define and fit the model
def get_model(X_train, y_train, firstNum = 120, firstDrop = .5, secondNum = 30, secondDrop = .3):
    model = Sequential()
    model.add(Dense(firstNum, input_shape=(size,), activation='relu'))
    model.add(Dropout(firstDrop))
    model.add(Dense(secondNum, activation='relu'))
    model.add(Dropout(secondDrop))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=50, batch_size=200)
    return model
 
# generate data
# fit model
#model = get_model(X_train, y_train)
 
def stats_out(X_test, y_test):
# predict probabilities for test set
    yhat_probs = model.predict(X_test, verbose=0)
    # predict crisp classes for test set
    # yhat_classes = model.predict_classes(X_test, verbose=0)
    predict_x=model.predict(X_test) 
    yhat_classes=np.round(predict_x)
    # reduce to 1d array
    yhat_probs = yhat_probs
    yhat_classes = yhat_classes

    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_test, yhat_classes)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test, yhat_classes)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test, yhat_classes)
    print('F1 score: %f' % f1)

    # kappa
    kappa = cohen_kappa_score(y_test, yhat_classes)
    print('Cohens kappa: %f' % kappa)
    # ROC AUC
    auc = roc_auc_score(y_test, yhat_probs)
    print('ROC AUC: %f' % auc)
    # confusion matrix
    matrix = confusion_matrix(y_test, yhat_classes)
    print(matrix)
    
    

#stats_out(X_test, y_test)
    
    


# In[23]:


del(X_test, y_test, X_train, y_train)
X, Y, size, b_size, X_train, X_test, y_train, y_test = get_data("brca_AFE_class_in.txt")
model = get_model(X_train, y_train)
stats_out(X_test, y_test)


# In[ ]:




