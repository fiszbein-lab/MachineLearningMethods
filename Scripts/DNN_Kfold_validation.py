#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# load pima indians dataset

dataset = np.loadtxt("brca_AFE_class_in.txt", delimiter=",")
size = dataset.shape[1]-1
b_size = round(dataset.shape[0]/10)
# split into input (X) and output (Y) variables
X = dataset[:,0:size]
Y = dataset[:,size]
# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []


# In[5]:


for train, test in kfold.split(X, Y):
# create model
    model = Sequential()
    model.add(Dense(70, input_shape=(size,), activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(30, activation='relu'))
    model.add(Dropout(.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    # Fit the model
    train = model.fit(X[train], Y[train], epochs=20, batch_size=b_size) # , verbose=0
    # evaluate the model
    scores = model.evaluate(X[test], Y[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


# In[ ]:




