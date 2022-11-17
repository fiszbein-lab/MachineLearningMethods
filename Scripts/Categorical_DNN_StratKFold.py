import pandas
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
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
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score


# load dataset
# dataset = loadtxt("/projectnb/evolution/zwakefield/tcga/analysis/classifier/withoutHeaders/fullAFEPSI.txt", delimiter=',')
import numpy as np

def iter_loadtxt(filename, delimiter=',', skiprows=0, dtype=float):
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                for item in line:
                    yield dtype(item)
        iter_loadtxt.rowlength = len(line)

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, iter_loadtxt.rowlength))
    return data

dataset = iter_loadtxt("/projectnb/evolution/zwakefield/tcga/analysis/classifier/withoutHeaders/fullAFEPSI.txt")


size = dataset.shape[1]-1
b_size = round(dataset.shape[0]/10)
X = dataset[:,0:size]
Y = dataset[:,size]
#dummy_y = np_utils.to_categorical(Y)
print("done!")



def unique(list1):
 
    # initialize a null list
    unique_list = []
 
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    # print list
    for x in unique_list:
        print(x)
    return unique_list
classes = len(unique(Y))
print(classes)

cvscores = []
accuracy_sc = []
precision_sc = []
recall_sc = []
f1_sc = []
kappa_sc = []
auc_sc = []
matrix_sc = []
def baseline_model(numOut = classes):
    model = Sequential()
    model.add(Dense(120, input_dim=size, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(.3))
    model.add(Dense(numOut, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("working........")
    return model
 
estimator = KerasClassifier(build_fn=baseline_model, epochs=25, batch_size=b_size, verbose=0)
kfold = StratifiedKFold(n_splits=4, shuffle=True)
print("done!")


scores = ['accuracy', 'recall_weighted', 'precision_weighted', 'f1_weighted', 'roc_auc_ovr_weighted']
full_score = cross_validate(estimator, X, Y, cv=kfold, scoring=scores, return_train_score=False)
#roc_auc = cross_validate(estimator, X, Y, cv=kfold, scoring='roc_auc_weighted')
#cross_val_score
#print("Baseline: %.2f%% (%.2f%%)" % (accuracy.mean()*100, accuracy.std()*100))
#print("Baseline: %.2f%% (%.2f%%)" % (recall.mean()*100, recall.std()*100))
#print("Baseline: %.2f%% (%.2f%%)" % (precision.mean()*100, precision.std()*100))
#print("Baseline: %.2f%% (%.2f%%)" % (f1.mean()*100, f1.std()*100))
#print("Baseline: %.2f%% (%.2f%%)" % (roc_auc.mean()*100, roc_auc.std()*100))
print("done!")


print(full_score)

# print("accuracy %.4f%% (+/- %.4f%%)" % (np.mean(full_score['test_accuracy']), np.std(full_score['test_accuracy'])))
print("recall_weighted %.4f%% (+/- %.4f%%)" % (np.mean(full_score['test_recall_weighted']), np.std(full_score['test_recall_weighted'])))
print("precision_weighted %.4f%% (+/- %.4f%%)" % (np.mean(full_score['test_precision_weighted']), np.std(full_score['test_precision_weighted'])))
print("f1_weighted %.4f%% (+/- %.4f%%)" % (np.mean(full_score['test_f1_weighted']), np.std(full_score['test_f1_weighted'])))
print("roc_roc_ovr_weighted %.4f%% (+/- %.4f%%)" % (np.mean(full_score['test_roc_auc_ovr_weighted']), np.std(full_score['test_roc_auc_ovr_weighted'])))








