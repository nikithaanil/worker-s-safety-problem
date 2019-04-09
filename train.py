from __future__ import print_function

import enum
import re
import struct
import sys
import threading
import time
import math
import pywt
import itertools
import pandas as pd
import numpy as np
import pickle
import string
import serial
import datetime

from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_validate
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import svm, grid_search
from sklearn import cross_validation

from serial.tools.list_ports import comports
from numpy import array
from common import *



seed = 65  #gives an accuracy of 76%
df = pd.read_csv("./inputData/s_data1.csv")
X = df.loc[:, df.columns != "0c"]
X = X.loc[:, X.columns != "57c"] #label name dropped
y = df["58c"] #labels seperated
X = X.loc[:, X.columns != '58c']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.5,random_state=seed)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)
# parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
clf = svm.SVC(kernel='linear',C=8.0,gamma='auto')
clf.fit(X_train, y_train)
predict = clf.predict(X_test)
y_pred = predict
y_true = y_test
print(accuracy_score(y_true, y_pred))
