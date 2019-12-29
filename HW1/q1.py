import random
import pandas as pd
import numpy as np
import statistics
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import time
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import roc_auc_score
xTrain = pd.read_csv("q4xTrain.csv")
xTest = pd.read_csv("q4xTest.csv")
yTrain = pd.read_csv("q4yTrain.csv")['label']
yTest = pd.read_csv("q4yTest.csv")['label']
Train = pd.concat([xTrain, yTrain], axis=1)
print(len(Train))
Train99 = Train.sample(frac=0.99)
print(len(Train99))
