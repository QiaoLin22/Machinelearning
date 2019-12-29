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
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
xTrain = pd.read_csv("q4xTrain.csv")
xTest = pd.read_csv("q4xTest.csv")
yTrain = pd.read_csv("q4yTrain.csv")['label']
yTest = pd.read_csv("q4yTest.csv")['label']
knnClass = KNeighborsClassifier(n_neighbors = 8)

knnClass.fit(xTrain, yTrain)
trainpredict = knnClass.predict(xTest)
trainAuc = roc_auc_score(yTest,trainpredict)
trainscore = accuracy_score(yTest,trainpredict)

Train = pd.concat([xTrain, yTrain], axis=1)
Train99 = Train.sample(frac=0.99)
Train95 = Train.sample(frac=0.95)
Train90 = Train.sample(frac=0.90)
yTrain99 = Train99[Train99.columns[-1]]
xTrain99 = Train99.drop(Train99.columns[-1],axis=1)
knnClass.fit(xTrain99, yTrain99)
trainpredict99 = knnClass.predict(xTest)
trainAuc99 = roc_auc_score(yTest,trainpredict99)
trainscore99 = accuracy_score(yTest,trainpredict99)

yTrain95 = Train95[Train95.columns[-1]]
xTrain95 = Train95.drop(Train95.columns[-1],axis=1)
knnClass.fit(xTrain95, yTrain95)
trainpredict95 = knnClass.predict(xTest)
trainAuc95 = roc_auc_score(yTest,trainpredict95)
trainscore95 = accuracy_score(yTest,trainpredict95)

yTrain90 = Train90[Train90.columns[-1]]
xTrain90 = Train90.drop(Train90.columns[-1],axis=1)
knnClass.fit(xTrain90, yTrain90)
trainpredict90 = knnClass.predict(xTest)
trainAuc90 = roc_auc_score(yTest,trainpredict90)
trainscore90 = accuracy_score(yTest,trainpredict90)
print("full data:")
print(trainAuc)
print(trainscore)
print("drop 1%:")
print(trainAuc99)
print(trainscore99)
print("drop 5%:")
print(trainAuc95)
print(trainscore95)
print("drop 10%:")
print(trainAuc90)
print(trainscore90)
print("DT")
dtClass = DecisionTreeClassifier(max_depth=10,
                                     min_samples_leaf=10)

dtClass.fit(xTrain, yTrain)
trainpredict = knnClass.predict(xTest)
trainAuc = roc_auc_score(yTest,trainpredict)
trainscore = accuracy_score(yTest,trainpredict)

Train = pd.concat([xTrain, yTrain], axis=1)
Train99 = Train.sample(frac=0.99)
Train95 = Train.sample(frac=0.95)
Train90 = Train.sample(frac=0.90)
yTrain99 = Train99[Train99.columns[-1]]
xTrain99 = Train99.drop(Train99.columns[-1],axis=1)
dtClass.fit(xTrain99, yTrain99)
trainpredict99 = dtClass.predict(xTest)
trainAuc99 = roc_auc_score(yTest,trainpredict99)
trainscore99 = accuracy_score(yTest,trainpredict99)

yTrain95 = Train95[Train95.columns[-1]]
xTrain95 = Train95.drop(Train95.columns[-1],axis=1)
dtClass.fit(xTrain95, yTrain95)
trainpredict95 = dtClass.predict(xTest)
trainAuc95 = roc_auc_score(yTest,trainpredict95)
trainscore95 = accuracy_score(yTest,trainpredict95)

yTrain90 = Train90[Train90.columns[-1]]
xTrain90 = Train90.drop(Train90.columns[-1],axis=1)
dtClass.fit(xTrain90, yTrain90)
trainpredict90 = dtClass.predict(xTest)
trainAuc90 = roc_auc_score(yTest,trainpredict90)
trainscore90 = accuracy_score(yTest,trainpredict90)
print("full data:")
print(trainAuc)
print(trainscore)
print("drop 1%:")
print(trainAuc99)
print(trainscore99)
print("drop 5%:")
print(trainAuc95)
print(trainscore95)
print("drop 10%:")
print(trainAuc90)
print(trainscore90)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# x = [1,1,1,7,7,7,9,9,9]
# y = [1,5,10,1,5,10,1,5,10]
# z = [86.416,86.416,86.416,87.67,87.04,85.79,87.04,86.15,85.79]
x = [1,1,1,7,7,7,9,9,9]
y = [1,5,10,1,5,10,1,5,10]
z = [86.458,86.458,86.458,86.67,85.625,85.21,86.25,85,85.21]


ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
