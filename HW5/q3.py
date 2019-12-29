import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
xTrain = pd.read_csv("q1xTrain.csv")
yTrain = pd.read_csv("q1yTrain.csv")
xTest = pd.read_csv("q1xTest.csv")
yTest = pd.read_csv("q1yTest.csv")
xTrain = xTrain.to_numpy()
yTrain = yTrain.to_numpy()
xTest = xTest.to_numpy()
yTest = yTest.to_numpy()
yTrain = np.ravel(yTrain)
yTest = np.ravel(yTest)
model=xgb.XGBClassifier(random_state=1,num_round = 10,max_depth=10,learning_rate=0.2)
model.fit(xTrain, yTrain)
print(model.score(xTest,yTest))