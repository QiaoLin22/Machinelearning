import argparse
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0, solver='lbfgs',max_iter=1000,multi_class='multinomial')
def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()
def calc_mistakes(yHat, yTrue):
    err = 0
    for i in range(len(yHat)):
        if yHat[i] == yTrue[i]:
            err += 0
        else:
            err += 1
    return err

xTrainbinary = pd.read_csv("xTrainbinary.csv")
xTrainbinary = file_to_numpy("xTrainbinary.csv")
xTraincount = pd.read_csv("xTraincount.csv")
xTraincount = file_to_numpy("xTraincount.csv")
xTraintfidf = pd.read_csv("xTraintfidf.csv")
xTraintfidf = file_to_numpy("xTraintfidf.csv")
yTrain = pd.read_csv("yTrain.csv")
yTrain = file_to_numpy("yTrain.csv")
yTrain = np.ravel(yTrain)
xTestbinary = pd.read_csv("xTestbinary.csv")
xTestbinary = file_to_numpy("xTestbinary.csv")
xTestcount = pd.read_csv("xTestcount.csv")
xTestcount = file_to_numpy("xTestcount.csv")
xTesttfidf = pd.read_csv("xTesttfidf.csv")
xTesttfidf = file_to_numpy("xTesttfidf.csv")
yTest = pd.read_csv("yTest.csv")
yTest = file_to_numpy("yTest.csv")

model.fit(xTraintfidf, yTrain)
#model.fit(xTraincount, yTrain)
#model.fit(xTraintfidf, yTrain)

predicted = model.predict(xTesttfidf)
print(calc_mistakes(predicted,yTest))

print(metrics.classification_report(yTest, predicted))
