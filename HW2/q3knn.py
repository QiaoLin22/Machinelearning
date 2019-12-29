import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import tree
from sklearn.model_selection import KFold
import statistics
from sklearn.neighbors import KNeighborsClassifier
xTrain = pd.read_csv("q4xTrain.csv")
xTest = pd.read_csv("q4xTest.csv")
yTrain = pd.read_csv("q4yTrain.csv")['label']
yTest = pd.read_csv("q4yTest.csv")['label']



def sktree_train_test(model, xTrain, yTrain, xTest, yTest):
    # fit the data to the training dataset
    model.fit(xTrain, yTrain)
    # predict training and testing probabilties
    yHatTrain = model.predict_proba(xTrain)
    yHatTest = model.predict_proba(xTest)
    # calculate auc for training
    fpr, tpr, thresholds = metrics.roc_curve(yTrain['label'],
                                             yHatTrain[:, 1])
    trainAuc = metrics.auc(fpr, tpr)
    # calculate auc for test dataset
    fpr, tpr, thresholds = metrics.roc_curve(yTest['label'],
                                             yHatTest[:, 1])
    testAuc = metrics.auc(fpr, tpr)
    return trainAuc, testAuc

def kfold_cv(model, xFeat, y, k):
    trainAuc = 0
    testAuc = 0
    timeElapsed = 0
    # TODO FILL IN
    start = time.time()
    model = tree.DecisionTreeClassifier()
    trainscore = []
    testscore = []
    kf = KFold(n_splits=k)
    xFeat = xFeat.to_numpy()
    y = y.to_numpy()
    for train_index, test_index in kf.split(xFeat):
        xTrain, xTest, yTrain, yTest = xFeat[train_index], xFeat[test_index], y[train_index], y[test_index]
        model.fit(xTrain, yTrain)
        trainpredict = model.predict(xTrain)
        trainscore.append(roc_auc_score(yTrain, trainpredict))
        testpredict = model.predict(xTest)
        testscore.append(roc_auc_score(yTest, testpredict))
    trainAuc = statistics.mean(trainscore)
    testAuc = statistics.mean(testscore)


    end = time.time()
    timeElapsed = end - start
    return trainAuc, testAuc, timeElapsed
def main():
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--xTrain",
                        default="q4xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q4yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q4xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q4yTest.csv",
                        help="filename for labels associated with the test data")
    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)
    # create the decision tree classifier
    knnClass = KNeighborsClassifier(n_neighbors = 10)
    #print(dtClass.get_params())
    aucTrain2, aucVal2, time2 = kfold_cv(knnClass, xTrain, yTrain, 2)
    aucTrain3, aucVal3, time3 = kfold_cv(knnClass, xTrain, yTrain, 3)
    aucTrain4, aucVal4, time4 = kfold_cv(knnClass, xTrain, yTrain, 4)
    aucTrain5, aucVal5, time5 = kfold_cv(knnClass, xTrain, yTrain, 5)
    aucTrain6, aucVal6, time6 = kfold_cv(knnClass, xTrain, yTrain, 6)
    aucTrain7, aucVal7, time7 = kfold_cv(knnClass, xTrain, yTrain, 7)
    aucTrain8, aucVal8, time8 = kfold_cv(knnClass, xTrain, yTrain, 8)
    aucTrain9, aucVal9, time9 = kfold_cv(knnClass, xTrain, yTrain, 9)
    aucTrain10, aucVal10, time10 = kfold_cv(knnClass, xTrain, yTrain, 10)

    trainAuc, testAuc = sktree_train_test(knnClass, xTrain, yTrain, xTest, yTest)
    perfDF = pd.DataFrame([
                            ['2-fold', aucTrain2, aucVal2, time2],
                           ['3-fold', aucTrain3, aucVal3, time3],
                            ['4-fold', aucTrain4, aucVal4, time4],
                           ['5-fold', aucTrain5, aucVal5, time5],
                            ['6-fold', aucTrain6, aucVal6, time6],
                            ['7-fold', aucTrain7, aucVal7, time7],
                            ['8-fold', aucTrain8, aucVal8, time8],
                            ['9-fold', aucTrain9, aucVal9, time9],
                           ['10-fold', aucTrain10, aucVal10, time10],
                           ['True Test', trainAuc, testAuc, 0]],
                           columns=['Strategy', 'TrainAUC', 'ValAUC', 'Time'])
    print(perfDF)


if __name__ == "__main__":
    main()
