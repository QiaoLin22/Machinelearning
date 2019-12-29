import argparse
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

class RandomForest(object):
    nest = 11    # number of trees
    maxFeat = 3        # maximum number of features
    maxDepth = 5       # maximum depth of the decision tree
    minLeafSample = 1  # minimum number of samples in a leaf
    criterion = 'entropy'  # splitting criterion


    def __init__(self, nest, maxFeat, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor

        Parameters
        ----------
        nest: int
            Number of trees to have in the forest
        maxFeat: int
            Maximum number of features to consider in each tree
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int 
            Maximum depth of the decision tree
        minLeafSample : int 
            Minimum number of samples in the decision tree
        """
        self.nest = nest
        self.maxFeat = maxFeat
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample

    def train(self, xFeat, y):
        """
        Train the random forest using the data

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : object
            Keys represent the number of trees and
            the values are the out of bag errors
        """
        sample = []
        featurelist = []
        clflist = []
        stats = {}
        np.random.seed(334)
        for i in range(self.nest):
            sampleindex = np.random.randint(1119,size=707)
            sample.append(sampleindex)
            featureindex = np.random.choice(11, size=self.maxFeat,replace=False)
            featurelist.append(featureindex)
            dtsample = xFeat[sampleindex,:]
            dtsample = dtsample[:,featureindex]
            dtsampley = y[sampleindex,:]
            clf = DecisionTreeClassifier(criterion='entropy',max_depth =self.maxDepth,min_samples_leaf=
                                    self.minLeafSample,random_state=0)
            clf.fit(dtsample,dtsampley)
            clflist.append(clf)
        oobpredict = []
        oobtrue = []
        for i in range(len(xFeat)):
            predict = []
            for j in range(self.nest):
                if i not in sample[j]:
                    oobfeature = featurelist[j]
                    oobsample = xFeat[i][oobfeature]
                    oobsample = oobsample.reshape(1,-1)
                    predicty = clflist[j].predict(oobsample)
                    predict.append(predicty)
            if len(predict) == 0:
                continue
            else:
                if sum(predict) == 0:
                    oobpredict.append(0)
                    oobtrue.append(y[i])
                elif(sum(predict) / len(predict)) < 0.5:
                    oobpredict.append(0)
                    oobtrue.append(y[i])
                else:
                    oobpredict.append(1)
                    oobtrue.append(y[i])

        correct = 0
        for i in range(len(oobpredict)):
            if oobpredict[i] == oobtrue[i]:
                correct += 1
        ooberr = (len(oobpredict) - correct) / len(oobpredict)
        stats[self.nest] = ooberr
        return stats

    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict 
        what class the values will have.

        Parameters
        ----------
        xFeat : nd-array with shape m x d
            The data to predict.  

        Returns
        -------
        yHat : 1d array or list with shape m
            Predicted response per sample
        """
        yHat = []
        sample = []
        featurelist = []
        clflist = []
        np.random.seed(334)
        y = pd.read_csv("q1yTrain.csv")
        y = y.to_numpy()
        for i in range(self.nest):
            sampleindex = np.random.randint(480, size=298)
            sample.append(sampleindex)
            featureindex = np.random.choice(11, size=self.maxFeat, replace=False)
            featurelist.append(featureindex)
            dtsample = xFeat[sampleindex, :]
            dtsample = dtsample[:, featureindex]
            dtsampley = y[sampleindex, :]
            clf = DecisionTreeClassifier(criterion='entropy', max_depth=self.maxDepth, min_samples_leaf=
            self.minLeafSample, random_state=0)
            clf.fit(dtsample, dtsampley)
            clflist.append(clf)
        ytrue = []
        for i in range(len(xFeat)):
            predict = []
            for j in range(self.nest):
                if i not in sample[j]:
                    oobfeature = featurelist[j]
                    oobsample = xFeat[i][oobfeature]
                    oobsample = oobsample.reshape(1,-1)
                    predicty = clflist[j].predict(oobsample)
                    predict.append(predicty)
            if len(predict) == 0:
                continue
            else:
                if sum(predict) == 0:
                    yHat.append(0)
                    ytrue.append(y[i])
                elif(sum(predict) / len(predict)) < 0.5:
                    yHat.append(0)
                    ytrue.append(y[i])
                else:
                    yHat.append(1)
                    ytrue.append(y[i])

        return yHat,ytrue


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("xTrain",
                        help="filename for features of the training data")
    parser.add_argument("yTrain",
                        help="filename for labels associated with training data")
    parser.add_argument("xTest",
                        help="filename for features of the test data")
    parser.add_argument("yTest",
                        help="filename for labels associated with the test data")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")
    
    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    np.random.seed(args.seed)

    model = RandomForest(RandomForest.nest,RandomForest.maxFeat,RandomForest.criterion,RandomForest.maxDepth
                         ,RandomForest.minLeafSample)
    trainStats = model.train(xTrain, yTrain)
    print(trainStats)
    yHat,ytrue = model.predict(xTest)
    print(accuracy_score(ytrue,yHat))



if __name__ == "__main__":
    main()