import argparse
import numpy as np
import pandas as pd
import math
import copy


class Knn(object):
    k = 3    # number of neighbors to use
    xFeat = pd.read_csv("q3xTrain.csv")
    y = pd.read_csv("q3yTrain.csv")
    def __init__(self, k):
        """
        Knn constructor

        Parameters
        ----------
        k : int
            Number of neighbors to use.
        """
        self.k = k


    def train(self, xFeat, y):
        """
        Train the k-nn model.

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data
        y : 1d array with shape n
            Array of labels associated with training data.

        Returns
        -------
        self : object
        """
        # TODO do whatever you need

        self.xFeat = xFeat
        self.y = y
        return self


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
            Predicted class label per sample
        """
        #self.xFeat = xFeat
        yHat = [] # variable to store the estimated class label
        # TODO
        D = []
        #pos = []
        vote0 = 0
        vote1 = 0
        for i in range(len(xFeat)):
            for j in range(len(xFeat)):
                if i == j:
                    continue
                else:
                    sqrsum = 0
                    for d in range(len(xFeat.columns)):
                        sqrsum += (xFeat.iloc[j][d] - self.xFeat.iloc[i][d]) ** 2
                distance = math.sqrt(sqrsum)
                D.append(distance)
            S = sorted(D)
            for k in range(self.k):
                res = D.index(S[k])
                if self.y.iloc[res] == 0.0:
                    vote0 += 1
                elif self.y.iloc[res] == 1.0:
                    vote1 += 1
            if vote0 >= vote1:
                yHat.append(0.0)
            else:
                yHat.append(1.0)
            #pos.append(res)
            #yHat.append(self.y.iloc[res][0])
            D = []
            vote0 = 0.0;
            vote1 = 0.0;


        return yHat


def accuracy(yHat, yTrue):
    """
    Calculate the accuracy of the prediction

    Parameters
    ----------
    yHat : 1d-array with shape n
        Predicted class label for n samples
    yTrue : 1d-array with shape n
        True labels associated with the n samples

    Returns
    -------
    acc : float between [0,1]
        The accuracy of the model
    """
    # TODO calculate the accuracy
    acc = 0
    correct = 0
    for i in range(len(yHat)):
        if yHat[i] == yTrue[i]:
            correct += 1
    acc = correct / len(yTrue)
    return acc


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
    parser.add_argument("--xTrain",
                        default="q3xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q3yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q3xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q3yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)
    # create an instance of the model
    knn = Knn(args.k)
    knn.train(xTrain, yTrain['label'])
    # predict the training dataset
    yHatTrain = knn.predict(xTrain)
    trainAcc = accuracy(yHatTrain, yTrain['label'])
    # predict the test dataset
    yHatTest = knn.predict(xTest)
    testAcc = accuracy(yHatTest, yTest['label'])
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)


if __name__ == "__main__":
   main()

#k2 = []
#k2 = (Knn.predict(Knn,Knn.xFeat))
#ytest = pd.read_csv("q3yTrain.csv")
#print(accuracy(yHat=k2,yTrue=ytest['label']))








