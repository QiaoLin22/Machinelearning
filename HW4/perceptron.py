import argparse
import numpy as np
import pandas as pd
import time
import perceptron
from abc import ABC


class Perceptron(object):
    mEpoch = 1  # maximum epoch size
    w = None       # weights of the perceptron


    def __init__(self, epoch):
        self.mEpoch = epoch


    def train(self, xFeat, y):
        """
        Train the perceptron using the data

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : object
            Keys represent the epochs and values the number of mistakes
        """
        trainStats = {}
        # TODO implement this
        #xFeat = np.append(xFeat,y,axis=1)
        #mistakes = []
        self.w = [0 for i in range(len(xFeat[0]))]
        for epoch in range(self.mEpoch):
            mistake = 0
            for row, target in zip(xFeat,y):
                if self.helper(row) >= 0:
                    update = target - 1
                else:
                    update = target
                self.w[1:] += update * row[:-1]
                self.w[0] += update
                if int(update) != 0:
                    mistake = mistake + 1
            trainStats[epoch] = mistake
            if mistake == 0:
                break
        # sortweight = sorted(self.w)
        # for i in range(15):
        #     print(sortweight[i])
        #     indexes = [index for index in range(len(self.w)) if self.w[index] == sortweight[i]]
        #     print(indexes)
        return trainStats

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
        for i in range(len(xFeat)):
            if np.dot(xFeat[i],self.w) >= 0:
                yHat.append(1)
            else:
                yHat.append(0)

        return yHat

    def helper(self, row):
        return np.dot(row[:-1], self.w[1:]) + self.w[0]

def calc_mistakes(yHat, yTrue):
    """
    Calculate the number of mistakes
    that the algorithm makes based on the prediction.

    Parameters
    ----------
    yHat : 1-d array or list with shape n
        The predicted label.
    yTrue : 1-d array or list with shape n
        The true label.      

    Returns
    -------
    err : int
        The number of mistakes that are made
    """
    err = 0
    for i in range(len(yHat)):
        if yHat[i] == yTrue[i]:
            err += 0
        else:
            err += 1

    return err


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
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")
    
    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    np.random.seed(args.seed)   
    model = Perceptron(args.epoch)
    trainStats = model.train(xTrain, yTrain)
    print(trainStats)
    yHat = model.predict(xTest)
    # print out the number of mistakes
    print("Number of mistakes on the test dataset")
    print(calc_mistakes(yHat, yTest))


if __name__ == "__main__":
    main()